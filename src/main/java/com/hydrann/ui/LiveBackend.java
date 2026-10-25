package com.hydrann.ui;

import com.hydrann.annheads.BruteForceHead;
import com.hydrann.annheads.Neighbor;
import com.hydrann.controller.HydraConfig;
import com.hydrann.controller.HydraController;
import com.hydrann.controller.HydraStatus;
import com.hydrann.controller.IndexManager;
import com.hydrann.embedding.ChunkEmbedder;
import com.hydrann.embedding.MiniLMEmbedder;
import com.hydrann.ingestion.ChunkEmbedding;
import com.hydrann.ingestion.PdfDirectoryLoader;
import com.hydrann.ingestion.PdfExtractor;
import com.hydrann.ingestion.SemanticChunker;
import com.hydrann.ingestion.TextChunk;
import com.hydrann.persistence.DocumentManifest;
import com.hydrann.persistence.IncrementalIngestionService;
import com.hydrann.persistence.MetadataFile;
import com.hydrann.persistence.PersistentVectorStore;
import com.hydrann.persistence.VectorMetadata;
import com.hydrann.vectorstore.SearchResult;

import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.function.Consumer;

/** Wires the real pipeline (PDFs, MiniLM, persistent store, Hydra controller) behind {@link SearchBackend}. */
public final class LiveBackend implements SearchBackend {

    private static final int DIM = 384;
    private static final Path RAW_DOCS = Path.of("data/raw-docs");
    private static final Path INDEX_DIR = Path.of("data/persisted-index");
    private static final long INFO_TTL_MS = 2_000;

    private record Snapshot(long at, int documents, int active, int stored, List<DocInfo> docs) {}

    private final MiniLMEmbedder embedder;
    private final PersistentVectorStore store;
    private final IndexManager manager;
    private final BruteForceHead exactHead;
    private final HydraController hydra;
    private final IncrementalIngestionService ingestion;
    private volatile Snapshot snapshot;

    private LiveBackend(MiniLMEmbedder embedder, PersistentVectorStore store, IndexManager manager,
                        HydraController hydra, IncrementalIngestionService ingestion) {
        this.embedder = embedder;
        this.store = store;
        this.manager = manager;
        this.exactHead = new BruteForceHead(store);
        this.hydra = hydra;
        this.ingestion = ingestion;
    }

    /** Loads the model, syncs data/raw-docs with the persisted index, and starts the Hydra controller. */
    public static LiveBackend open(Consumer<String> progress) throws Exception {
        MiniLMEmbedder embedder = null;
        PersistentVectorStore store = null;
        HydraController hydra = null;
        try {
            progress.accept("Loading the embedding model...");
            embedder = new MiniLMEmbedder();

            progress.accept("Opening the vector store...");
            store = new PersistentVectorStore(INDEX_DIR, DIM, new MetadataFile());

            progress.accept("Reading your PDFs...");
            IncrementalIngestionService ingestion = new IncrementalIngestionService(
                    RAW_DOCS,
                    new PdfDirectoryLoader(),
                    new PdfExtractor(),
                    new SemanticChunker(embedder, 0.45),
                    new ChunkEmbedder(embedder),
                    store,
                    new DocumentManifest(INDEX_DIR.resolve("documents.json")));
            ingestion.synchronize();

            progress.accept("Building the search index...");
            IndexManager manager = new IndexManager(store, new BruteForceHead(store));

            HydraConfig cfg = HydraConfig.defaults();
            cfg.dim = DIM;
            cfg.k = 5;
            cfg.sampleEvery = 1;       // people type few queries, so check every one
            cfg.minSamples = 5;
            cfg.windowSize = 20;
            cfg.cooldownMs = 30_000;

            hydra = new HydraController(manager, store, cfg, HydraController.defaultCandidates(cfg));
            hydra.start();
            return new LiveBackend(embedder, store, manager, hydra, ingestion);
        } catch (Exception e) {
            if (hydra != null) hydra.close();
            closeQuietly(store);
            closeQuietly(embedder);
            throw e;
        }
    }

    // ------------------------------------------------------------------ SearchBackend

    @Override
    public Outcome search(String query, int k, boolean exact) throws Exception {
        long t0 = System.nanoTime();
        float[] q = embedder.embed(query);

        List<Hit> hits = new ArrayList<>();
        String served;
        if (exact) {
            for (Neighbor n : exactHead.search(q, k)) hits.add(toHit(store.get(n.id()), n.score()));
            served = "Exact scan";
        } else {
            for (SearchResult r : manager.search(q, k)) hits.add(toHit(r.getChunkEmbedding(), r.getScore()));
            served = display(typeOf(manager.activeHead().name()));
        }
        double ms = (System.nanoTime() - t0) / 1_000_000.0;
        return new Outcome(hits, ms, served);
    }

    @Override
    public IndexInfo indexInfo() {
        Snapshot s = snapshot();
        String name = manager.activeHead().name();
        return new IndexInfo(s.documents(), s.active(), s.stored(), display(typeOf(name)), name,
                "MiniLM (" + DIM + "d)");
    }

    @Override
    public List<DocInfo> documents() {
        return snapshot().docs();
    }

    @Override
    public HydraInfo hydra() {
        HydraStatus s = hydra.status();
        String name = s.activeHead();
        return new HydraInfo(s.phase().name(), s.phaseNote(), name, typeOf(name),
                s.activeVectors(), s.storedVectors(), s.samples(), s.recall(),
                s.avgLatencyUs(), s.p95LatencyUs(), s.bruteAvgUs(), s.speedup(),
                s.memoryBytes(), s.activeSinceMs(), s.promotions(), s.competitions(), s.tunings(),
                s.lastDecision(), s.lastDecisionAtMs(), s.scoreboard(), s.recentEvents());
    }

    @Override
    public void rescan() throws Exception {
        ingestion.synchronize();
        manager.catchUp();
        snapshot = null;
    }

    @Override
    public void setEventListener(Consumer<String> listener) {
        hydra.setListener(listener);
    }

    @Override
    public String libraryFolder() {
        return RAW_DOCS.toAbsolutePath().normalize().toString();
    }

    @Override
    public void close() {
        hydra.close();
        closeQuietly(store);
        closeQuietly(embedder);
    }

    // ------------------------------------------------------------------ helpers

    private Hit toHit(ChunkEmbedding ce, double score) {
        TextChunk c = ce.getChunk();
        String path = c.getDocumentId();
        String file = TextTools.fileName(path);
        String text = c.getText();
        return new Hit(TextTools.deriveTitle(text, file), file, path,
                asInt(c.getPageNumber()), asInt(c.getChunkIndex()), score, text);
    }

    private static int asInt(Object o) {
        try {
            return Integer.parseInt(String.valueOf(o));
        } catch (NumberFormatException e) {
            return 0;
        }
    }

    /** Documents, passages and pages are counted from the store's metadata, cached for two seconds. */
    private Snapshot snapshot() {
        Snapshot s = snapshot;
        long now = System.currentTimeMillis();
        if (s != null && now - s.at() < INFO_TTL_MS) return s;

        List<VectorMetadata> meta = store.getMetadataSnapshot();
        Map<String, Integer> chunks = new LinkedHashMap<>();
        Map<String, Set<String>> pages = new HashMap<>();
        int active = 0;
        for (VectorMetadata r : meta) {
            if (!r.active()) continue;
            active++;
            String path = r.documentPath();
            chunks.merge(path, 1, Integer::sum);
            pages.computeIfAbsent(path, p -> new HashSet<>()).add(String.valueOf(r.pageNumber()));
        }
        List<DocInfo> docs = new ArrayList<>();
        for (Map.Entry<String, Integer> e : chunks.entrySet()) {
            docs.add(new DocInfo(TextTools.fileName(e.getKey()), e.getKey(), e.getValue(), pages.get(e.getKey()).size()));
        }
        docs.sort(Comparator.comparing(DocInfo::fileName, String.CASE_INSENSITIVE_ORDER));

        s = new Snapshot(now, docs.size(), active, meta.size(), docs);
        snapshot = s;
        return s;
    }

    /** "hnsw(M=16,efC=100,efS=50)" becomes "hnsw". */
    private static String typeOf(String headName) {
        int i = headName.indexOf('(');
        return i < 0 ? headName : headName.substring(0, i);
    }

    private static String display(String type) {
        return switch (type.toLowerCase()) {
            case "brute-force" -> "Brute force";
            case "lsh" -> "LSH";
            case "hnsw" -> "HNSW";
            case "ivf" -> "IVF";
            default -> type;
        };
    }

    private static void closeQuietly(AutoCloseable c) {
        if (c == null) return;
        try {
            c.close();
        } catch (Exception ignored) {
            // nothing useful to do while shutting down
        }
    }
}
