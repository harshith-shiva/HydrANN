package com.hydrann;

import com.hydrann.annheads.BruteForceHead;
import com.hydrann.controller.HydraConfig;
import com.hydrann.controller.HydraController;
import com.hydrann.controller.IndexManager;
import com.hydrann.embedding.ChunkEmbedder;
import com.hydrann.embedding.MiniLMEmbedder;
import com.hydrann.evaluation.SyntheticCorpus;
import com.hydrann.ingestion.PdfDirectoryLoader;
import com.hydrann.ingestion.PdfExtractor;
import com.hydrann.ingestion.SemanticChunker;
import com.hydrann.ingestion.TextChunk;
import com.hydrann.persistence.DocumentManifest;
import com.hydrann.persistence.IncrementalIngestionService;
import com.hydrann.persistence.MetadataFile;
import com.hydrann.persistence.PersistentVectorStore;
import com.hydrann.vectorstore.SearchResult;
import com.hydrann.vectorstore.SyntheticVectorStore;

import java.nio.file.Path;
import java.util.List;
import java.util.Scanner;
import java.util.concurrent.atomic.AtomicBoolean;

/**
 * Usage: HydraDemo [synthetic|pdf] [auto]
 *
 *   synthetic  generated clustered vectors; commands: :status  :grow N  :ghost F  exit
 *   auto       (synthetic only) grows the corpus in stages and prints a status line every 5 s
 *   pdf        your real PDFs; type queries, ':status', or 'exit'
 */
public class HydraDemo {

    private static final int DIM = 384;
    private static final int K = 5;

    public static void main(String[] args) throws Exception {
        String mode = "synthetic";
        boolean auto = false;
        for (String a : args) {
            switch (a.toLowerCase()) {
                case "synthetic", "pdf" -> mode = a.toLowerCase();
                case "auto" -> auto = true;
                default -> System.out.println("Ignoring unknown argument: " + a);
            }
        }
        if (mode.equals("pdf")) runPdf(); else runSynthetic(auto);
    }

    // ------------------------------------------------------------------
    // Synthetic mode
    // ------------------------------------------------------------------

    private static void runSynthetic(boolean auto) throws Exception {
        SyntheticCorpus corpus = new SyntheticCorpus(DIM, 40, 0.04, 7L);
        SyntheticVectorStore store = new SyntheticVectorStore();
        store.append(corpus.generate(500));

        IndexManager manager = new IndexManager(store, new BruteForceHead(store));

        HydraConfig cfg = HydraConfig.defaults();
        cfg.dim = DIM;
        cfg.k = K;
        cfg.sampleEvery = 3;
        cfg.cooldownMs = 15_000;
        cfg.latencyBudgetUs = 3_000;

        try (HydraController hydra = new HydraController(
                manager, store, cfg, HydraController.defaultCandidates(cfg));
             Scanner scanner = new Scanner(System.in)) {

            hydra.setListener(msg -> System.out.println("\n[HYDRA] " + msg));
            hydra.start();

            AtomicBoolean running = new AtomicBoolean(true);

            // steady query traffic, ~25 queries per second
            startDaemon("demo-queries", () -> {
                while (running.get()) {
                    try {
                        manager.search(corpus.next(), K);
                        Thread.sleep(40);
                    } catch (InterruptedException e) {
                        return;
                    } catch (RuntimeException ignored) { }
                }
            });

            if (auto) {
                startDaemon("demo-growth", () -> {
                    int[] steps = {1_500, 6_000, 12_000, 10_000};   // 500 -> 2k -> 8k -> 20k -> 30k
                    for (int step : steps) {
                        if (!sleepSeconds(45, running)) return;
                        store.append(corpus.generate(step));
                        manager.catchUp();
                        System.out.printf("%n[DEMO] corpus grew by %,d -> %,d active vectors%n",
                                step, store.size());
                    }
                });
                startDaemon("demo-heartbeat", () -> {
                    while (sleepSeconds(5, running)) System.out.println(hydra.status().oneLine());
                });
            }

            System.out.println("Synthetic corpus: " + store.size() + " vectors. "
                    + "Commands: :status  :grow N  :ghost F  exit");

            while (true) {
                System.out.print("\nhydra> ");
                if (!scanner.hasNextLine()) break;
                String line = scanner.nextLine().trim();
                if (line.equalsIgnoreCase("exit")) break;
                if (line.isBlank()) continue;

                if (line.equalsIgnoreCase(":status")) {
                    System.out.println(hydra.status().format());
                } else if (line.toLowerCase().startsWith(":grow ")) {
                    try {
                        int n = Integer.parseInt(line.substring(6).trim());
                        store.append(corpus.generate(n));
                        manager.catchUp();
                        System.out.printf("Added %,d vectors -> %,d active%n", n, store.size());
                    } catch (NumberFormatException e) {
                        System.out.println("Usage: :grow 5000");
                    }
                } else if (line.toLowerCase().startsWith(":ghost ")) {
                    try {
                        double f = Double.parseDouble(line.substring(7).trim());
                        int n = store.deactivateRandom(f, 1L);
                        System.out.printf("Deactivated %,d vectors -> %,d active / %,d stored%n",
                                n, store.size(), store.idLimit());
                    } catch (NumberFormatException e) {
                        System.out.println("Usage: :ghost 0.4");
                    }
                } else {
                    System.out.println("Unknown command.");
                }
            }
            running.set(false);
        }
    }

    // ------------------------------------------------------------------
    // PDF mode (real documents, real embedder)
    // ------------------------------------------------------------------

    private static void runPdf() throws Exception {
        Path rawDocs = Path.of("data/raw-docs");
        Path persistenceDir = Path.of("data/persisted-index");

        try (MiniLMEmbedder embedder = new MiniLMEmbedder();
             PersistentVectorStore vectorStore = new PersistentVectorStore(
                     persistenceDir, DIM, new MetadataFile());
             Scanner scanner = new Scanner(System.in)) {

            SemanticChunker chunker = new SemanticChunker(embedder, 0.45);
            PdfDirectoryLoader loader = new PdfDirectoryLoader();
            PdfExtractor extractor = new PdfExtractor();
            ChunkEmbedder chunkEmbedder = new ChunkEmbedder(embedder);
            DocumentManifest manifest = new DocumentManifest(persistenceDir.resolve("documents.json"));

            new IncrementalIngestionService(rawDocs, loader, extractor, chunker,
                    chunkEmbedder, vectorStore, manifest).synchronize();

            IndexManager manager = new IndexManager(vectorStore, new BruteForceHead(vectorStore));

            HydraConfig cfg = HydraConfig.defaults();
            cfg.dim = DIM;
            cfg.k = K;
            cfg.sampleEvery = 1;       // you type few queries, so sample all of them
            cfg.minSamples = 5;
            cfg.windowSize = 20;
            cfg.cooldownMs = 30_000;

            try (HydraController hydra = new HydraController(
                    manager, vectorStore, cfg, HydraController.defaultCandidates(cfg))) {

                hydra.setListener(msg -> System.out.println("\n[HYDRA] " + msg));
                hydra.start();

                while (true) {
                    System.out.print("\nSearch (':status', 'exit'): ");
                    if (!scanner.hasNextLine()) break;
                    String line = scanner.nextLine().trim();
                    if (line.equalsIgnoreCase("exit")) break;
                    if (line.isBlank()) continue;

                    if (line.equalsIgnoreCase(":status")) {
                        System.out.println(hydra.status().format());
                        continue;
                    }

                    float[] queryVector = embedder.embed(line);
                    printResults(manager.search(queryVector, K));
                    System.out.println("\n" + hydra.status().oneLine());
                }
            }
        }
    }

    // ------------------------------------------------------------------
    // Helpers
    // ------------------------------------------------------------------

    private static void printResults(List<SearchResult> results) {
        System.out.println();
        if (results.isEmpty()) {
            System.out.println("No matching chunks found.");
            return;
        }
        for (int i = 0; i < results.size(); i++) {
            SearchResult r = results.get(i);
            TextChunk c = r.getChunkEmbedding().getChunk();
            String text = c.getText().replaceAll("\\s+", " ");
            if (text.length() > 160) text = text.substring(0, 160) + "...";
            System.out.printf("#%d  %.4f  %s  p.%s%n    %s%n",
                    i + 1, r.getScore(), Path.of(c.getDocumentId()).getFileName(), c.getPageNumber(), text);
        }
    }

    private static void startDaemon(String name, Runnable r) {
        Thread t = new Thread(r, name);
        t.setDaemon(true);
        t.start();
    }

    private static boolean sleepSeconds(int seconds, AtomicBoolean running) {
        try {
            for (int i = 0; i < seconds; i++) {
                if (!running.get()) return false;
                Thread.sleep(1000);
            }
            return running.get();
        } catch (InterruptedException e) {
            return false;
        }
    }
}