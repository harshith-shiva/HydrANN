package com.hydrann;

import com.hydrann.annheads.ANNHead;
import com.hydrann.annheads.BruteForceHead;
import com.hydrann.annheads.HnswHead;
import com.hydrann.annheads.IvfHead;
import com.hydrann.annheads.LshHead;
import com.hydrann.annheads.Neighbor;
import com.hydrann.controller.IndexManager;
import com.hydrann.embedding.ChunkEmbedder;
import com.hydrann.embedding.MiniLMEmbedder;
import com.hydrann.ingestion.PdfDirectoryLoader;
import com.hydrann.ingestion.PdfExtractor;
import com.hydrann.ingestion.SemanticChunker;
import com.hydrann.ingestion.TextChunk;
import com.hydrann.persistence.DocumentManifest;
import com.hydrann.persistence.IncrementalIngestionService;
import com.hydrann.persistence.MetadataFile;
import com.hydrann.persistence.PersistentVectorStore;
import com.hydrann.vectorstore.SearchResult;
import com.hydrann.vectorstore.VectorStore;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Scanner;
import java.util.Set;
import java.util.stream.Collectors;

/**
 * Usage: SearchTest [mode] [quiet] [key=value ...]
 *
 *   mode   lsh (default) | hnsw | ivf | brute | all
 *   quiet  hide chunk text, show stats only
 *   params tables= bits=           (LSH)
 *          m= efc= efs=            (HNSW)
 *          clusters= probe=        (IVF)
 *
 * Prompt commands: :bench (runs data/queries.txt), :stats, exit
 */
public class SearchTest {

    private static final int DIM = 384;
    private static final int K = 5;

    /** Accumulated stats for one head. */
    private static final class Run {
        final ANNHead head;
        long buildMs;
        int queries;
        double recallSum;
        long latencyUs;

        Run(ANNHead head) { this.head = head; }

        void record(double recall, long us) {
            queries++;
            recallSum += recall;
            latencyUs += us;
        }

        void reset() { queries = 0; recallSum = 0; latencyUs = 0; }
        double avgRecall() { return queries == 0 ? 0 : recallSum / queries; }
        long avgUs() { return queries == 0 ? 0 : latencyUs / queries; }
    }

    public static void main(String[] args) throws Exception {

        // ---------- parse arguments ----------
        String mode = "lsh";
        boolean quiet = false;
        Map<String, Integer> params = new HashMap<>();

        for (String a : args) {
            if (a.equalsIgnoreCase("quiet")) {
                quiet = true;
            } else if (a.contains("=")) {
                String[] kv = a.split("=", 2);
                params.put(kv[0].toLowerCase(), Integer.parseInt(kv[1]));
            } else {
                mode = a.toLowerCase();
            }
        }

        if (!Set.of("lsh", "hnsw", "ivf", "brute", "all").contains(mode)) {
            System.out.println("Unknown mode '" + mode
                    + "'. Use: lsh | hnsw | ivf | brute | all");
            return;
        }

        Path rawDocs = Path.of("data/raw-docs");
        Path persistenceDir = Path.of("data/persisted-index");
        Path benchFile = Path.of("data/queries.txt");

        try (
                MiniLMEmbedder embedder = new MiniLMEmbedder();
                PersistentVectorStore vectorStore =
                        new PersistentVectorStore(
                                persistenceDir, DIM, new MetadataFile());
                Scanner scanner = new Scanner(System.in)
        ) {

            // ---------- ingestion (unchanged) ----------
            SemanticChunker chunker = new SemanticChunker(embedder, 0.45);
            PdfDirectoryLoader loader = new PdfDirectoryLoader();
            PdfExtractor extractor = new PdfExtractor();
            ChunkEmbedder chunkEmbedder = new ChunkEmbedder(embedder);

            DocumentManifest manifest = new DocumentManifest(
                    persistenceDir.resolve("documents.json"));

            IncrementalIngestionService ingestionService =
                    new IncrementalIngestionService(
                            rawDocs, loader, extractor, chunker,
                            chunkEmbedder, vectorStore, manifest);

            ingestionService.synchronize();

            // ---------- choose heads ----------
            BruteForceHead truthHead = new BruteForceHead(vectorStore);
            Run baseline = new Run(truthHead);

            int active = vectorStore.size();
            List<Run> runs = new ArrayList<>();      // heads under test
            ANNHead served = truthHead;              // head that serves displayed results

            switch (mode) {
                case "brute" -> { }
                case "all" -> {
                    for (String type : List.of("lsh", "hnsw", "ivf")) {
                        runs.add(new Run(createHead(type, vectorStore, params, active)));
                    }
                }
                default -> {
                    Run r = new Run(createHead(mode, vectorStore, params, active));
                    runs.add(r);
                    served = r.head;
                }
            }

            // ---------- build (timed) ----------
            long t = System.nanoTime();
            IndexManager index = new IndexManager(vectorStore, served);  // builds 'served'
            long servedBuildMs = (System.nanoTime() - t) / 1_000_000;

            for (Run r : runs) {
                if (r.head == served) {
                    r.buildMs = servedBuildMs;
                } else {
                    long s = System.nanoTime();
                    r.head.build();
                    r.buildMs = (System.nanoTime() - s) / 1_000_000;
                }
            }

            System.out.printf("%nMode: %s | active vectors: %d | K=%d%n",
                    mode, active, K);
            printSummary(baseline, runs);

            // ---------- interactive loop ----------
            while (true) {

                System.out.println();
                System.out.print("Search (':bench', ':stats', 'exit'): ");

                if (!scanner.hasNextLine()) break;

                String line = scanner.nextLine().trim();

                if (line.equalsIgnoreCase("exit")) break;
                if (line.isBlank()) continue;

                if (line.equalsIgnoreCase(":stats")) {
                    printSummary(baseline, runs);
                    continue;
                }

                if (line.equalsIgnoreCase(":bench")) {
                    runBench(benchFile, embedder, baseline, runs);
                    continue;
                }

                float[] queryVector = embedder.embed(line);

                if (!quiet) {
                    show(index.search(queryVector, K));
                }

                measure(queryVector, baseline, runs, true);
            }
        }
    }

    // ------------------------------------------------------------------
    // Head construction
    // ------------------------------------------------------------------

    private static ANNHead createHead(String type, VectorStore store,
                                      Map<String, Integer> p, int active) {
        return switch (type) {
            case "lsh" -> new LshHead(store, DIM,
                    p.getOrDefault("tables", 8),
                    p.getOrDefault("bits", 10),
                    true, 42L);

            case "hnsw" -> new HnswHead(store,
                    p.getOrDefault("m", 16),
                    p.getOrDefault("efc", 200),
                    p.getOrDefault("efs", 50),
                    42L);

            case "ivf" -> {
                int clusters = p.getOrDefault("clusters",
                        Math.max(1, (int) Math.sqrt(active)));
                int probe = Math.min(clusters,
                        p.getOrDefault("probe", Math.max(1, clusters / 4)));
                yield new IvfHead(store, clusters, probe);
            }

            default -> throw new IllegalArgumentException("Unknown head: " + type);
        };
    }

    // ------------------------------------------------------------------
    // Measurement
    // ------------------------------------------------------------------

    /** Run one query through the baseline and every head under test. */
    private static void measure(float[] q, Run baseline, List<Run> runs,
                                boolean print) {

        long t0 = System.nanoTime();
        List<Neighbor> truth = baseline.head.search(q, K);
        long baseUs = (System.nanoTime() - t0) / 1_000;
        baseline.record(1.0, baseUs);

        Set<Integer> truthIds = truth.stream()
                .map(Neighbor::id)
                .collect(Collectors.toSet());

        if (print && runs.isEmpty()) {
            System.out.printf("[%s] %d us%n", baseline.head.name(), baseUs);
        }

        for (Run r : runs) {
            long t1 = System.nanoTime();
            List<Neighbor> approx = r.head.search(q, K);
            long us = (System.nanoTime() - t1) / 1_000;

            long hits = approx.stream()
                    .filter(n -> truthIds.contains(n.id()))
                    .count();

            double recall = truthIds.isEmpty()
                    ? 1.0
                    : (double) hits / truthIds.size();

            r.record(recall, us);

            if (print) {
                System.out.printf(
                        "[%s] recall@%d = %d/%d | %d us vs brute %d us (%.1fx) | avg recall %.2f%n",
                        r.head.name(), K, hits, truthIds.size(),
                        us, baseUs, (double) baseUs / Math.max(1, us),
                        r.avgRecall());
            }
        }
    }

    /** Run every query in data/queries.txt; warm up first, then record. */
    private static void runBench(Path file, MiniLMEmbedder embedder,
                                 Run baseline, List<Run> runs) throws Exception {

        if (!Files.exists(file)) {
            System.out.println("Create " + file
                    + " (one query per line, '#' for comments), then run :bench again.");
            return;
        }

        List<String> queries = Files.readAllLines(file).stream()
                .map(String::trim)
                .filter(s -> !s.isBlank() && !s.startsWith("#"))
                .toList();

        if (queries.isEmpty()) {
            System.out.println(file + " has no queries.");
            return;
        }

        List<float[]> vectors = new ArrayList<>();
        for (String q : queries) vectors.add(embedder.embed(q));   // embedding is not timed

        for (float[] v : vectors) measure(v, baseline, runs, false);   // warm-up, discarded

        baseline.reset();
        runs.forEach(Run::reset);

        for (float[] v : vectors) measure(v, baseline, runs, false);

        System.out.println("\nBenchmark over " + vectors.size()
                + " queries (averages reset by :bench)");
        printSummary(baseline, runs);
    }

    // ------------------------------------------------------------------
    // Output
    // ------------------------------------------------------------------

    private static void printSummary(Run baseline, List<Run> runs) {
        System.out.println();
        System.out.println("====================== SUMMARY (K=" + K + ") ======================");
        System.out.printf("%-30s %7s %9s %8s %8s %9s %8s%n",
                "head", "queries", "recall@K", "avg us", "speedup", "build ms", "mem KB");

        System.out.printf("%-30s %7d %9.2f %8d %8s %9s %8d%n",
                baseline.head.name() + " [base]", baseline.queries, 1.0,
                baseline.avgUs(), "1.0x", "-", 0);

        for (Run r : runs) {
            double speedup = r.queries == 0
                    ? 0
                    : (double) baseline.avgUs() / Math.max(1, r.avgUs());

            System.out.printf("%-30s %7d %9.2f %8d %7.1fx %9d %8d%n",
                    r.head.name(), r.queries, r.avgRecall(), r.avgUs(),
                    speedup, r.buildMs, r.head.memoryBytes() / 1024);
        }
    }

    private static void show(List<SearchResult> results) {

        System.out.println();
        System.out.println("========== RESULTS ==========");

        if (results.isEmpty()) {
            System.out.println("No matching chunks found.");
            return;
        }

        for (int i = 0; i < results.size(); i++) {

            SearchResult result = results.get(i);
            TextChunk chunk = result.getChunkEmbedding().getChunk();

            System.out.println();
            System.out.println("#" + (i + 1));
            System.out.printf("Score: %.4f%n", result.getScore());
            System.out.println("Document: " + chunk.getDocumentId());
            System.out.println("Page: " + chunk.getPageNumber());
            System.out.println("Chunk: " + chunk.getChunkIndex());
            System.out.println("Text:");
            System.out.println(chunk.getText());
        }
    }
}