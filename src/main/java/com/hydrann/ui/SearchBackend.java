package com.hydrann.ui;

import java.util.List;
import java.util.function.Consumer;

/**
 * Everything the JavaFX layer needs from the search engine, expressed with plain data
 * types so the UI never touches the vector store, heads or controller directly.
 */
public interface SearchBackend extends AutoCloseable {

    record Hit(String title, String fileName, String path, int page, int chunk,
               double score, String text) {}

    record Outcome(List<Hit> hits, double millis, String servedBy) {}

    record IndexInfo(int documents, int activeVectors, int storedVectors,
                     String indexType, String indexDetail, String embedder) {}

    record DocInfo(String fileName, String path, int chunks, int pages) {}

    record HydraInfo(String phase, String phaseNote, String headName, String headType,
                     int activeVectors, int storedVectors, int samples, double recall,
                     double avgLatencyUs, double p95LatencyUs, double bruteAvgUs, double speedup,
                     long memoryBytes, long activeSinceMs,
                     int promotions, int competitions, int tunings,
                     String lastDecision, long lastDecisionAtMs,
                     List<String> scoreboard, List<String> events) {}

    /** Runs a semantic query. {@code exact = true} bypasses the ANN head and scans everything. */
    Outcome search(String query, int k, boolean exact) throws Exception;

    IndexInfo indexInfo();

    List<DocInfo> documents();

    HydraInfo hydra();

    /** Looks for new, changed or removed PDFs and updates the index. */
    void rescan() throws Exception;

    /** Receives one-line notices from the Hydra controller (switches, tuning, failures). */
    void setEventListener(Consumer<String> listener);

    String libraryFolder();

    @Override
    void close();
}
