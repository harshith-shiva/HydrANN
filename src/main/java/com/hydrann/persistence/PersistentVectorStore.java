
package com.hydrann.persistence;

import com.hydrann.ingestion.ChunkEmbedding;
import com.hydrann.ingestion.TextChunk;
import com.hydrann.vectorstore.SearchResult;
import com.hydrann.vectorstore.VectorStore;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import java.util.Arrays;

public final class PersistentVectorStore
        implements VectorStore, AutoCloseable {

    private final Path directory;
    private final Path metadataPath;
    private final int dimension;

    private final VectorFile vectorFile;
    private final MetadataFile metadataFile;
    private final List<VectorMetadata> metadata;
    private volatile float[][] vectorCache;   // index = vector id, includes deactivated vectors

    private boolean closed;

    public PersistentVectorStore(
            Path directory,
            int expectedDimension,
            MetadataFile metadataFile) throws IOException {

        if (directory == null) {
            throw new IllegalArgumentException(
                    "Persistent directory cannot be null");
        }

        if (expectedDimension <= 0) {
            throw new IllegalArgumentException(
                    "Embedding dimension must be positive");
        }

        if (metadataFile == null) {
            throw new IllegalArgumentException(
                    "MetadataFile cannot be null");
        }

        this.directory = directory.toAbsolutePath().normalize();
        this.metadataPath = this.directory.resolve("metadata.jsonl");
        this.dimension = expectedDimension;
        this.metadataFile = metadataFile;

        Files.createDirectories(this.directory);

        VectorFile openedVectorFile = new VectorFile(
                this.directory.resolve("vectors.bin"),
                expectedDimension);

        try {
            List<VectorMetadata> loadedMetadata =
                    metadataFile.load(metadataPath);

            if (openedVectorFile.getCount() != loadedMetadata.size()) {
                throw new IOException(
                        "Vector/metadata count mismatch: vectors=" +
                        openedVectorFile.getCount() +
                        ", metadata=" + loadedMetadata.size());
            }

            validateMetadata(loadedMetadata, openedVectorFile);

            this.vectorFile = openedVectorFile;
            this.metadata = new ArrayList<>(loadedMetadata);
            float[][] loaded = new float[Math.toIntExact(openedVectorFile.getCount())][];
            for (int i = 0; i < loaded.length; i++) {
                loaded[i] = openedVectorFile.read(i);
            }
            this.vectorCache = loaded;

        } catch (IOException | RuntimeException e) {
            openedVectorFile.close();
            throw e;
        }
    }

    private void validateMetadata(
            List<VectorMetadata> records,
            VectorFile file) throws IOException {

        for (int i = 0; i < records.size(); i++) {
            VectorMetadata record = records.get(i);

            if (record == null || record.vectorId() != i) {
                throw new IOException(
                        "Invalid metadata vector ID at position " + i);
            }

            if (record.documentId() == null
                    || record.documentPath() == null
                    || record.text() == null) {
                throw new IOException(
                        "Missing required metadata at vector ID " + i);
            }
        }

        if (file.getDimension() != dimension) {
            throw new IOException("Unexpected vector dimension");
        }
    }
    @Override
    public int idLimit() {
        return vectorCache.length;
    }

@Override
public synchronized boolean isActive(int id) {
    ensureOpen();
    return id >= 0 && id < metadata.size() && metadata.get(id).active();
}

@Override
public float[] vector(int id) {
    return vectorCache[id];     // callers must not modify the returned array
}

@Override
public synchronized ChunkEmbedding get(int id) {
    ensureOpen();
    VectorMetadata r = metadata.get(id);
    TextChunk chunk = new TextChunk(
            r.documentId(), r.pageNumber(), r.chunkIndex(), r.text());
    return new ChunkEmbedding(chunk, vector(id));
}

    @Override
    public synchronized void add(ChunkEmbedding embedding) {
        ensureOpen();

        if (embedding == null) {
            throw new IllegalArgumentException(
                    "Chunk embedding cannot be null");
        }

        TextChunk chunk = embedding.getChunk();

        if (chunk == null) {
            throw new IllegalArgumentException(
                    "Text chunk cannot be null");
        }

        float[] vector = embedding.getEmbedding();
        validateVector(vector);

        throw new UnsupportedOperationException(
                "Use addDocumentBatch() to append a vector with its " +
                "required source metadata. Single-vector add() cannot " +
                "safely infer documentPath or fingerprint.");
    }

    @Override
    public synchronized void addAll(
            List<ChunkEmbedding> embeddings) {

        ensureOpen();

        if (embeddings == null) {
            throw new IllegalArgumentException(
                    "Embeddings cannot be null");
        }

        throw new UnsupportedOperationException(
                "Use addDocumentBatch() to append vectors with their " +
                "required source metadata.");
    }

    /**
     * Append all embeddings belonging to one source document.
     *
     * The caller supplies a stable document ID, normalized source path,
     * and SHA-256 fingerprint for that PDF.
     */
    public synchronized void addDocumentBatch(
            String documentId,
            String documentPath,
            String fingerprint,
            List<ChunkEmbedding> embeddings) throws IOException {

        ensureOpen();

        if (documentId == null || documentId.isBlank()
                || documentPath == null || documentPath.isBlank()
                || fingerprint == null || fingerprint.isBlank()
                || embeddings == null) {
            throw new IllegalArgumentException(
                    "Document identity, fingerprint and embeddings " +
                    "are required");
        }

        for (ChunkEmbedding embedding : embeddings) {
            if (embedding == null || embedding.getChunk() == null) {
                throw new IllegalArgumentException(
                        "Embedding and chunk cannot be null");
            }
            validateVector(embedding.getEmbedding());

            if (!documentId.equals(
                    embedding.getChunk().getDocumentId())) {
                throw new IllegalArgumentException(
                        "Chunk document ID does not match batch ID");
            }
        }

        // Build the next metadata state in memory first.
        List<VectorMetadata> updated =
                new ArrayList<>(metadata);

        long nextId = vectorFile.getCount();

        for (ChunkEmbedding embedding : embeddings) {
            TextChunk chunk = embedding.getChunk();

            updated.add(new VectorMetadata(
                    nextId++,
                    documentId,
                    documentPath,
                    fingerprint,
                    chunk.getPageNumber(),
                    chunk.getChunkIndex(),
                    chunk.getText(),
                    true));
        }

        /*
         * Save metadata after appending vectors. If the process fails
         * during this operation, startup count validation will detect
         * an inconsistent state instead of silently accepting it.
         */
        for (ChunkEmbedding embedding : embeddings) {
            vectorFile.append(embedding.getEmbedding());
        }

        metadataFile.save(metadataPath, updated);

        metadata.clear();
        metadata.addAll(updated);
        float[][] old = vectorCache;
        float[][] next = Arrays.copyOf(old, old.length + embeddings.size());
        for (int i = 0; i < embeddings.size(); i++) {
            next[old.length + i] = embeddings.get(i).getEmbedding().clone();
        }
        vectorCache = next;
    }

    /**
     * Mark all vectors belonging to the given source path inactive.
     * Vector IDs are retained; vectors are never shifted in the file.
     */
    public synchronized void deactivateDocument(
            String documentPath) throws IOException {

        ensureOpen();

        if (documentPath == null || documentPath.isBlank()) {
            throw new IllegalArgumentException(
                    "Document path cannot be blank");
        }

        List<VectorMetadata> updated =
                new ArrayList<>(metadata);

        boolean changed = false;

        for (int i = 0; i < updated.size(); i++) {
            VectorMetadata old = updated.get(i);

            if (old.active()
                    && old.documentPath().equals(documentPath)) {

                updated.set(i, new VectorMetadata(
                        old.vectorId(),
                        old.documentId(),
                        old.documentPath(),
                        old.fingerprint(),
                        old.pageNumber(),
                        old.chunkIndex(),
                        old.text(),
                        false));

                changed = true;
            }
        }

        if (!changed) {
            return;
        }

        metadataFile.save(metadataPath, updated);

        metadata.clear();
        metadata.addAll(updated);
    }

    @Override
    public synchronized List<SearchResult> search(
            float[] queryVector,
            int k) {

        ensureOpen();

        if (k <= 0) {
            throw new IllegalArgumentException(
                    "k must be greater than zero");
        }

        validateVector(queryVector);

        List<SearchResult> results = new ArrayList<>();

        try {
            for (VectorMetadata record : metadata) {
                if (!record.active()) {
                    continue;
                }

                float[] storedVector =
                        vectorFile.read(record.vectorId());

                double score = dotProduct(queryVector, storedVector);

                TextChunk chunk = new TextChunk(
                        record.documentId(),
                        record.pageNumber(),
                        record.chunkIndex(),
                        record.text());

                ChunkEmbedding embedding =
                        new ChunkEmbedding(chunk, storedVector);

                results.add(new SearchResult(embedding, score));
            }
        } catch (IOException e) {
            throw new IllegalStateException(
                    "Could not read persisted vectors", e);
        }

        results.sort(
                Comparator.comparingDouble(SearchResult::getScore)
                        .reversed());

        return new ArrayList<>(
                results.subList(0, Math.min(k, results.size())));
    }

    @Override
    public synchronized int size() {
        ensureOpen();

        int activeCount = 0;

        for (VectorMetadata record : metadata) {
            if (record.active()) {
                activeCount++;
            }
        }

        return activeCount;
    }

    public synchronized long totalStoredVectors() {
        ensureOpen();
        return vectorFile.getCount();
    }

    public synchronized List<VectorMetadata> getMetadataSnapshot() {
        ensureOpen();
        return List.copyOf(metadata);
    }

    private double dotProduct(float[] a, float[] b) {
        double sum = 0.0;

        for (int i = 0; i < dimension; i++) {
            sum += (double) a[i] * b[i];
        }

        return sum;
    }

    private void validateVector(float[] vector) {
        if (vector == null) {
            throw new IllegalArgumentException(
                    "Vector cannot be null");
        }

        if (vector.length != dimension) {
            throw new IllegalArgumentException(
                    "Expected dimension " + dimension +
                    ", received " + vector.length);
        }

        for (float value : vector) {
            if (!Float.isFinite(value)) {
                throw new IllegalArgumentException(
                        "Vector contains NaN or Infinity");
            }
        }
    }

    private void ensureOpen() {
        if (closed) {
            throw new IllegalStateException(
                    "PersistentVectorStore is closed");
        }
    }

    @Override
    public synchronized void close() throws IOException {
        if (!closed) {
            closed = true;
            vectorFile.close();
        }
    }
}
