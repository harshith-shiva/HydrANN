package com.hydrann.annheads;

import com.hydrann.similarity.CosineSimilarity;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.PriorityQueue;

/**
 * BruteForceHead — exact k-nearest-neighbour search.
 *
 * <p>This head stores every vector in memory and computes cosine distance from
 * the query to <b>every</b> stored vector at search time.  It is O(n·d) per
 * query where n = corpus size, d = vector dimension.</p>
 *
 * <p>Being exact (recall = 1.0), BruteForce is used as the <em>ground-truth
 * baseline</em> by the evaluation layer to measure how well the approximate
 * heads (IVF, HNSW) perform.</p>
 */
public class BruteForceHead implements ANNHead {

    /** All indexed vectors.  Index in the list == original corpus index. */
    private List<float[]> corpus;

    /** Flag: build() has been called at least once. */
    private boolean built = false;

    // -------------------------------------------------------------------------
    // ANNHead contract
    // -------------------------------------------------------------------------

    /**
     * {@inheritDoc}
     *
     * <p>BruteForce just stores the vectors — there is no pre-processing step.</p>
     */
    @Override
    public void build(List<float[]> vectors) {
        if (vectors == null || vectors.isEmpty()) {
            throw new IllegalArgumentException(
                    "BruteForceHead.build: vectors list must not be null or empty");
        }

        // Validate uniform dimensionality
        int dim = vectors.get(0).length;
        for (int i = 1; i < vectors.size(); i++) {
            if (vectors.get(i).length != dim) {
                throw new IllegalArgumentException(
                        "BruteForceHead.build: dimension mismatch at index " + i
                        + " (expected " + dim + ", got " + vectors.get(i).length + ")");
            }
        }

        // Shallow-copy the list so external mutations don't affect the index
        this.corpus = new ArrayList<>(vectors);
        this.built  = true;
    }

    /**
     * {@inheritDoc}
     *
     * <p>Scans every stored vector, computes cosine distance, and returns the
     * {@code k} smallest distances in ascending order using a max-heap of size
     * {@code k} (avoids sorting the full corpus).</p>
     */
    @Override
    public List<Integer> search(float[] query, int k) {
        checkBuilt();
        validateQuery(query);
        if (k < 1) {
            throw new IllegalArgumentException(
                    "BruteForceHead.search: k must be >= 1, got " + k);
        }

        int effectiveK = Math.min(k, corpus.size());

        /*
         * Max-heap keyed by distance.  We keep at most k entries:
         * if the heap has k items and the new distance is smaller than the
         * worst (top of the max-heap), we evict the worst and insert the new.
         *
         * Heap element: int[] { index, Float.floatToIntBits(distance) }
         */
        PriorityQueue<int[]> maxHeap = new PriorityQueue<>(
                effectiveK + 1,
                Comparator.comparingDouble(e -> -Float.intBitsToFloat(e[1]))
        );

        for (int i = 0; i < corpus.size(); i++) {
            float dist = CosineSimilarity.distance(query, corpus.get(i));
            if (maxHeap.size() < effectiveK) {
                maxHeap.offer(new int[]{ i, Float.floatToIntBits(dist) });
            } else {
                // peek at the current worst distance
                float worstDist = Float.intBitsToFloat(maxHeap.peek()[1]);
                if (dist < worstDist) {
                    maxHeap.poll();
                    maxHeap.offer(new int[]{ i, Float.floatToIntBits(dist) });
                }
            }
        }

        // Drain the heap and sort by distance ascending
        List<int[]> results = new ArrayList<>(maxHeap);
        results.sort(Comparator.comparingDouble(e -> Float.intBitsToFloat(e[1])));

        List<Integer> indices = new ArrayList<>(results.size());
        for (int[] entry : results) {
            indices.add(entry[0]);
        }
        return indices;
    }

    @Override
    public String name() {
        return "BruteForce";
    }

    // -------------------------------------------------------------------------
    // Internal helpers
    // -------------------------------------------------------------------------

    private void checkBuilt() {
        if (!built) {
            throw new IllegalStateException(
                    "BruteForceHead: call build() before search()");
        }
    }

    private void validateQuery(float[] query) {
        if (query == null) {
            throw new IllegalArgumentException(
                    "BruteForceHead.search: query must not be null");
        }
        int dim = corpus.get(0).length;
        if (query.length != dim) {
            throw new IllegalArgumentException(
                    "BruteForceHead.search: query dimension mismatch "
                    + "(expected " + dim + ", got " + query.length + ")");
        }
    }
}
