package com.hydrann.annheads;

import com.hydrann.similarity.CosineSimilarity;
import com.hydrann.vectorstore.VectorStore;

import java.util.*;

/**
 * IvfHead — Inverted File Index (IVF) approximate nearest-neighbour head.
 *
 * <h2>Algorithm</h2>
 * <ol>
 *   <li><b>Build:</b> Run k-means on all active vectors to produce
 *       {@code nClusters} centroids.  Each vector is assigned to its nearest
 *       centroid and recorded in that centroid's inverted list.</li>
 *   <li><b>Search:</b> Find the {@code nProbe} closest centroids to the
 *       query, then score every vector in those lists exactly.  Keeps only
 *       the top-k via {@link TopK}.</li>
 *   <li><b>Insert:</b> Assign the new vector to the nearest existing centroid
 *       (no retraining).  The centroid list stays valid until the next
 *       {@link #build()} call.</li>
 * </ol>
 *
 * <h2>Integration contract (matches BruteForceHead / LshHead)</h2>
 * <ul>
 *   <li>Similarity metric: {@code CosineSimilarity.dot()} — valid because
 *       MiniLM vectors are L2-normalised.</li>
 *   <li>Vectors and ids are read from the shared {@link VectorStore}.</li>
 *   <li>Memory estimate: centroid storage + pointer lists.</li>
 * </ul>
 */
public class IvfHead implements ANNHead,Tunable {

    private final VectorStore store;
    private final int nClusters;
    // private final int nProbe;
    private final int maxKmeansIter;
    private final long seed;

    // ---- index state set by build() ----
    private float[][] centroids;                        // [nClusters][dim]
    private List<List<Integer>> invertedLists;          // invertedLists.get(c) = list of store ids
    private int dim = -1;

    private volatile int nProbe;         // was: private final int nProbe;

// (c)
    @Override public int getEffort()  { return nProbe; }
    @Override public void setEffort(int effort) {
        this.nProbe = Math.max(1, Math.min(nClusters, effort));
    }
    @Override public int minEffort()  { return 1; }
    @Override public int maxEffort()  { return nClusters; }
    @Override public String effortName() { return "nProbe"; }

    // -------------------------------------------------------------------------
    // Constructors
    // -------------------------------------------------------------------------

    /**
     * Full constructor.
     *
     * @param store          shared vector store (same instance used by BruteForceHead)
     * @param nClusters      number of Voronoi cells; rule of thumb: sqrt(n)
     * @param nProbe         cells to inspect per query (1 ≤ nProbe ≤ nClusters)
     * @param maxKmeansIter  Lloyd's algorithm iteration cap
     * @param seed           reproducible random seed for centroid initialisation
     */
    public IvfHead(VectorStore store, int nClusters, int nProbe,
                   int maxKmeansIter, long seed) {
        if (nClusters < 1) throw new IllegalArgumentException("nClusters must be >= 1");
        if (nProbe    < 1) throw new IllegalArgumentException("nProbe must be >= 1");
        if (nProbe > nClusters)
            throw new IllegalArgumentException("nProbe must be <= nClusters");
        this.store          = store;
        this.nClusters      = nClusters;
        this.nProbe         = nProbe;
        this.maxKmeansIter  = maxKmeansIter;
        this.seed           = seed;
    }

    /** Convenience: 25 k-means iterations, fixed seed 42. */
    public IvfHead(VectorStore store, int nClusters, int nProbe) {
        this(store, nClusters, nProbe, 25, 42L);
    }

    // -------------------------------------------------------------------------
    // ANNHead contract
    // -------------------------------------------------------------------------

    @Override public String name() {
        return "ivf(nClusters=" + nClusters + ",nProbe=" + nProbe + ")";
    }

    /**
     * Full rebuild: runs k-means on every active vector in the store and
     * (re)builds all inverted lists.
     */
    @Override
    public void build() {
        int n = store.idLimit();

        // Collect active ids and discover dimension
        List<Integer> activeIds = new ArrayList<>(n);
        for (int id = 0; id < n; id++) {
            if (store.isActive(id)) {
                if (dim < 0) dim = store.vector(id).length;
                activeIds.add(id);
            }
        }
        if (activeIds.isEmpty()) return;   // nothing to index yet

        int k = Math.min(nClusters, activeIds.size());
        centroids     = kMeans(activeIds, k);
        invertedLists = buildInvertedLists(activeIds);
    }

    /**
     * Incremental insert: assign the new vector to the nearest existing
     * centroid.  If no centroids exist yet (first ever insert before build)
     * the vector is silently deferred — a subsequent {@link #build()} will
     * pick it up.
     */
    @Override
    public void insert(int id, float[] vector) {
        if (centroids == null) return;      // build() not yet called; deferred
        if (dim < 0) dim = vector.length;
        int c = nearestCentroid(vector);
        invertedLists.get(c).add(id);
    }

    /**
     * Search the {@code nProbe} nearest Voronoi cells and return the top-k
     * results by cosine similarity.
     */
    @Override
    public List<Neighbor> search(float[] query, int k) {
        if (centroids == null || invertedLists == null) return Collections.emptyList();

        // 1. Find nProbe nearest centroids (small linear scan over centroids only)
        int effectiveProbe = Math.min(nProbe, centroids.length);
        List<Integer> probeCentroids = nearestCentroidsTo(query, effectiveProbe);

        // 2. Exact scoring over candidate ids
        TopK top = new TopK(k);
        for (int c : probeCentroids) {
            for (int id : invertedLists.get(c)) {
                if (!store.isActive(id)) continue;    // skip soft-deleted
                top.offer(id, CosineSimilarity.dot(query, store.vector(id)));
            }
        }
        return top.toSortedList();
    }

    /** Rough estimate: centroid float arrays + boxed Integer pointers in lists. */
    @Override
    public long memoryBytes() {
        if (centroids == null || dim < 0) return 0L;
        long centroidBytes = (long) centroids.length * dim * 4;
        long listBytes     = store.idLimit() * 20L;  // ~20 bytes per boxed entry
        return centroidBytes + listBytes;
    }

    // -------------------------------------------------------------------------
    // K-means (Lloyd's algorithm with cosine similarity = dot for unit vectors)
    // -------------------------------------------------------------------------

    private float[][] kMeans(List<Integer> activeIds, int k) {
        int n = activeIds.size();
        Random rng = new Random(seed);

        // --- Initialise: pick k distinct random vectors as centroids ---
        float[][] cents = new float[k][dim];
        int[] shuffled  = reservoirSample(n, k, rng);
        for (int c = 0; c < k; c++) {
            cents[c] = store.vector(activeIds.get(shuffled[c])).clone();
        }

        int[] assignments = new int[n];

        for (int iter = 0; iter < maxKmeansIter; iter++) {
            boolean changed = false;

            // Assignment step
            for (int i = 0; i < n; i++) {
                int best = nearestAmong(store.vector(activeIds.get(i)), cents);
                if (best != assignments[i]) {
                    assignments[i] = best;
                    changed = true;
                }
            }
            if (!changed) break;    // converged

            // Update step: mean of assigned vectors
            double[][] sums  = new double[k][dim];
            int[]      counts = new int[k];

            for (int i = 0; i < n; i++) {
                int c       = assignments[i];
                float[] v   = store.vector(activeIds.get(i));
                counts[c]++;
                for (int d = 0; d < dim; d++) sums[c][d] += v[d];
            }

            for (int c = 0; c < k; c++) {
                if (counts[c] == 0) {
                    // Empty cluster — reinitialise to a random vector
                    cents[c] = store.vector(activeIds.get(rng.nextInt(n))).clone();
                } else {
                    double normSq = 0.0;
                    for (int d = 0; d < dim; d++) {
                        float val = (float) (sums[c][d] / counts[c]);
                        cents[c][d] = val;
                        normSq += (double) val * val;
                    }
                    double norm = Math.sqrt(normSq);
                    if (norm > 1e-12) {
                        for (int d = 0; d < dim; d++) {
                            cents[c][d] = (float) (cents[c][d] / norm);
                        }
                    }
                }
            }
        }
        return cents;
    }

    /** Build inverted lists from the current centroids by assigning every active id. */
    private List<List<Integer>> buildInvertedLists(List<Integer> activeIds) {
        List<List<Integer>> lists = new ArrayList<>(centroids.length);
        for (int c = 0; c < centroids.length; c++) lists.add(new ArrayList<>());
        for (int id : activeIds) {
            lists.get(nearestCentroid(store.vector(id))).add(id);
        }
        return lists;
    }

    /** Return the centroid index nearest to {@code v} (highest dot product). */
    private int nearestCentroid(float[] v) {
        return nearestAmong(v, centroids);
    }

    /** Return index of the centroid array entry with the highest dot product with {@code v}. */
    private int nearestAmong(float[] v, float[][] cents) {
        int best = 0;
        double bestScore = Double.NEGATIVE_INFINITY;
        for (int c = 0; c < cents.length; c++) {
            double s = CosineSimilarity.dot(v, cents[c]);
            if (s > bestScore) { bestScore = s; best = c; }
        }
        return best;
    }

    /**
     * Return the {@code nProbe} centroid indices with the highest similarity
     * to {@code query}, sorted highest-first.
     */
    private List<Integer> nearestCentroidsTo(float[] query, int nProbe) {
        // Min-heap on score so we keep the nProbe highest scores
        PriorityQueue<int[]> minHeap = new PriorityQueue<>(
                Comparator.comparingDouble(e -> Double.longBitsToDouble(((long) e[1] << 32) | (e[2] & 0xFFFFFFFFL))));

        // Simpler: just collect all centroid scores and take top-nProbe
        double[] scores = new double[centroids.length];
        for (int c = 0; c < centroids.length; c++) {
            scores[c] = CosineSimilarity.dot(query, centroids[c]);
        }

        // Partial sort via a min-heap of size nProbe
        PriorityQueue<int[]> heap = new PriorityQueue<>(
                Comparator.comparingDouble(e -> scores[e[0]]));
        for (int c = 0; c < centroids.length; c++) {
            if (heap.size() < nProbe) {
                heap.offer(new int[]{ c });
            } else if (scores[c] > scores[heap.peek()[0]]) {
                heap.poll();
                heap.offer(new int[]{ c });
            }
        }

        List<int[]> res = new ArrayList<>(heap);
        res.sort((a, b) -> Double.compare(scores[b[0]], scores[a[0]])); // descending
        List<Integer> ids = new ArrayList<>(res.size());
        for (int[] e : res) ids.add(e[0]);
        return ids;
    }

    /** Fisher-Yates partial shuffle — returns {@code k} distinct indices from [0, n). */
    private int[] reservoirSample(int n, int k, Random rng) {
        int[] indices = new int[n];
        for (int i = 0; i < n; i++) indices[i] = i;
        for (int i = 0; i < k; i++) {
            int j = i + rng.nextInt(n - i);
            int tmp = indices[i]; indices[i] = indices[j]; indices[j] = tmp;
        }
        return Arrays.copyOf(indices, k);
    }
}
