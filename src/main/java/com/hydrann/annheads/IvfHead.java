package com.hydrann.annheads;

import com.hydrann.similarity.CosineSimilarity;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.PriorityQueue;
import java.util.Random;

/**
 * IvfHead — Inverted File Index (IVF) approximate nearest-neighbour search.
 *
 * <h2>Algorithm overview</h2>
 * <ol>
 *   <li><b>Training / Build:</b> Run k-means clustering on the corpus to
 *       produce {@code nClusters} centroids.  Each vector is assigned to its
 *       nearest centroid and recorded in an "inverted list" for that centroid.</li>
 *   <li><b>Search:</b> Compute the distance from the query to every centroid
 *       and choose the {@code nProbe} closest clusters.  Perform exact k-NN
 *       within the selected inverted lists only — far fewer comparisons than
 *       brute force.</li>
 * </ol>
 *
 * <h2>Tuning parameters</h2>
 * <ul>
 *   <li>{@code nClusters} — number of Voronoi cells (typical: √n where n is
 *       corpus size, or 100–4096 for large corpora).</li>
 *   <li>{@code nProbe} — number of clusters to inspect at query time.
 *       Higher → better recall, higher latency.
 *       {@code nProbe = nClusters} degrades to brute force.</li>
 *   <li>{@code maxKmeansIter} — k-means convergence iterations (default 25).</li>
 * </ul>
 *
 * <h2>Similarity metric</h2>
 * Cosine distance (1 − cosine similarity) via {@link CosineSimilarity}.
 */
public class IvfHead implements ANNHead {

    // -------------------------------------------------------------------------
    // Configuration (set via constructors)
    // -------------------------------------------------------------------------

    /** Number of Voronoi cells (centroids) to partition the space. */
    private final int nClusters;

    /**
     * Number of nearest clusters to probe at search time.
     * 1 ≤ nProbe ≤ nClusters.
     */
    private final int nProbe;

    /** Maximum k-means iterations during build. */
    private final int maxKmeansIter;

    /** Seed for reproducible k-means initialisation (use -1 for random). */
    private final long seed;

    // -------------------------------------------------------------------------
    // Index state (populated by build())
    // -------------------------------------------------------------------------

    /** Centroid vectors [nClusters][dim]. */
    private float[][] centroids;

    /**
     * Inverted lists: invertedLists[c] holds the corpus indices of all
     * vectors assigned to centroid c.
     */
    private List<List<Integer>> invertedLists;

    /** Original corpus — retained for distance computation during search. */
    private List<float[]> corpus;

    /** Dimensionality of indexed vectors. */
    private int dim;

    private boolean built = false;

    // -------------------------------------------------------------------------
    // Constructors
    // -------------------------------------------------------------------------

    /**
     * Full constructor with explicit random seed (useful for reproducible tests).
     *
     * @param nClusters     number of k-means centroids
     * @param nProbe        number of clusters to probe at query time
     * @param maxKmeansIter maximum k-means iterations
     * @param seed          random seed for centroid initialisation
     */
    public IvfHead(int nClusters, int nProbe, int maxKmeansIter, long seed) {
        if (nClusters < 1) throw new IllegalArgumentException("nClusters must be >= 1");
        if (nProbe    < 1) throw new IllegalArgumentException("nProbe must be >= 1");
        if (nProbe > nClusters)
            throw new IllegalArgumentException("nProbe must be <= nClusters");
        if (maxKmeansIter < 1)
            throw new IllegalArgumentException("maxKmeansIter must be >= 1");

        this.nClusters     = nClusters;
        this.nProbe        = nProbe;
        this.maxKmeansIter = maxKmeansIter;
        this.seed          = seed;
    }

    /**
     * Convenience constructor with sensible defaults and a random seed.
     *
     * @param nClusters number of k-means centroids
     * @param nProbe    clusters to probe at search time
     */
    public IvfHead(int nClusters, int nProbe) {
        this(nClusters, nProbe, 25, System.currentTimeMillis());
    }

    /**
     * Default constructor: 8 clusters, 2 probes, 25 k-means iterations.
     * Suitable for small test corpora.
     */
    public IvfHead() {
        this(8, 2, 25, System.currentTimeMillis());
    }

    // -------------------------------------------------------------------------
    // ANNHead contract
    // -------------------------------------------------------------------------

    /**
     * {@inheritDoc}
     *
     * <p>Runs k-means clustering with cosine distance on the provided vectors,
     * then builds the inverted lists by assigning each vector to its nearest
     * centroid.</p>
     */
    @Override
    public void build(List<float[]> vectors) {
        if (vectors == null || vectors.isEmpty()) {
            throw new IllegalArgumentException(
                    "IvfHead.build: vectors list must not be null or empty");
        }

        int n = vectors.size();
        this.dim = vectors.get(0).length;

        // Validate uniform dimensionality
        for (int i = 1; i < n; i++) {
            if (vectors.get(i).length != dim) {
                throw new IllegalArgumentException(
                        "IvfHead.build: dimension mismatch at index " + i
                        + " (expected " + dim + ", got " + vectors.get(i).length + ")");
            }
        }

        this.corpus = new ArrayList<>(vectors);

        // Cap nClusters to corpus size to avoid empty centroids
        int k = Math.min(nClusters, n);

        this.centroids     = kMeans(vectors, k);
        this.invertedLists = buildInvertedLists(vectors, centroids);
        this.built         = true;
    }

    /**
     * {@inheritDoc}
     *
     * <p>Finds the {@code nProbe} nearest centroids to the query, then
     * performs exact k-NN within those clusters' inverted lists.</p>
     */
    @Override
    public List<Integer> search(float[] query, int k) {
        checkBuilt();
        validateQuery(query);
        if (k < 1) {
            throw new IllegalArgumentException(
                    "IvfHead.search: k must be >= 1, got " + k);
        }

        // 1. Find the nProbe nearest centroids
        int effectiveProbe = Math.min(nProbe, centroids.length);
        List<Integer> nearestCentroids = nearestCentroidsTo(query, effectiveProbe);

        // 2. Collect candidate corpus indices from selected inverted lists
        //    (use a set-based approach to avoid duplicates if a vector ended
        //    up at the boundary of two cells — IVF assigns each vector to
        //    exactly one cell, so duplicates are impossible here, but
        //    guarding is good practice)
        List<Integer> candidates = new ArrayList<>();
        for (int c : nearestCentroids) {
            candidates.addAll(invertedLists.get(c));
        }

        if (candidates.isEmpty()) {
            return new ArrayList<>();
        }

        // 3. Exact k-NN within candidates using a max-heap of size k
        int effectiveK = Math.min(k, candidates.size());

        PriorityQueue<int[]> maxHeap = new PriorityQueue<>(
                effectiveK + 1,
                Comparator.comparingDouble(e -> -Float.intBitsToFloat(e[1]))
        );

        for (int idx : candidates) {
            float dist = CosineSimilarity.distance(query, corpus.get(idx));
            if (maxHeap.size() < effectiveK) {
                maxHeap.offer(new int[]{ idx, Float.floatToIntBits(dist) });
            } else {
                float worstDist = Float.intBitsToFloat(maxHeap.peek()[1]);
                if (dist < worstDist) {
                    maxHeap.poll();
                    maxHeap.offer(new int[]{ idx, Float.floatToIntBits(dist) });
                }
            }
        }

        // 4. Sort by distance ascending and return index list
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
        return "IVF(nClusters=" + nClusters + ",nProbe=" + nProbe + ")";
    }

    // -------------------------------------------------------------------------
    // K-means clustering (Lloyd's algorithm with cosine distance)
    // -------------------------------------------------------------------------

    /**
     * Run k-means with {@code k} clusters on {@code vectors}.
     *
     * @return centroid array of shape [k][dim]
     */
    private float[][] kMeans(List<float[]> vectors, int k) {
        int n = vectors.size();
        Random rng = (seed < 0) ? new Random() : new Random(seed);

        // ---- Initialise centroids by sampling k distinct vectors at random ----
        float[][] cents = new float[k][dim];
        int[] chosen = reservoirSample(n, k, rng);
        for (int c = 0; c < k; c++) {
            cents[c] = vectors.get(chosen[c]).clone();
        }

        int[] assignments = new int[n];

        // ---- Lloyd iteration ----
        for (int iter = 0; iter < maxKmeansIter; iter++) {

            boolean changed = false;

            // Assignment step
            for (int i = 0; i < n; i++) {
                int nearest = nearestCentroid(vectors.get(i), cents);
                if (nearest != assignments[i]) {
                    assignments[i] = nearest;
                    changed = true;
                }
            }

            if (!changed) break; // converged

            // Update step — recompute centroids as mean of assigned vectors
            float[][] sums   = new float[k][dim];
            int[]     counts = new int[k];

            for (int i = 0; i < n; i++) {
                int c = assignments[i];
                counts[c]++;
                float[] v = vectors.get(i);
                for (int d = 0; d < dim; d++) {
                    sums[c][d] += v[d];
                }
            }

            for (int c = 0; c < k; c++) {
                if (counts[c] == 0) {
                    // Empty cluster — reinitialise to a random vector
                    cents[c] = vectors.get(rng.nextInt(n)).clone();
                } else {
                    for (int d = 0; d < dim; d++) {
                        cents[c][d] = sums[c][d] / counts[c];
                    }
                }
            }
        }

        return cents;
    }

    /**
     * Assign each vector to its nearest centroid and return the inverted lists.
     */
    private List<List<Integer>> buildInvertedLists(List<float[]> vectors,
                                                    float[][] cents) {
        int k = cents.length;
        List<List<Integer>> lists = new ArrayList<>(k);
        for (int c = 0; c < k; c++) {
            lists.add(new ArrayList<>());
        }
        for (int i = 0; i < vectors.size(); i++) {
            int c = nearestCentroid(vectors.get(i), cents);
            lists.get(c).add(i);
        }
        return lists;
    }

    /**
     * Return the index (0-based) of the centroid nearest to {@code v}.
     */
    private int nearestCentroid(float[] v, float[][] cents) {
        int best  = 0;
        float bestDist = Float.MAX_VALUE;
        for (int c = 0; c < cents.length; c++) {
            float d = CosineSimilarity.distance(v, cents[c]);
            if (d < bestDist) {
                bestDist = d;
                best     = c;
            }
        }
        return best;
    }

    /**
     * Return the {@code nProbe} centroid indices nearest to {@code query},
     * sorted nearest-first.
     */
    private List<Integer> nearestCentroidsTo(float[] query, int nProbe) {
        // Max-heap: keeps the nProbe smallest distances
        PriorityQueue<int[]> maxHeap = new PriorityQueue<>(
                nProbe + 1,
                Comparator.comparingDouble(e -> -Float.intBitsToFloat(e[1]))
        );
        for (int c = 0; c < centroids.length; c++) {
            float d = CosineSimilarity.distance(query, centroids[c]);
            if (maxHeap.size() < nProbe) {
                maxHeap.offer(new int[]{ c, Float.floatToIntBits(d) });
            } else {
                float worst = Float.intBitsToFloat(maxHeap.peek()[1]);
                if (d < worst) {
                    maxHeap.poll();
                    maxHeap.offer(new int[]{ c, Float.floatToIntBits(d) });
                }
            }
        }
        List<int[]> res = new ArrayList<>(maxHeap);
        res.sort(Comparator.comparingDouble(e -> Float.intBitsToFloat(e[1])));
        List<Integer> ids = new ArrayList<>(res.size());
        for (int[] e : res) ids.add(e[0]);
        return ids;
    }

    /**
     * Fisher-Yates reservoir sample — picks {@code k} distinct indices
     * from [0, n) without replacement.
     */
    private int[] reservoirSample(int n, int k, Random rng) {
        int[] indices = new int[n];
        for (int i = 0; i < n; i++) indices[i] = i;
        for (int i = 0; i < k; i++) {
            int j = i + rng.nextInt(n - i);
            int tmp  = indices[i];
            indices[i] = indices[j];
            indices[j] = tmp;
        }
        int[] sample = new int[k];
        System.arraycopy(indices, 0, sample, 0, k);
        return sample;
    }

    // -------------------------------------------------------------------------
    // Internal guards
    // -------------------------------------------------------------------------

    private void checkBuilt() {
        if (!built) {
            throw new IllegalStateException(
                    "IvfHead: call build() before search()");
        }
    }

    private void validateQuery(float[] query) {
        if (query == null) {
            throw new IllegalArgumentException(
                    "IvfHead.search: query must not be null");
        }
        if (query.length != dim) {
            throw new IllegalArgumentException(
                    "IvfHead.search: query dimension mismatch "
                    + "(expected " + dim + ", got " + query.length + ")");
        }
    }
}
