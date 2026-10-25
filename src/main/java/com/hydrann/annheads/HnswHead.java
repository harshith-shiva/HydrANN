package com.hydrann.annheads;

import com.hydrann.similarity.CosineSimilarity;

import java.util.*;

/**
 * HnswHead — Hierarchical Navigable Small World (HNSW) approximate
 * nearest-neighbour search.
 *
 * <h2>Algorithm overview</h2>
 * HNSW builds a multi-layer proximity graph:
 * <ul>
 *   <li><b>Layer 0</b> (bottom) contains all vectors and is the densest graph.</li>
 *   <li>Higher layers contain exponentially fewer nodes and act as "express
 *       highways" for fast long-range navigation.</li>
 * </ul>
 *
 * <p><b>Insertion:</b> For each new vector a random max-layer {@code ℓ} is
 * drawn from a geometric distribution.  The vector is inserted into layers
 * 0 … ℓ.  At each layer the {@code efConstruction} nearest neighbours are
 * found via greedy graph traversal and bidirectional edges are added (subject
 * to the per-layer connectivity limit {@code M}).</p>
 *
 * <p><b>Search:</b> Start at the entry point in the top layer, greedily
 * descend to layer 0, then run a beam search at layer 0 with beam width
 * {@code efSearch} to collect the {@code k} nearest neighbours.</p>
 *
 * <h2>Tuning parameters</h2>
 * <ul>
 *   <li>{@code M} — max connections per node per layer (default 16).
 *       Layer 0 uses {@code 2·M} connections (paper convention).</li>
 *   <li>{@code efConstruction} — beam width during build (default 200).
 *       Higher → better graph quality, slower build.</li>
 *   <li>{@code efSearch} — beam width during search (default 50).
 *       Higher → better recall, slower search.
 *       Must be ≥ k.</li>
 *   <li>{@code mL} — level multiplier, set to {@code 1 / ln(M)} by default
 *       (matches the original paper).</li>
 * </ul>
 *
 * <h2>Reference</h2>
 * Malkov &amp; Yashunin, "Efficient and robust approximate nearest neighbor
 * search using Hierarchical Navigable Small World graphs", IEEE TPAMI 2020.
 * (<a href="https://arxiv.org/abs/1603.09320">arXiv:1603.09320</a>)
 */
public class HnswHead implements ANNHead {

    // -------------------------------------------------------------------------
    // Configuration
    // -------------------------------------------------------------------------

    /** Max connections per layer (layer 0 uses 2·M). */
    private final int M;

    /** Max connections at layer 0. */
    private final int M0;

    /** Beam width used while inserting (build phase). */
    private final int efConstruction;

    /** Beam width used while searching (query phase). */
    private final int efSearch;

    /** Level probability multiplier: geometric distribution parameter. */
    private final double mL;

    private final Random rng;

    // -------------------------------------------------------------------------
    // Index state
    // -------------------------------------------------------------------------

    /** Number of nodes inserted so far. */
    private int nodeCount = 0;

    /**
     * For each node, its connections per layer.
     * {@code graph[nodeId][layer]} → list of neighbour node IDs.
     * graph is a list of per-node layer arrays.
     */
    private final List<List<List<Integer>>> graph = new ArrayList<>();

    /** Raw vectors stored during build. vectorData[nodeId] = float[]. */
    private final List<float[]> vectorData = new ArrayList<>();

    /** Maximum layer currently in the graph. */
    private int maxLayer = 0;

    /** Entry-point node ID (topmost layer). */
    private int entryPoint = -1;

    /** Dimension of indexed vectors. */
    private int dim = -1;

    private boolean built = false;

    // -------------------------------------------------------------------------
    // Constructors
    // -------------------------------------------------------------------------

    /**
     * Full constructor.
     *
     * @param M               max connections per layer (typical: 12–48)
     * @param efConstruction  beam width at build time (typical: 100–500)
     * @param efSearch        beam width at search time (typical: 50–200)
     * @param seed            random seed; use -1 for time-based seed
     */
    public HnswHead(int M, int efConstruction, int efSearch, long seed) {
        if (M < 2)              throw new IllegalArgumentException("M must be >= 2");
        if (efConstruction < 1) throw new IllegalArgumentException("efConstruction must be >= 1");
        if (efSearch < 1)       throw new IllegalArgumentException("efSearch must be >= 1");

        this.M               = M;
        this.M0              = 2 * M;
        this.efConstruction  = efConstruction;
        this.efSearch        = efSearch;
        this.mL              = 1.0 / Math.log(M);
        this.rng             = (seed < 0) ? new Random() : new Random(seed);
    }

    /**
     * Convenience constructor with typical defaults.
     *
     * @param M             max connections per layer
     * @param efConstruction build-time beam width
     * @param efSearch       search-time beam width
     */
    public HnswHead(int M, int efConstruction, int efSearch) {
        this(M, efConstruction, efSearch, System.currentTimeMillis());
    }

    /**
     * Default constructor — M=16, efConstruction=200, efSearch=50.
     * Suitable for small-to-medium corpora.
     */
    public HnswHead() {
        this(16, 200, 50);
    }

    // -------------------------------------------------------------------------
    // ANNHead contract
    // -------------------------------------------------------------------------

    /**
     * {@inheritDoc}
     *
     * <p>Inserts all vectors into the HNSW graph one by one.  The graph is
     * built incrementally — each insertion uses the already-indexed vectors
     * as navigation structure.</p>
     */
    @Override
    public void build(List<float[]> vectors) {
        if (vectors == null || vectors.isEmpty()) {
            throw new IllegalArgumentException(
                    "HnswHead.build: vectors list must not be null or empty");
        }

        this.dim = vectors.get(0).length;

        for (int i = 1; i < vectors.size(); i++) {
            if (vectors.get(i).length != dim) {
                throw new IllegalArgumentException(
                        "HnswHead.build: dimension mismatch at index " + i
                        + " (expected " + dim + ", got " + vectors.get(i).length + ")");
            }
        }

        // Reset index state before rebuild
        graph.clear();
        vectorData.clear();
        nodeCount  = 0;
        maxLayer   = 0;
        entryPoint = -1;

        for (float[] v : vectors) {
            insert(v);
        }

        built = true;
    }

    /**
     * {@inheritDoc}
     *
     * <p>Performs greedy layer descent from the top layer, then a beam search
     * at layer 0 with beam width {@code efSearch}.  Returns the {@code k}
     * closest vectors in ascending distance order.</p>
     */
    @Override
    public List<Integer> search(float[] query, int k) {
        checkBuilt();
        validateQuery(query);
        if (k < 1) {
            throw new IllegalArgumentException(
                    "HnswHead.search: k must be >= 1, got " + k);
        }

        int ef       = Math.max(efSearch, k);
        int ep       = entryPoint;
        float epDist = CosineSimilarity.distance(query, vectorData.get(ep));

        // Greedy descent from maxLayer down to layer 1
        for (int layer = maxLayer; layer >= 1; layer--) {
            boolean changed = true;
            while (changed) {
                changed = false;
                List<Integer> neighbours = graph.get(ep).get(layer);
                for (int nb : neighbours) {
                    float d = CosineSimilarity.distance(query, vectorData.get(nb));
                    if (d < epDist) {
                        epDist  = d;
                        ep      = nb;
                        changed = true;
                    }
                }
            }
        }

        // Beam search at layer 0 with ef candidates
        PriorityQueue<int[]> W = searchLayer(query, ep, ef, 0);

        // Keep only the k nearest
        while (W.size() > k) {
            W.poll(); // max-heap: removes the worst
        }

        // Collect and sort nearest-first
        List<int[]> results = new ArrayList<>(W);
        results.sort(Comparator.comparingDouble(e -> Float.intBitsToFloat(e[1])));

        List<Integer> ids = new ArrayList<>(results.size());
        for (int[] e : results) {
            ids.add(e[0]);
        }
        return ids;
    }

    @Override
    public String name() {
        return "HNSW(M=" + M + ",efC=" + efConstruction + ",efS=" + efSearch + ")";
    }

    // -------------------------------------------------------------------------
    // HNSW insertion
    // -------------------------------------------------------------------------

    /**
     * Insert a single vector into the HNSW graph.
     */
    private void insert(float[] v) {
        int id = nodeCount++;

        // Draw the maximum layer for this node from a geometric distribution
        int nodeMaxLayer = drawLevel();

        // Allocate per-node layer list
        List<List<Integer>> nodeLayers = new ArrayList<>();
        for (int layer = 0; layer <= nodeMaxLayer; layer++) {
            nodeLayers.add(new ArrayList<>());
        }
        graph.add(nodeLayers);
        vectorData.add(v);

        if (entryPoint == -1) {
            // Very first node — becomes the entry point
            entryPoint = id;
            maxLayer   = nodeMaxLayer;
            return;
        }

        int ep       = entryPoint;
        float epDist = CosineSimilarity.distance(v, vectorData.get(ep));

        // Greedy descent from current maxLayer down to nodeMaxLayer + 1
        for (int layer = maxLayer; layer > nodeMaxLayer; layer--) {
            boolean changed = true;
            while (changed) {
                changed = false;
                for (int nb : graph.get(ep).get(layer)) {
                    if (nb >= id) continue; // node not yet inserted
                    float d = CosineSimilarity.distance(v, vectorData.get(nb));
                    if (d < epDist) {
                        epDist  = d;
                        ep      = nb;
                        changed = true;
                    }
                }
            }
        }

        // Insert at each layer from min(nodeMaxLayer, maxLayer) down to 0
        for (int layer = Math.min(nodeMaxLayer, maxLayer); layer >= 0; layer--) {
            int mLayer = (layer == 0) ? M0 : M;

            // Find efConstruction nearest neighbours at this layer
            PriorityQueue<int[]> candidates = searchLayer(v, ep, efConstruction, layer);

            // Select best M neighbours (smallest distances)
            List<int[]> neighbours = selectNeighbours(candidates, mLayer);

            // Add bidirectional edges
            for (int[] nb : neighbours) {
                int nbId = nb[0];

                // v → nb
                graph.get(id).get(layer).add(nbId);

                // nb → v (also add edge back)
                List<Integer> nbConns = graph.get(nbId).get(layer);
                nbConns.add(id);

                // If nb now exceeds its connection limit, prune excess
                if (nbConns.size() > mLayer) {
                    pruneConnections(nbId, layer, mLayer);
                }
            }

            // Update entry point for the next lower layer to the best found
            if (!candidates.isEmpty()) {
                // Use the closest candidate we found as the new ep
                int[] best = new int[]{ -1, Float.floatToIntBits(Float.MAX_VALUE) };
                for (int[] e : candidates) {
                    if (Float.intBitsToFloat(e[1]) < Float.intBitsToFloat(best[1])) {
                        best = e;
                    }
                }
                ep     = best[0];
                epDist = Float.intBitsToFloat(best[1]);
            }
        }

        // Update global entry point if this node has a higher layer
        if (nodeMaxLayer > maxLayer) {
            maxLayer   = nodeMaxLayer;
            entryPoint = id;
        }
    }

    // -------------------------------------------------------------------------
    // Core HNSW sub-routines
    // -------------------------------------------------------------------------

    /**
     * Beam search at a single graph layer.
     *
     * <p>Uses a candidate set C (min-heap, best first) and a result set W
     * (max-heap, worst first so we can easily evict).  We expand the closest
     * unvisited candidate and add its neighbours to C, keeping W at most
     * {@code ef} elements.</p>
     *
     * @param query  query vector
     * @param ep     entry-point node ID
     * @param ef     beam width (max candidates to keep)
     * @param layer  which graph layer to traverse
     * @return max-heap of {@code int[] {nodeId, Float.floatToIntBits(dist)}}
     *         (worst distance at top)
     */
    private PriorityQueue<int[]> searchLayer(float[] query, int ep, int ef, int layer) {
        Set<Integer> visited = new HashSet<>();
        visited.add(ep);

        float epDist = CosineSimilarity.distance(query, vectorData.get(ep));

        // C: min-heap (smallest dist first)
        PriorityQueue<int[]> C = new PriorityQueue<>(
                Comparator.comparingDouble(e -> Float.intBitsToFloat(e[1])));

        // W: max-heap (largest dist first) — our result set
        PriorityQueue<int[]> W = new PriorityQueue<>(
                Comparator.comparingDouble(e -> -Float.intBitsToFloat(e[1])));

        C.offer(new int[]{ ep, Float.floatToIntBits(epDist) });
        W.offer(new int[]{ ep, Float.floatToIntBits(epDist) });

        while (!C.isEmpty()) {
            int[] current = C.poll();
            int   currId  = current[0];
            float currDist = Float.intBitsToFloat(current[1]);

            // If the closest candidate in C is farther than the worst in W, stop
            float worstInW = Float.intBitsToFloat(W.peek()[1]);
            if (currDist > worstInW && W.size() >= ef) {
                break;
            }

            // Explore neighbours of currId at this layer
            List<Integer> neighbours = getNeighbours(currId, layer);
            for (int nb : neighbours) {
                if (visited.contains(nb)) continue;
                visited.add(nb);

                float d = CosineSimilarity.distance(query, vectorData.get(nb));
                float worstW = Float.intBitsToFloat(W.peek()[1]);

                if (W.size() < ef || d < worstW) {
                    C.offer(new int[]{ nb, Float.floatToIntBits(d) });
                    W.offer(new int[]{ nb, Float.floatToIntBits(d) });
                    if (W.size() > ef) {
                        W.poll(); // evict worst
                    }
                }
            }
        }

        return W;
    }

    /**
     * Select the best (nearest) {@code M} entries from the candidate heap.
     *
     * @param candidates max-heap from searchLayer
     * @param mMax       max neighbours to keep
     * @return sorted list of selected neighbours, nearest-first
     */
    private List<int[]> selectNeighbours(PriorityQueue<int[]> candidates, int mMax) {
        // Convert to list and sort by distance ascending
        List<int[]> sorted = new ArrayList<>(candidates);
        sorted.sort(Comparator.comparingDouble(e -> Float.intBitsToFloat(e[1])));

        return sorted.subList(0, Math.min(mMax, sorted.size()));
    }

    /**
     * Prune the connection list of node {@code nodeId} at {@code layer} to
     * at most {@code mMax} entries, keeping the nearest neighbours.
     */
    private void pruneConnections(int nodeId, int layer, int mMax) {
        List<Integer> conns = graph.get(nodeId).get(layer);
        if (conns.size() <= mMax) return;

        // Score each connection by distance
        List<int[]> scored = new ArrayList<>();
        for (int nb : conns) {
            float d = CosineSimilarity.distance(vectorData.get(nodeId), vectorData.get(nb));
            scored.add(new int[]{ nb, Float.floatToIntBits(d) });
        }
        scored.sort(Comparator.comparingDouble(e -> Float.intBitsToFloat(e[1])));

        conns.clear();
        for (int i = 0; i < mMax && i < scored.size(); i++) {
            conns.add(scored.get(i)[0]);
        }
    }

    /**
     * Get the connection list for {@code nodeId} at {@code layer}.
     * Returns an empty list if the node's max layer is below {@code layer}.
     */
    private List<Integer> getNeighbours(int nodeId, int layer) {
        List<List<Integer>> nodeLayers = graph.get(nodeId);
        if (layer >= nodeLayers.size()) {
            return Collections.emptyList();
        }
        return nodeLayers.get(layer);
    }

    /**
     * Draw a random level for a new node using the geometric distribution
     * described in the HNSW paper: ℓ = floor(−ln(uniform(0,1)) · mL).
     */
    private int drawLevel() {
        double r = rng.nextDouble();
        if (r == 0.0) r = 1e-9; // avoid log(0)
        return (int) Math.floor(-Math.log(r) * mL);
    }

    // -------------------------------------------------------------------------
    // Internal guards
    // -------------------------------------------------------------------------

    private void checkBuilt() {
        if (!built) {
            throw new IllegalStateException(
                    "HnswHead: call build() before search()");
        }
    }

    private void validateQuery(float[] query) {
        if (query == null) {
            throw new IllegalArgumentException(
                    "HnswHead.search: query must not be null");
        }
        if (query.length != dim) {
            throw new IllegalArgumentException(
                    "HnswHead.search: query dimension mismatch "
                    + "(expected " + dim + ", got " + query.length + ")");
        }
    }
}
