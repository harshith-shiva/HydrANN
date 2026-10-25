package com.hydrann.annheads;

import com.hydrann.similarity.CosineSimilarity;
import com.hydrann.vectorstore.VectorStore;

import java.util.*;

/**
 * HnswHead — Hierarchical Navigable Small World graph, approximate k-NN.
 *
 * <h2>Algorithm (Malkov &amp; Yashunin, IEEE TPAMI 2020)</h2>
 * <p>HNSW organises vectors as a multi-layer proximity graph:</p>
 * <ul>
 *   <li><b>Layer 0</b> contains every node, connected to up to {@code M0 = 2·M}
 *       neighbours.</li>
 *   <li>Each higher layer contains an exponentially smaller subset of nodes
 *       (geometric distribution), each connected to up to {@code M} neighbours.
 *       These sparse layers act as a fast long-range navigation highway.</li>
 * </ul>
 *
 * <h2>Build / Insert</h2>
 * <p>Each vector is inserted incrementally:</p>
 * <ol>
 *   <li>Draw a random max-layer {@code ℓ} from a geometric distribution.</li>
 *   <li>Greedily descend from the top layer to {@code ℓ+1}, moving to the
 *       nearest neighbour at each step.</li>
 *   <li>For layers {@code ℓ} down to 0: beam-search with width
 *       {@code efConstruction}, add bidirectional edges, prune to M.</li>
 * </ol>
 * {@link #build()} calls {@link #insert} for every active id in the store.
 *
 * <h2>Search</h2>
 * <ol>
 *   <li>Greedy descent from the top layer to layer 1.</li>
 *   <li>Beam search at layer 0 with width {@code efSearch}.</li>
 * </ol>
 *
 * <h2>Integration contract (matches BruteForceHead / LshHead / IvfHead)</h2>
 * <ul>
 *   <li>Similarity: {@code CosineSimilarity.dot()} — valid because MiniLM
 *       vectors are L2-normalised (higher = more similar).</li>
 *   <li>Same {@link VectorStore} injection used by every other head.</li>
 *   <li>{@link TopK} collects final results.</li>
 *   <li>Returns {@code List<Neighbor>} (highest score first).</li>
 * </ul>
 */
public class HnswHead implements ANNHead {

    // -------------------------------------------------------------------------
    // Configuration
    // -------------------------------------------------------------------------

    private final VectorStore store;

    /** Max edges per node per layer (layer 0 uses {@code M0 = 2·M}). */
    private final int M;
    private final int M0;

    /** Beam width used during insertion (controls graph quality). */
    private final int efConstruction;

    /** Beam width used at query time (controls recall vs speed). */
    private final int efSearch;

    /** Geometric distribution multiplier: {@code mL = 1 / ln(M)}. */
    private final double mL;

    private final Random rng;

    // -------------------------------------------------------------------------
    // Graph state
    // -------------------------------------------------------------------------

    /**
     * Per-node adjacency lists, one list per layer.
     * {@code graph.get(nodeId).get(layer)} → list of neighbour node IDs.
     * Node IDs here are store ids (same namespace as VectorStore).
     */
    private final Map<Integer, List<List<Integer>>> graph = new HashMap<>();

    /** Highest layer currently present in the graph. */
    private int maxLayer = 0;

    /** Entry-point store id (lives in the topmost layer). */
    private int entryPoint = -1;

    // -------------------------------------------------------------------------
    // Constructors
    // -------------------------------------------------------------------------

    /**
     * Full constructor.
     *
     * @param store           shared vector store
     * @param M               max connections per layer (typical: 12–48)
     * @param efConstruction  beam width at build time (typical: 100–500)
     * @param efSearch        beam width at search time (typical: 50–200)
     * @param seed            RNG seed for reproducible level draws
     */
    public HnswHead(VectorStore store, int M, int efConstruction,
                    int efSearch, long seed) {
        if (M < 2)              throw new IllegalArgumentException("M must be >= 2");
        if (efConstruction < 1) throw new IllegalArgumentException("efConstruction must be >= 1");
        if (efSearch < 1)       throw new IllegalArgumentException("efSearch must be >= 1");
        this.store          = store;
        this.M              = M;
        this.M0             = 2 * M;
        this.efConstruction = efConstruction;
        this.efSearch       = efSearch;
        this.mL             = 1.0 / Math.log(M);
        this.rng            = new Random(seed);
    }

    /** Convenience: M=16, efConstruction=200, efSearch=50, seed=42. */
    public HnswHead(VectorStore store) {
        this(store, 16, 200, 50, 42L);
    }

    // -------------------------------------------------------------------------
    // ANNHead contract
    // -------------------------------------------------------------------------

    @Override public String name() {
        return "hnsw(M=" + M + ",efC=" + efConstruction + ",efS=" + efSearch + ")";
    }

    /**
     * Full rebuild: clears the graph and re-inserts every active vector from
     * the store in id order.
     */
    @Override
    public void build() {
        graph.clear();
        maxLayer   = 0;
        entryPoint = -1;
        for (int id = 0; id < store.idLimit(); id++) {
            if (store.isActive(id)) insertNode(id, store.vector(id));
        }
    }

    /**
     * Incremental insert: adds a single new vector to the existing graph.
     * Called by {@link com.hydrann.controller.IndexManager#catchUp()} after
     * each ingestion batch.
     */
    @Override
    public void insert(int id, float[] vector) {
        insertNode(id, vector);
    }

    /**
     * Greedy layer descent followed by a beam search at layer 0.
     * Returns top-k neighbours by cosine similarity (highest score first).
     */
    @Override
    public List<Neighbor> search(float[] query, int k) {
        if (entryPoint < 0) return Collections.emptyList();

        int ep      = entryPoint;
        double epScore = CosineSimilarity.dot(query, store.vector(ep));

        // Greedy descent: top layer → layer 1 (find the best entry for layer 0)
        for (int layer = maxLayer; layer >= 1; layer--) {
            boolean changed = true;
            while (changed) {
                changed = false;
                for (int nb : neighbours(ep, layer)) {
                    double s = CosineSimilarity.dot(query, store.vector(nb));
                    if (s > epScore) { epScore = s; ep = nb; changed = true; }
                }
            }
        }

        // Beam search at layer 0 with ef = max(efSearch, k)
        int ef = Math.max(efSearch, k);
        PriorityQueue<int[]> W = searchLayer(query, ep, ef, 0);

        // Keep only the k best
        while (W.size() > k) W.poll();    // W is a min-heap; evict worst

        TopK top = new TopK(k);
        for (int[] e : W) top.offer(e[0], Double.longBitsToDouble(
                ((long) e[1] << 32) | (e[2] & 0xFFFFFFFFL)));
        return top.toSortedList();
    }

    /**
     * Rough memory estimate: edge storage.
     * Each node has at most M edges per layer (M0 at layer 0).
     * Layers per node average to ~2 (geometric distribution with p=1/M).
     */
    @Override
    public long memoryBytes() {
        // ~(M0 + M) ints per node average, 4 bytes each + HashMap overhead (~60 bytes/entry)
        long n = graph.size();
        return n * (M0 + M) * 4L + n * 60L;
    }

    // -------------------------------------------------------------------------
    // Core HNSW operations
    // -------------------------------------------------------------------------

    /** Insert a single node — called by both {@link #build()} and {@link #insert}. */
    private void insertNode(int id, float[] v) {
        int nodeMaxLayer = drawLevel();

        // Allocate layer lists for this node
        List<List<Integer>> nodeLayers = new ArrayList<>(nodeMaxLayer + 1);
        for (int i = 0; i <= nodeMaxLayer; i++) nodeLayers.add(new ArrayList<>());
        graph.put(id, nodeLayers);

        if (entryPoint < 0) {           // very first node
            entryPoint = id;
            maxLayer   = nodeMaxLayer;
            return;
        }

        int    ep      = entryPoint;
        double epScore = CosineSimilarity.dot(v, store.vector(ep));

        // Greedy descent from maxLayer down to nodeMaxLayer + 1
        for (int layer = maxLayer; layer > nodeMaxLayer; layer--) {
            boolean changed = true;
            while (changed) {
                changed = false;
                for (int nb : neighbours(ep, layer)) {
                    double s = CosineSimilarity.dot(v, store.vector(nb));
                    if (s > epScore) { epScore = s; ep = nb; changed = true; }
                }
            }
        }

        // Insert at layers nodeMaxLayer down to 0
        for (int layer = Math.min(nodeMaxLayer, maxLayer); layer >= 0; layer--) {
            int mLayer = (layer == 0) ? M0 : M;

            // Beam search for efConstruction candidates
            PriorityQueue<int[]> candidates = searchLayer(v, ep, efConstruction, layer);

            // Select best mLayer neighbours
            List<int[]> chosen = selectNeighbours(candidates, mLayer);

            // Add bidirectional edges
            for (int[] nb : chosen) {
                int nbId = nb[0];
                addEdge(id, nbId, layer);
                addEdge(nbId, id, layer);
                pruneNeighbours(nbId, layer, mLayer);
            }

            // Update ep for the next lower layer to the best candidate found
            if (!candidates.isEmpty()) {
                // Drain to find the maximum-score entry
                int bestId = ep; double bestScore = epScore;
                for (int[] e : candidates) {
                    double s = scoreFromEntry(e);
                    if (s > bestScore) { bestScore = s; bestId = e[0]; }
                }
                ep      = bestId;
                epScore = bestScore;
            }
        }

        if (nodeMaxLayer > maxLayer) { maxLayer = nodeMaxLayer; entryPoint = id; }
    }

    // -------------------------------------------------------------------------
    // Beam search at one layer
    // -------------------------------------------------------------------------

    /**
     * Standard HNSW beam search at a single layer.
     *
     * @param v     query / insertion vector
     * @param ep    entry-point node id
     * @param ef    beam width
     * @param layer graph layer to traverse
     * @return min-heap (lowest score at top) of {@code int[]{ id, scoreHigh, scoreLow }}
     *         with at most {@code ef} entries
     */
    private PriorityQueue<int[]> searchLayer(float[] v, int ep, int ef, int layer) {
        Set<Integer> visited = new HashSet<>();
        visited.add(ep);

        double epScore = CosineSimilarity.dot(v, store.vector(ep));

        // C: min-heap — best (highest score) first for expansion
        PriorityQueue<int[]> C = new PriorityQueue<>(
                Comparator.comparingDouble(HnswHead::scoreFromEntry).reversed());

        // W: min-heap — worst (lowest score) at top for easy eviction
        PriorityQueue<int[]> W = new PriorityQueue<>(
                Comparator.comparingDouble(HnswHead::scoreFromEntry));

        int[] epEntry = entryOf(ep, epScore);
        C.offer(epEntry);
        W.offer(epEntry);

        while (!C.isEmpty()) {
            int[] curr      = C.poll();
            double currScore = scoreFromEntry(curr);

            // If the best unexplored node is worse than the worst in W, stop
            if (W.size() >= ef && currScore < scoreFromEntry(W.peek())) break;

            for (int nb : neighbours(curr[0], layer)) {
                if (visited.contains(nb)) continue;
                visited.add(nb);

                double s = CosineSimilarity.dot(v, store.vector(nb));
                if (W.size() < ef || s > scoreFromEntry(W.peek())) {
                    int[] e = entryOf(nb, s);
                    C.offer(e);
                    W.offer(e);
                    if (W.size() > ef) W.poll();
                }
            }
        }
        return W;   // min-heap, worst at top
    }

    // -------------------------------------------------------------------------
    // Neighbour selection & pruning
    // -------------------------------------------------------------------------

    /**
     * Simple greedy neighbour selection: take the {@code mMax} entries with
     * the highest scores from {@code candidates}.
     */
    private List<int[]> selectNeighbours(PriorityQueue<int[]> candidates, int mMax) {
        List<int[]> sorted = new ArrayList<>(candidates);
        sorted.sort(Comparator.comparingDouble(HnswHead::scoreFromEntry).reversed());
        return sorted.subList(0, Math.min(mMax, sorted.size()));
    }

    /** Add a directed edge from {@code from} to {@code to} at {@code layer}. */
    private void addEdge(int from, int to, int layer) {
        List<List<Integer>> layers = graph.get(from);
        if (layers == null || layer >= layers.size()) return;
        List<Integer> conn = layers.get(layer);
        if (!conn.contains(to)) conn.add(to);
    }

    /**
     * If node {@code id}'s connection list at {@code layer} exceeds
     * {@code mMax}, keep only the {@code mMax} nearest neighbours.
     */
    private void pruneNeighbours(int id, int layer, int mMax) {
        List<List<Integer>> layers = graph.get(id);
        if (layers == null || layer >= layers.size()) return;
        List<Integer> conn = layers.get(layer);
        if (conn.size() <= mMax) return;

        float[] v = store.vector(id);
        conn.sort(Comparator.comparingDouble(
                nb -> -CosineSimilarity.dot(v, store.vector(nb))));   // descending score
        conn.subList(mMax, conn.size()).clear();
    }

    // -------------------------------------------------------------------------
    // Utilities
    // -------------------------------------------------------------------------

    /** Get the neighbour list for {@code id} at {@code layer} (empty if absent). */
    private List<Integer> neighbours(int id, int layer) {
        List<List<Integer>> layers = graph.get(id);
        if (layers == null || layer >= layers.size()) return Collections.emptyList();
        return layers.get(layer);
    }

    /**
     * Draw a level from the geometric distribution used by the HNSW paper:
     * {@code floor(-ln(uniform(0,1)) * mL)}.
     */
    private int drawLevel() {
        double r = rng.nextDouble();
        if (r == 0.0) r = 1e-10;
        return (int) Math.floor(-Math.log(r) * mL);
    }

    /**
     * Pack a (nodeId, score) pair into an int[] so we can put it in a
     * PriorityQueue without boxing.  The score is stored as two ints via
     * {@code Double.doubleToLongBits}.
     */
    private static int[] entryOf(int id, double score) {
        long bits = Double.doubleToLongBits(score);
        return new int[]{ id, (int) (bits >>> 32), (int) bits };
    }

    private static double scoreFromEntry(int[] e) {
        long bits = ((long) e[1] << 32) | (e[2] & 0xFFFFFFFFL);
        return Double.longBitsToDouble(bits);
    }
}
