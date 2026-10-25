package com.hydrann.controller;

import com.hydrann.annheads.ANNHead;
import com.hydrann.annheads.BruteForceHead;
import com.hydrann.annheads.HnswHead;
import com.hydrann.annheads.IvfHead;
import com.hydrann.annheads.LshHead;
import com.hydrann.annheads.Neighbor;
import com.hydrann.annheads.Tunable;
import com.hydrann.controller.DegradationTrigger.Reason;
import com.hydrann.controller.HydraStatus.Phase;
import com.hydrann.evaluation.FitnessEvaluator;
import com.hydrann.vectorstore.SnapshotStore;
import com.hydrann.vectorstore.VectorStore;

import java.time.LocalTime;
import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.Deque;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.concurrent.ArrayBlockingQueue;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorCompletionService;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.ThreadFactory;
import java.util.concurrent.ThreadPoolExecutor;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicLong;
import java.util.function.Consumer;
import java.util.function.Function;

public final class HydraController implements AutoCloseable {

    /** A head the controller may build: {@code indexed=false} means no build, runs on the live store. */
    public record Candidate(String type, boolean indexed, Function<VectorStore, ANNHead> factory) {}

    private record Built(Candidate candidate, ANNHead head, long buildMs) {}
    private record Sample(double recall, double activeUs, double bruteUs) {}
    private record WindowStats(int n, long serial, double recall, double avgActiveUs,
                               double p95ActiveUs, double avgBruteUs) {}

    private static final class Contender {
        final String label, type;
        final ANNHead head;
        final boolean incumbent;
        FitnessEvaluator.Measurement m;
        long memoryBytes, buildMs;
        boolean eligible;
        double score = Double.NaN;
        String failure;

        Contender(String label, String type, ANNHead head, boolean incumbent) {
            this.label = label; this.type = type; this.head = head; this.incumbent = incumbent;
        }
    }

    private final IndexManager manager;
    private final VectorStore store;
    private final HydraConfig cfg;
    private final List<Candidate> candidates;
    private final QueryLog queryLog = new QueryLog(200);
    private final DegradationTrigger trigger;
    private final BruteForceHead monitorTruth;

    private final ExecutorService sampler = new ThreadPoolExecutor(
            1, 1, 0, TimeUnit.MILLISECONDS, new ArrayBlockingQueue<>(8),
            daemon("hydra-sampler"), new ThreadPoolExecutor.DiscardPolicy());
    private final ScheduledExecutorService scheduler =
            Executors.newSingleThreadScheduledExecutor(daemon("hydra-monitor"));
    private final ExecutorService competitionExecutor =
            Executors.newSingleThreadExecutor(daemon("hydra-competition"));
    private final ExecutorService buildPool;

    private final Object windowLock = new Object();
    private final ArrayDeque<Sample> window = new ArrayDeque<>();
    private long sampleSerial = 0;

    private final AtomicLong queryCounter = new AtomicLong();
    private final AtomicBoolean competing = new AtomicBoolean(false);
    private final AtomicInteger promotions = new AtomicInteger();
    private final AtomicInteger competitions = new AtomicInteger();
    private final AtomicInteger tunings = new AtomicInteger();

    private volatile Phase phase = Phase.STABLE;
    private volatile String phaseNote = "";
    private volatile long activeSinceMs = System.currentTimeMillis();
    private volatile long lastCompetitionEndMs = 0;
    private volatile String lastDecision = "none yet";
    private volatile long lastDecisionAtMs = 0;
    private volatile List<String> scoreboard = List.of();
    private volatile int tuningRounds = 0;
    private volatile Consumer<String> listener = s -> {};
    private final Deque<String> events = new ArrayDeque<>();

    public HydraController(IndexManager manager, VectorStore store,
                           HydraConfig cfg, List<Candidate> candidates) {
        this.manager = manager;
        this.store = store;
        this.cfg = cfg;
        this.candidates = candidates;
        this.trigger = new DegradationTrigger(cfg);
        this.monitorTruth = new BruteForceHead(store);
        int threads = Math.max(1, Math.min(candidates.size(), Runtime.getRuntime().availableProcessors()));
        this.buildPool = Executors.newFixedThreadPool(threads, daemon("hydra-build"));
    }

    // ------------------------------------------------------------------
    // Default contestants
    // ------------------------------------------------------------------

    public static List<Candidate> defaultCandidates(HydraConfig cfg) {
        return List.of(
                new Candidate("brute-force", false, s -> new BruteForceHead(s)),
                new Candidate("lsh", true, s -> {
                    int n = activeCount(s);
                    int bits = (int) Math.round(Math.log(Math.max(32, n) / 4.0) / Math.log(2));
                    bits = Math.max(4, Math.min(16, bits));
                    return new LshHead(s, cfg.dim, cfg.lshTables, bits, true, cfg.seed);
                }),
                new Candidate("hnsw", true, s ->
                        new HnswHead(s, cfg.hnswM, cfg.hnswEfConstruction, cfg.hnswEfSearch, cfg.seed)),
                new Candidate("ivf", true, s -> {
                    int clusters = Math.max(1, (int) Math.sqrt(activeCount(s)));
                    return new IvfHead(s, clusters, Math.max(1, clusters / 4));
                }));
    }

    private static int activeCount(VectorStore s) {
        int n = 0;
        for (int id = 0; id < s.idLimit(); id++) if (s.isActive(id)) n++;
        return n;
    }

    // ------------------------------------------------------------------
    // Lifecycle
    // ------------------------------------------------------------------

    public void start() {
        manager.setQueryObserver(this::onQuery);
        scheduler.scheduleWithFixedDelay(this::safeTick,
                cfg.checkIntervalMs, cfg.checkIntervalMs, TimeUnit.MILLISECONDS);
        log("Hydra online. Active head: " + manager.activeHead().name());
    }

    @Override
    public void close() {
        manager.setQueryObserver(null);
        scheduler.shutdownNow();
        sampler.shutdownNow();
        competitionExecutor.shutdownNow();
        buildPool.shutdownNow();
    }

    public void setListener(Consumer<String> listener) { this.listener = listener; }

    // ------------------------------------------------------------------
    // Monitoring
    // ------------------------------------------------------------------

    private void onQuery(float[] q) {
        queryLog.add(q);
        if (queryCounter.incrementAndGet() % cfg.sampleEvery == 0 && phase != Phase.COMPETING) {
            sampler.execute(() -> takeSample(q));      // dropped silently when the sampler is busy
        }
    }

    private void takeSample(float[] q) {
        try {
            boolean brute = manager.activeHead() instanceof BruteForceHead;

            long t0 = System.nanoTime();
            List<Neighbor> approx = manager.rawSearch(q, cfg.k);
            double activeUs = (System.nanoTime() - t0) / 1_000.0;

            double bruteUs, recall;
            if (brute) {
                bruteUs = activeUs;
                recall = 1.0;
            } else {
                long t1 = System.nanoTime();
                List<Neighbor> truth = monitorTruth.search(q, cfg.k);
                bruteUs = (System.nanoTime() - t1) / 1_000.0;
                recall = FitnessEvaluator.recall(truth, approx, Integer.MAX_VALUE);
            }

            synchronized (windowLock) {
                window.addLast(new Sample(recall, activeUs, bruteUs));
                while (window.size() > cfg.windowSize) window.removeFirst();
                sampleSerial++;
            }
        } catch (RuntimeException e) {
            log("sample failed: " + e);
        }
    }

    private WindowStats windowStats() {
        synchronized (windowLock) {
            int n = window.size();
            if (n == 0) return new WindowStats(0, sampleSerial, 1.0, 0, 0, 0);
            double[] act = new double[n];
            double r = 0, a = 0, b = 0;
            int i = 0;
            for (Sample s : window) {
                act[i++] = s.activeUs();
                r += s.recall();
                a += s.activeUs();
                b += s.bruteUs();
            }
            return new WindowStats(n, sampleSerial, r / n, a / n,
                    FitnessEvaluator.percentile(act, 0.95), b / n);
        }
    }

    private void clearWindow() {
        synchronized (windowLock) { window.clear(); }
    }

    // ------------------------------------------------------------------
    // Trigger tick (monitor thread, once per checkIntervalMs)
    // ------------------------------------------------------------------

    private void safeTick() {
        try {
            tick();
        } catch (Throwable t) {
            log("monitor error: " + t);
        }
    }

    private void tick() {
        if (phase == Phase.COMPETING) return;

        WindowStats w = windowStats();

        if (phase == Phase.TUNING) {
            if (w.n() < cfg.minSamples) {
                phaseNote = "re-measuring after tune (" + w.n() + "/" + cfg.minSamples + " samples)";
                return;
            }
            phase = Phase.STABLE;
            phaseNote = "";
        }

        if (lastCompetitionEndMs != 0
                && System.currentTimeMillis() - lastCompetitionEndMs < cfg.cooldownMs) return;

        ANNHead head = manager.activeHead();
        DegradationTrigger.Observation obs = new DegradationTrigger.Observation(
                w.n(), w.serial(), w.recall(), w.avgActiveUs(), w.p95ActiveUs(), w.avgBruteUs(),
                store.size(), store.idLimit(), head instanceof BruteForceHead);

        Reason reason = trigger.check(obs);
        if (reason == Reason.NONE) {
            if (!trigger.breaching()) tuningRounds = 0;
            return;
        }

        if ((reason == Reason.LOW_RECALL || reason == Reason.HIGH_LATENCY)
                && tuningRounds < cfg.maxTuningRounds
                && tryTune(reason, obs)) {
            tuningRounds++;
            return;
        }

        tuningRounds = 0;
        startCompetition(reason, trigger.explain(reason, obs));
    }

    /** Cheapest response first: change query-time effort without rebuilding anything. */
    private boolean tryTune(Reason reason, DegradationTrigger.Observation obs) {
        ANNHead head = manager.activeHead();
        if (!(head instanceof Tunable t)) return false;

        int cur = t.getEffort();
        int next;
        if (reason == Reason.LOW_RECALL) {
            next = Math.min(t.maxEffort(), (int) Math.ceil(cur * cfg.effortStep));
        } else {
            if (obs.recall() < cfg.minRecall + cfg.recallHeadroom) return false;   // no headroom to trade
            next = Math.max(t.minEffort(), (int) Math.floor(cur / cfg.effortStep));
        }
        if (next == cur) return false;

        t.setEffort(next);
        tunings.incrementAndGet();
        clearWindow();
        trigger.reset();
        phase = Phase.TUNING;
        phaseNote = "re-measuring after tune (0/" + cfg.minSamples + " samples)";
        log(String.format("TUNE %s: %s %d -> %d (%s)", typeOf(head), t.effortName(), cur, next,
                trigger.explain(reason, obs)));
        return true;
    }

    // ------------------------------------------------------------------
    // Competition
    // ------------------------------------------------------------------

    private void startCompetition(Reason reason, String why) {
        if (!competing.compareAndSet(false, true)) return;
        phase = Phase.COMPETING;
        phaseNote = "starting (" + reason + ")";

        competitionExecutor.execute(() -> {
            try {
                runCompetition(reason, why);
            } catch (Throwable t) {
                log("COMPETITION failed: " + t);
            } finally {
                lastCompetitionEndMs = System.currentTimeMillis();
                clearWindow();
                trigger.reset();
                phase = Phase.STABLE;
                phaseNote = "";
                competing.set(false);
            }
        });
    }

    private void runCompetition(Reason reason, String why) throws Exception {
        competitions.incrementAndGet();
        long t0 = System.nanoTime();
        log("COMPETITION started: " + why);

        SnapshotStore snap = new SnapshotStore(store);
        int snapActive = snap.size();
        double snapGhost = snap.idLimit() == 0 ? 0 : 1.0 - (double) snapActive / snap.idLimit();
        trigger.markCompetition(snapActive, snapGhost);

        // test queries + ground truth, both on the frozen snapshot
        phaseNote = "preparing test queries and ground truth";
        List<float[]> queries = evaluationQueries(snap);
        BruteForceHead truthHead = new BruteForceHead(snap);
        List<List<Neighbor>> truth = FitnessEvaluator.computeTruth(truthHead::search, queries, cfg.k);

        // phase A: build every challenger in parallel (the active head keeps serving)
        ExecutorCompletionService<Built> ecs = new ExecutorCompletionService<>(buildPool);
        Map<Future<Built>, Candidate> pending = new HashMap<>();
        for (Candidate c : candidates) {
            Future<Built> f = ecs.submit(() -> {
                ANNHead head = c.factory().apply(c.indexed() ? snap : store);
                long b0 = System.nanoTime();
                if (c.indexed()) head.build();
                return new Built(c, head, (System.nanoTime() - b0) / 1_000_000);
            });
            pending.put(f, c);
        }

        List<Built> built = new ArrayList<>();
        List<Contender> contenders = new ArrayList<>();
        for (int done = 0; done < candidates.size(); done++) {
            Future<Built> f = ecs.take();
            phaseNote = "building heads in parallel (" + (done + 1) + "/" + candidates.size() + " finished)";
            try {
                built.add(f.get());
            } catch (ExecutionException e) {
                Candidate c = pending.get(f);
                Contender failed = new Contender(c.type(), c.type(), null, false);
                failed.failure = String.valueOf(e.getCause());
                contenders.add(failed);
                log("build failed for " + c.type() + ": " + e.getCause());
            }
        }

        // phase B: score everyone one at a time, so builds do not distort latency
        phaseNote = "scoring contenders on " + queries.size() + " queries";
        int maxId = snap.snapshotLimit();

        ANNHead incumbent = manager.activeHead();
        Contender inc = new Contender(incumbent.name(), typeOf(incumbent), incumbent, true);
        inc.m = FitnessEvaluator.evaluate(manager::rawSearch, queries, truth, cfg.k, maxId);
        inc.memoryBytes = incumbent.memoryBytes();
        inc.buildMs = 0;                                     // already paid for
        contenders.add(0, inc);

        for (Built b : built) {
            Contender c = new Contender(b.head().name(), b.candidate().type(), b.head(), false);
            c.m = FitnessEvaluator.evaluate(b.head()::search, queries, truth, cfg.k, maxId);
            c.memoryBytes = b.head().memoryBytes();
            c.buildMs = b.buildMs();
            contenders.add(c);
        }

        score(contenders);

        // decide
        Contender best = null;
        for (Contender c : contenders) {
            if (c.eligible && (best == null || c.score < best.score)) best = c;
        }

        Contender winner = inc;
        String decision;
        if (best == null) {
            decision = "KEEP " + inc.label + " (no eligible contender)";
        } else if (best == inc) {
            decision = "KEEP " + inc.label + " (still the best option)";
        } else if (!inc.eligible) {
            winner = best;
            decision = String.format("SWITCH %s -> %s (active head fell below the %.2f recall gate)",
                    inc.label, best.label, cfg.minRecall);
        } else {
            boolean sameType = best.type.equals(inc.type);
            double margin = (reason == Reason.GHOST_VECTORS && sameType) ? 0.0 : cfg.switchMargin;
            if (best.score <= inc.score * (1.0 - margin)) {
                winner = best;
                decision = String.format("%s %s -> %s (score %.3f vs %.3f)",
                        sameType ? "REBUILD" : "SWITCH", inc.label, best.label, best.score, inc.score);
            } else {
                decision = String.format("KEEP %s (best challenger %s is not %.0f%% better)",
                        inc.label, best.label, margin * 100);
            }
        }

        if (winner != inc) {
            snap.release();                                  // new vectors become visible to the head
            manager.promote(winner.head, snap.snapshotLimit());
            activeSinceMs = System.currentTimeMillis();
            promotions.incrementAndGet();
        }

        final Contender w = winner;
        List<String> board = new ArrayList<>();
        for (Contender c : contenders) board.add(formatContender(c, c == w));
        scoreboard = board;
        lastDecision = decision;
        lastDecisionAtMs = System.currentTimeMillis();

        long secs = (System.nanoTime() - t0) / 1_000_000_000L;
        log("COMPETITION finished in " + secs + "s: " + decision + "\n   " + String.join("\n   ", board));
    }

    /**
     * Recall gate first, then a weighted cost where every term is relative to the worst
     * eligible contender (so each term lies in [0, 1] and lower is better).
     */
    private void score(List<Contender> cs) {
        double maxLat = 0, maxMem = 0, maxBuild = 0;
        for (Contender c : cs) {
            c.eligible = c.failure == null && c.m != null && c.m.recall() >= cfg.minRecall;
            if (c.eligible) {
                maxLat = Math.max(maxLat, c.m.avgLatencyUs());
                maxMem = Math.max(maxMem, c.memoryBytes);
                maxBuild = Math.max(maxBuild, c.buildMs);
            }
        }
        for (Contender c : cs) {
            if (!c.eligible) continue;
            c.score = cfg.weightLatency * ratio(c.m.avgLatencyUs(), maxLat)
                    + cfg.weightMemory * ratio(c.memoryBytes, maxMem)
                    + cfg.weightBuild * ratio(c.buildMs, maxBuild);
        }
    }

    private static double ratio(double v, double max) { return max <= 0 ? 0 : v / max; }

    /** Real recent queries when there are enough, else jittered copies of stored vectors. */
    private List<float[]> evaluationQueries(VectorStore snap) {
        List<float[]> qs = new ArrayList<>(queryLog.recent(cfg.evalQueries));
        if (qs.size() >= cfg.minEvalQueries) return qs;

        Random rnd = new Random(cfg.seed + competitions.get());
        int lim = snap.idLimit();
        int guard = 0;
        while (lim > 0 && qs.size() < cfg.evalQueries && guard++ < cfg.evalQueries * 50) {
            int id = rnd.nextInt(lim);
            if (!snap.isActive(id)) continue;
            float[] v = snap.vector(id).clone();
            for (int d = 0; d < v.length; d++) v[d] += (float) (cfg.jitterSigma * rnd.nextGaussian());
            qs.add(FitnessEvaluator.normalize(v));
        }
        return qs;
    }

    private String formatContender(Contender c, boolean winner) {
        String name = c.label + (c.incumbent ? " (incumbent)" : "");
        String tail;
        if (c.failure != null) {
            tail = "FAILED: " + c.failure;
        } else {
            String verdict = c.eligible
                    ? String.format("score %.3f", c.score)
                    : String.format("DISQUALIFIED (recall < %.2f)", cfg.minRecall);
            tail = String.format("recall %.2f | avg %,.0f us (p95 %,.0f) | mem %s | build %s | %s",
                    c.m.recall(), c.m.avgLatencyUs(), c.m.p95LatencyUs(),
                    HydraStatus.formatBytes(c.memoryBytes), HydraStatus.formatDuration(c.buildMs), verdict);
        }
        return (winner ? "> " : "  ") + String.format("%-32s ", name) + tail;
    }

    // ------------------------------------------------------------------
    // Status and logging
    // ------------------------------------------------------------------

    public HydraStatus status() {
        WindowStats w = windowStats();
        ANNHead head = manager.activeHead();
        List<String> recent;
        synchronized (events) { recent = new ArrayList<>(events); }
        return new HydraStatus(phase, head.name(), phaseNote,
                store.size(), store.idLimit(),
                w.n(), w.recall(), w.avgActiveUs(), w.p95ActiveUs(), w.avgBruteUs(),
                head.memoryBytes(), activeSinceMs,
                promotions.get(), competitions.get(), tunings.get(),
                lastDecision, lastDecisionAtMs, scoreboard, recent);
    }

    private void log(String msg) {
        String stamped = LocalTime.now().withNano(0) + "  " + msg;
        synchronized (events) {
            int nl = stamped.indexOf('\n');
            events.addLast(nl < 0 ? stamped : stamped.substring(0, nl));
            while (events.size() > 8) events.removeFirst();
        }
        listener.accept(stamped);
    }

    private static String typeOf(ANNHead head) {
        String n = head.name();
        int i = n.indexOf('(');
        return i < 0 ? n : n.substring(0, i);
    }

    private static ThreadFactory daemon(String name) {
        return r -> {
            Thread t = new Thread(r, name);
            t.setDaemon(true);
            return t;
        };
    }
}