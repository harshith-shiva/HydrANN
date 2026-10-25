package com.hydrann.controller;

public final class DegradationTrigger {

    public enum Reason {
        NONE, INITIAL_EVALUATION, LOW_RECALL, HIGH_LATENCY,
        ANN_SLOWER_THAN_BRUTE, CORPUS_GROWTH, GHOST_VECTORS
    }

    public record Observation(int samples, long sampleSerial, double recall,
                              double avgLatencyUs, double p95LatencyUs, double bruteAvgUs,
                              int activeCount, int storedCount, boolean activeIsBrute) {
        public double ghostFraction() {
            return storedCount == 0 ? 0.0 : 1.0 - (double) activeCount / storedCount;
        }
    }

    private final HydraConfig cfg;
    private int streak = 0;
    private long lastSerial = -1;
    private int baselineCount = 0;      // active vectors at the last competition (0 = none yet)
    private double ghostFloor = 0.0;

    public DegradationTrigger(HydraConfig cfg) { this.cfg = cfg; }

    /** Returns a reason only after {@code consecutiveBreaches} confirmed breaches in a row. */
    public synchronized Reason check(Observation o) {
        Reason r = classify(o);
        if (r == Reason.NONE) { streak = 0; return r; }

        boolean windowBased = r == Reason.LOW_RECALL || r == Reason.HIGH_LATENCY
                || r == Reason.ANN_SLOWER_THAN_BRUTE;
        if (windowBased) {
            if (o.sampleSerial() == lastSerial) return Reason.NONE;   // no new samples to judge
            lastSerial = o.sampleSerial();
        }

        streak++;
        if (streak >= cfg.consecutiveBreaches) { streak = 0; return r; }
        return Reason.NONE;
    }

    private Reason classify(Observation o) {
        if (o.activeCount() < cfg.minCorpusForCompetition) return Reason.NONE;

        if (o.samples() >= cfg.minSamples) {
            if (!o.activeIsBrute() && o.recall() < cfg.minRecall) return Reason.LOW_RECALL;
            if (o.p95LatencyUs() > cfg.latencyBudgetUs) return Reason.HIGH_LATENCY;
            if (!o.activeIsBrute() && o.bruteAvgUs() > 0
                    && o.avgLatencyUs() >= o.bruteAvgUs() * 1.2) return Reason.ANN_SLOWER_THAN_BRUTE;
        }

        if (baselineCount == 0) return Reason.INITIAL_EVALUATION;
        if (o.activeCount() >= baselineCount * cfg.growthFactor) return Reason.CORPUS_GROWTH;
        if (!o.activeIsBrute()
                && o.ghostFraction() > Math.max(cfg.maxGhostFraction, ghostFloor + 0.10)) {
            return Reason.GHOST_VECTORS;
        }
        return Reason.NONE;
    }

    /** Called when a competition starts: growth and ghost triggers measure from here. */
    public synchronized void markCompetition(int activeCount, double ghostFraction) {
        this.baselineCount = Math.max(1, activeCount);
        this.ghostFloor = ghostFraction;
    }

    public synchronized void reset() { streak = 0; }

    public synchronized boolean breaching() { return streak > 0; }

    public synchronized String explain(Reason r, Observation o) {
        return switch (r) {
            case INITIAL_EVALUATION -> String.format("initial evaluation on %,d active vectors", o.activeCount());
            case LOW_RECALL -> String.format("recall %.2f < target %.2f", o.recall(), cfg.minRecall);
            case HIGH_LATENCY -> String.format("p95 latency %,.0f us > budget %,.0f us", o.p95LatencyUs(), cfg.latencyBudgetUs);
            case ANN_SLOWER_THAN_BRUTE -> String.format("active head (%,.0f us) is slower than brute force (%,.0f us)", o.avgLatencyUs(), o.bruteAvgUs());
            case CORPUS_GROWTH -> String.format("corpus grew to %,d active vectors (was %,d at the last competition)", o.activeCount(), baselineCount);
            case GHOST_VECTORS -> String.format("%.0f%% of stored vectors are deactivated", o.ghostFraction() * 100);
            case NONE -> "none";
        };
    }
}