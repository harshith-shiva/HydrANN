package com.hydrann.evaluation;

import com.hydrann.annheads.Neighbor;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

public final class FitnessEvaluator {

    private FitnessEvaluator() {}

    @FunctionalInterface
    public interface SearchFn {
        List<Neighbor> search(float[] query, int k);
    }

    public record Measurement(double recall, double avgLatencyUs, double p95LatencyUs) {}

    public static List<List<Neighbor>> computeTruth(SearchFn truthFn, List<float[]> queries, int k) {
        List<List<Neighbor>> out = new ArrayList<>(queries.size());
        for (float[] q : queries) out.add(truthFn.search(q, k));
        return out;
    }

    /** Fraction of the true top-k that the approximate result found (ids >= maxIdExclusive are ignored). */
    public static double recall(List<Neighbor> truth, List<Neighbor> approx, int maxIdExclusive) {
        if (truth.isEmpty()) return 1.0;
        Set<Integer> t = new HashSet<>();
        for (Neighbor n : truth) t.add(n.id());
        Set<Integer> seen = new HashSet<>();
        int hits = 0;
        for (Neighbor n : approx) {
            if (n.id() < maxIdExclusive && t.contains(n.id()) && seen.add(n.id())) hits++;
        }
        return (double) hits / t.size();
    }

    /** One untimed warm-up pass, then one timed pass. */
    public static Measurement evaluate(SearchFn fn, List<float[]> queries,
                                       List<List<Neighbor>> truth, int k, int maxIdExclusive) {
        int n = queries.size();
        if (n == 0) return new Measurement(1.0, 0, 0);

        for (float[] q : queries) fn.search(q, k);

        double[] us = new double[n];
        double recallSum = 0;
        for (int i = 0; i < n; i++) {
            long t0 = System.nanoTime();
            List<Neighbor> r = fn.search(queries.get(i), k);
            us[i] = (System.nanoTime() - t0) / 1_000.0;
            recallSum += recall(truth.get(i), r, maxIdExclusive);
        }
        double sum = 0;
        for (double u : us) sum += u;
        return new Measurement(recallSum / n, sum / n, percentile(us, 0.95));
    }

    public static double percentile(double[] values, double p) {
        if (values.length == 0) return 0;
        double[] s = values.clone();
        Arrays.sort(s);
        int idx = (int) Math.ceil(p * s.length) - 1;
        return s[Math.max(0, Math.min(s.length - 1, idx))];
    }

    public static float[] normalize(float[] v) {
        double sq = 0;
        for (float x : v) sq += (double) x * x;
        double norm = Math.sqrt(sq);
        if (norm < 1e-12) return v;
        float[] out = new float[v.length];
        for (int i = 0; i < v.length; i++) out[i] = (float) (v[i] / norm);
        return out;
    }
}