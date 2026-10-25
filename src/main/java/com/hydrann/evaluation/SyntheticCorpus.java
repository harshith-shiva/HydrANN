package com.hydrann.evaluation;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

public final class SyntheticCorpus {

    private final int dim;
    private final double noise;
    private final float[][] centers;
    private final Random rng;

    public SyntheticCorpus(int dim, int clusters, double noise, long seed) {
        this.dim = dim;
        this.noise = noise;
        this.rng = new Random(seed);
        this.centers = new float[clusters][];
        for (int c = 0; c < clusters; c++) {
            float[] v = new float[dim];
            for (int d = 0; d < dim; d++) v[d] = (float) rng.nextGaussian();
            centers[c] = FitnessEvaluator.normalize(v);
        }
    }

    /** One new vector (also used as a query: same distribution, fresh noise). */
    public synchronized float[] next() {
        float[] v = centers[rng.nextInt(centers.length)].clone();
        if (rng.nextDouble() < 0.3) {                       // 30%: between two clusters
            float[] other = centers[rng.nextInt(centers.length)];
            double w = rng.nextDouble();
            for (int d = 0; d < dim; d++) v[d] = (float) ((1 - w) * v[d] + w * other[d]);
        }
        for (int d = 0; d < dim; d++) v[d] += (float) (noise * rng.nextGaussian());
        return FitnessEvaluator.normalize(v);
    }

    public synchronized List<float[]> generate(int n) {
        List<float[]> out = new ArrayList<>(n);
        for (int i = 0; i < n; i++) out.add(next());
        return out;
    }
}