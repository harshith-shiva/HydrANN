package com.hydrann.controller;

public final class HydraConfig {

    // search
    public int k = 5;
    public int dim = 384;

    // monitoring
    public int sampleEvery = 5;            // every Nth live query becomes a quality sample
    public int windowSize = 30;
    public int minSamples = 10;
    public long checkIntervalMs = 1_000;

    // degradation triggers
    public double minRecall = 0.90;
    public double latencyBudgetUs = 20_000;
    public double growthFactor = 1.5;
    public double maxGhostFraction = 0.30;
    public int consecutiveBreaches = 2;
    public long cooldownMs = 60_000;
    public int minCorpusForCompetition = 100;

    // tune-before-replace
    public double effortStep = 1.5;
    public int maxTuningRounds = 3;
    public double recallHeadroom = 0.05;   // latency tuning only if recall > minRecall + this

    // competition
    public int evalQueries = 60;
    public int minEvalQueries = 30;
    public double jitterSigma = 0.02;      // noise for synthetic eval queries
    public double switchMargin = 0.15;     // challenger must beat the incumbent by this much
    public double weightLatency = 0.6;
    public double weightMemory = 0.2;
    public double weightBuild = 0.2;

    // head parameters used by the default candidates
    public int hnswM = 16;
    public int hnswEfConstruction = 100;
    public int hnswEfSearch = 50;
    public int lshTables = 12;
    public long seed = 42L;

    public static HydraConfig defaults() { return new HydraConfig(); }
}