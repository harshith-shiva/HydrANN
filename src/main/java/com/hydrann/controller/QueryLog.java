package com.hydrann.controller;

import java.util.ArrayList;
import java.util.List;

/** Ring buffer of the most recent live query vectors, used as the competition's test set. */
public final class QueryLog {

    private final float[][] ring;
    private int next = 0;
    private int count = 0;

    public QueryLog(int capacity) { this.ring = new float[capacity][]; }

    public synchronized void add(float[] q) {
        ring[next] = q;
        next = (next + 1) % ring.length;
        if (count < ring.length) count++;
    }

    public synchronized int size() { return count; }

    /** The most recent {@code max} queries, oldest first. */
    public synchronized List<float[]> recent(int max) {
        int n = Math.min(max, count);
        List<float[]> out = new ArrayList<>(n);
        for (int i = n; i >= 1; i--) out.add(ring[(next - i + ring.length) % ring.length]);
        return out;
    }
}