package com.hydrann.vectorstore;

import com.hydrann.ingestion.ChunkEmbedding;
import com.hydrann.ingestion.TextChunk;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Random;

public final class SyntheticVectorStore implements VectorStore {

    private record State(float[][] vectors, boolean[] active) {}

    private volatile State state = new State(new float[0][], new boolean[0]);

    public synchronized void append(List<float[]> batch) {
        State s = state;
        int n = s.vectors().length;
        float[][] v = Arrays.copyOf(s.vectors(), n + batch.size());
        boolean[] a = Arrays.copyOf(s.active(), n + batch.size());
        for (int i = 0; i < batch.size(); i++) {
            v[n + i] = batch.get(i);
            a[n + i] = true;
        }
        state = new State(v, a);
    }

    /** Simulates deleted documents: deactivates a random fraction of the active vectors. */
    public synchronized int deactivateRandom(double fraction, long seed) {
        State s = state;
        boolean[] a = s.active().clone();
        List<Integer> ids = new ArrayList<>();
        for (int i = 0; i < a.length; i++) if (a[i]) ids.add(i);
        Collections.shuffle(ids, new Random(seed));
        int count = (int) Math.round(ids.size() * fraction);
        for (int i = 0; i < count; i++) a[ids.get(i)] = false;
        state = new State(s.vectors(), a);
        return count;
    }

    @Override public int idLimit() { return state.vectors().length; }

    @Override public boolean isActive(int id) {
        boolean[] a = state.active();
        return id >= 0 && id < a.length && a[id];
    }

    @Override public float[] vector(int id) { return state.vectors()[id]; }

    @Override public ChunkEmbedding get(int id) {
        return new ChunkEmbedding(
                new TextChunk("synthetic", 0, id, "Synthetic chunk #" + id),
                vector(id));
    }

    @Override public int size() {
        int n = 0;
        for (boolean b : state.active()) if (b) n++;
        return n;
    }

    @Override public void add(ChunkEmbedding e) { append(List.of(e.getEmbedding())); }

    @Override public void addAll(List<ChunkEmbedding> es) {
        List<float[]> batch = new ArrayList<>();
        for (ChunkEmbedding e : es) batch.add(e.getEmbedding());
        append(batch);
    }

    @Override public List<SearchResult> search(float[] q, int k) {
        throw new UnsupportedOperationException("use an ANNHead");
    }
}