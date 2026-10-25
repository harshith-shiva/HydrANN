package com.hydrann.vectorstore;

import com.hydrann.ingestion.ChunkEmbedding;

import java.util.List;

public final class SnapshotStore implements VectorStore {

    private final VectorStore delegate;
    private final int limit;
    private volatile boolean frozen = true;

    public SnapshotStore(VectorStore delegate) {
        this.delegate = delegate;
        this.limit = delegate.idLimit();
    }

    public int snapshotLimit() { return limit; }

    /** Called right before the head goes live: ids added after the snapshot become visible. */
    public void release() { frozen = false; }

    @Override public int idLimit() { return frozen ? limit : delegate.idLimit(); }

    @Override public boolean isActive(int id) {
        return id >= 0 && id < idLimit() && delegate.isActive(id);
    }

    @Override public float[] vector(int id) { return delegate.vector(id); }

    @Override public ChunkEmbedding get(int id) { return delegate.get(id); }

    @Override public int size() {
        int n = 0;
        int lim = idLimit();
        for (int id = 0; id < lim; id++) if (delegate.isActive(id)) n++;
        return n;
    }

    @Override public void add(ChunkEmbedding e) { throw new UnsupportedOperationException("read-only view"); }
    @Override public void addAll(List<ChunkEmbedding> es) { throw new UnsupportedOperationException("read-only view"); }
    @Override public List<SearchResult> search(float[] q, int k) { throw new UnsupportedOperationException("use an ANNHead"); }
}