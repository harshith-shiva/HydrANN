package com.hydrann.controller;

import com.hydrann.annheads.ANNHead;
import com.hydrann.annheads.Neighbor;
import com.hydrann.vectorstore.SearchResult;
import com.hydrann.vectorstore.VectorStore;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.locks.ReadWriteLock;
import java.util.concurrent.locks.ReentrantReadWriteLock;

public class IndexManager {

    public interface QueryObserver {
        void onQuery(float[] queryVector);
    }

    private static final int CATCH_UP_CHUNK = 256;

    private final VectorStore store;
    private final ReadWriteLock lock = new ReentrantReadWriteLock(true);
    private volatile ANNHead active;
    private volatile QueryObserver observer;
    private int watermark;                       // highest store id the active head has seen

    public IndexManager(VectorStore store, ANNHead initial) {
        this.store = store;
        this.active = initial;
        initial.build();
        this.watermark = store.idLimit() - 1;
    }

    public void setQueryObserver(QueryObserver observer) { this.observer = observer; }

    public ANNHead activeHead() { return active; }

    /** Feeds vectors added since the last call to the active head, in small locked chunks. */
    public void catchUp() {
        while (true) {
            lock.writeLock().lock();
            try {
                int limit = store.idLimit();
                if (watermark + 1 >= limit) return;
                int end = Math.min(limit, watermark + 1 + CATCH_UP_CHUNK);
                for (int id = watermark + 1; id < end; id++) {
                    if (store.isActive(id)) active.insert(id, store.vector(id));
                }
                watermark = end - 1;
            } finally {
                lock.writeLock().unlock();
            }
        }
    }

    /** Search used by the application: returns full chunks, hides deactivated ones. */
    public List<SearchResult> search(float[] query, int k) {
        List<SearchResult> out;
        lock.readLock().lock();
        try {
            out = searchLocked(query, k);
        } finally {
            lock.readLock().unlock();
        }
        QueryObserver o = observer;
        if (o != null) o.onQuery(query);
        return out;
    }

    /** Raw head search under the read lock; used by the controller's monitor and evaluator. */
    public List<Neighbor> rawSearch(float[] query, int k) {
        lock.readLock().lock();
        try {
            return active.search(query, k);
        } finally {
            lock.readLock().unlock();
        }
    }

    /**
     * Makes {@code newHead} the active head. It must already index every vector with
     * id below {@code fromId}. Remaining vectors are inserted first without blocking
     * queries; only the final small delta and the swap happen under the write lock.
     */
    public ANNHead promote(ANNHead newHead, int fromId) {
        int limit = store.idLimit();
        insertRange(newHead, fromId, limit);          // newHead is not visible yet: no lock needed

        lock.writeLock().lock();
        try {
            int finalLimit = store.idLimit();
            insertRange(newHead, limit, finalLimit);
            ANNHead old = active;
            active = newHead;
            watermark = finalLimit - 1;
            return old;
        } finally {
            lock.writeLock().unlock();
        }
    }

    private void insertRange(ANNHead head, int from, int toExclusive) {
        for (int id = from; id < toExclusive; id++) {
            if (store.isActive(id)) head.insert(id, store.vector(id));
        }
    }

    private List<SearchResult> searchLocked(float[] query, int k) {
        ANNHead head = active;
        int fetch = k;
        while (true) {
            List<Neighbor> hits = head.search(query, fetch);
            List<SearchResult> out = new ArrayList<>();
            for (Neighbor n : hits) {
                if (store.isActive(n.id())) {
                    out.add(new SearchResult(store.get(n.id()), n.score()));
                    if (out.size() == k) return out;
                }
            }
            if (hits.size() < fetch || fetch >= store.idLimit()) return out;
            fetch *= 2;
        }
    }
}