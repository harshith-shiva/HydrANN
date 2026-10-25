package com.hydrann.annheads;

import java.util.List;

/**
 * ANNHead — the shared interface for every ANN algorithm "head" in HydrANN.
 *
 * <p>Each head encapsulates one approximate (or exact) nearest-neighbour
 * algorithm.  The Hydra controller will call {@link #build} once to index the
 * corpus and {@link #search} for every incoming query.</p>
 *
 * <p>Vectors are represented as {@code float[]} arrays.  All vectors stored in
 * one head must have the same dimensionality.</p>
 */
public interface ANNHead {

    /**
     * Build / train the index from a corpus of vectors.
     *
     * @param vectors list of float vectors (all must have the same length)
     * @throws IllegalArgumentException if {@code vectors} is null, empty,
     *                                  or contains vectors of differing dimension
     */
    void build(List<float[]> vectors);

    /**
     * Return the indices (into the original {@code vectors} list) of the
     * {@code k} nearest neighbours of {@code query}, ordered from closest
     * to furthest.
     *
     * @param query the query vector — must have the same dimension as the
     *              indexed vectors
     * @param k     number of neighbours to return (must be ≥ 1)
     * @return list of at most {@code k} integer indices, nearest-first
     * @throws IllegalStateException    if {@link #build} has not been called yet
     * @throws IllegalArgumentException if {@code query} is null, has the wrong
     *                                  dimension, or {@code k} is less than 1
     */
    List<Integer> search(float[] query, int k);

    /**
     * Human-readable name of the algorithm (e.g. "BruteForce", "IVF", "HNSW").
     *
     * @return algorithm name
     */
    String name();
}
