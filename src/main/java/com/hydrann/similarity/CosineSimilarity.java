package com.hydrann.similarity;

/**
 * CosineSimilarity — utility class for computing cosine similarity and
 * cosine distance between float vectors.
 *
 * <p>Cosine similarity measures the cosine of the angle between two vectors:</p>
 * <pre>
 *   cos(θ) = (A · B) / (||A|| * ||B||)
 * </pre>
 * <p>Result range: [-1, 1] where 1 = identical direction, 0 = orthogonal,
 * -1 = opposite direction.</p>
 *
 * <p>Cosine distance = 1 - cosine similarity, so range is [0, 2].
 * It is used as a proper distance metric (smaller = more similar).</p>
 */
public class CosineSimilarity {

    // Static-only utility class — no instantiation needed
    private CosineSimilarity() {}

    /**
     * Compute the cosine similarity between two vectors.
     *
     * @param a first vector
     * @param b second vector — must have the same length as {@code a}
     * @return cosine similarity in [-1, 1]
     * @throws IllegalArgumentException if vectors are null, empty, or of
     *                                  differing lengths
     */
    public static float similarity(float[] a, float[] b) {
        validate(a, b);

        double dotProduct = 0.0;
        double normA = 0.0;
        double normB = 0.0;

        for (int i = 0; i < a.length; i++) {
            dotProduct += (double) a[i] * b[i];
            normA      += (double) a[i] * a[i];
            normB      += (double) b[i] * b[i];
        }

        if (normA == 0.0 || normB == 0.0) {
            return 0.0f; // zero-vector has no direction → treat as orthogonal
        }

        return (float) (dotProduct / (Math.sqrt(normA) * Math.sqrt(normB)));
    }

    /**
     * Compute the cosine distance between two vectors.
     * <pre>distance = 1 - similarity</pre>
     * Result range: [0, 2] — smaller means more similar.
     *
     * @param a first vector
     * @param b second vector
     * @return cosine distance in [0, 2]
     */
    public static float distance(float[] a, float[] b) {
        return 1.0f - similarity(a, b);
    }

    // -------------------------------------------------------------------------
    // Internal helpers
    // -------------------------------------------------------------------------

    private static void validate(float[] a, float[] b) {
        if (a == null || b == null) {
            throw new IllegalArgumentException("Vectors must not be null");
        }
        if (a.length == 0 || b.length == 0) {
            throw new IllegalArgumentException("Vectors must not be empty");
        }
        if (a.length != b.length) {
            throw new IllegalArgumentException(
                    "Vector dimension mismatch: " + a.length + " vs " + b.length);
        }
    }
}
