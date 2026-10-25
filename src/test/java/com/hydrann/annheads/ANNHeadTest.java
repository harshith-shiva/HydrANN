package com.hydrann.annheads;

import java.util.*;

/**
 * ANNHeadTest — manual smoke tests for BruteForceHead, IvfHead, and HnswHead.
 *
 * <p>Not a JUnit test (no test framework in pom.xml yet).  Run via
 * {@code mvn exec:java -Dexec.mainClass=com.hydrann.annheads.ANNHeadTest}
 * or add this class to the test source set once JUnit is added.</p>
 */
public class ANNHeadTest {

    public static void main(String[] args) {
        System.out.println("========== HydrANN — ANNHead Smoke Tests ==========\n");

        List<float[]> corpus = buildCorpus();
        float[] query        = new float[]{ 1.0f, 0.0f, 0.0f }; // closest to index 0

        System.out.println("Corpus size : " + corpus.size() + " vectors (dim=3)");
        System.out.println("Query       : " + Arrays.toString(query));
        System.out.println("Expected top-1: index 0 [1.0, 0.0, 0.0]\n");

        testHead(new BruteForceHead(),        corpus, query, 3);
        testHead(new IvfHead(4, 2, 25, 42L),  corpus, query, 3);
        testHead(new HnswHead(4, 50, 20, 42L),corpus, query, 3);

        System.out.println("========== All tests completed ==========");
    }

    // -------------------------------------------------------------------------

    private static void testHead(ANNHead head, List<float[]> corpus,
                                  float[] query, int k) {
        System.out.println("--- " + head.name() + " ---");
        head.build(corpus);
        List<Integer> results = head.search(query, k);
        System.out.println("  Top-" + k + " results: " + results);
        for (int idx : results) {
            System.out.printf("    idx=%d  vec=%s%n",
                    idx, Arrays.toString(corpus.get(idx)));
        }
        System.out.println();
    }

    private static List<float[]> buildCorpus() {
        List<float[]> c = new ArrayList<>();
        c.add(new float[]{ 1.0f,  0.0f,  0.0f });  // 0 — along +x
        c.add(new float[]{ 0.9f,  0.1f,  0.0f });  // 1 — close to +x
        c.add(new float[]{ 0.0f,  1.0f,  0.0f });  // 2 — along +y
        c.add(new float[]{ 0.0f,  0.9f,  0.1f });  // 3 — close to +y
        c.add(new float[]{ 0.0f,  0.0f,  1.0f });  // 4 — along +z
        c.add(new float[]{-1.0f,  0.0f,  0.0f });  // 5 — opposite of query
        c.add(new float[]{ 0.5f,  0.5f,  0.0f });  // 6 — diagonal xy
        c.add(new float[]{ 0.0f, -1.0f,  0.0f });  // 7 — -y
        return c;
    }
}
