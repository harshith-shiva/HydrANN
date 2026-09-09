package com.hydrann.embedding;

import java.io.FileWriter;
import java.io.PrintWriter;

public class EmbeddingTest {

    public static void main(String[] args) throws Exception {

        try (MiniLMEmbedder embedder = new MiniLMEmbedder()) {

            String text =
                    "The quick brown fox jumps over the lazy dog.";

            float[] embedding = embedder.embed(text);

            System.out.println(
                    "Embedding dimensions: "
                    + embedding.length
            );

            System.out.println("\nFirst 10 values:");

            for (int i = 0; i < 10; i++) {
                System.out.printf(
                        "%.8f ",
                        embedding[i]
                );
            }

            System.out.println();

            // Save embedding
            try (PrintWriter writer =
                         new PrintWriter(
                                 new FileWriter(
                                         "java_embedding.txt"))) {

                for (float value : embedding) {
                    writer.println(value);
                }
            }
        }
    }
}