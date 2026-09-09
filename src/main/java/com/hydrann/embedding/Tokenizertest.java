package com.hydrann.embedding;

import ai.djl.huggingface.tokenizers.Encoding;
import ai.djl.huggingface.tokenizers.HuggingFaceTokenizer;

import java.nio.file.Paths;
import java.util.Arrays;

public class Tokenizertest {

    public static void main(String[] args) throws Exception {

        String tokenizerPath =
                "models/all-MiniLM-L6-v2/tokenizer.json";

        HuggingFaceTokenizer tokenizer =
                HuggingFaceTokenizer.newInstance(
                        Paths.get(tokenizerPath)
                );

        String text =
                "The quick brown fox jumps over the lazy dog.";

        Encoding encoding = tokenizer.encode(text);

        System.out.println("TOKENS:");
        System.out.println(Arrays.toString(encoding.getTokens()));

        System.out.println("\nINPUT IDS:");
        System.out.println(Arrays.toString(encoding.getIds()));

        System.out.println("\nATTENTION MASK:");
        System.out.println(Arrays.toString(encoding.getAttentionMask()));

        System.out.println("\nTYPE IDS:");
        System.out.println(Arrays.toString(encoding.getTypeIds()));

        tokenizer.close();
    }
}