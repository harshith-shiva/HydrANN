package com.hydrann.embedding;

import ai.djl.huggingface.tokenizers.Encoding;
import ai.djl.huggingface.tokenizers.HuggingFaceTokenizer;
import ai.onnxruntime.*;

import java.nio.file.Paths;
import java.util.HashMap;
import java.util.Map;

public class MiniLMEmbedder implements AutoCloseable {

    private final OrtEnvironment environment;
    private final OrtSession session;
    private final HuggingFaceTokenizer tokenizer;

    public MiniLMEmbedder() throws Exception {

        // Load ONNX environment
        environment = OrtEnvironment.getEnvironment();

        // Load MiniLM ONNX model
        session = environment.createSession(
                "models/all-MiniLM-L6-v2/onnx/model.onnx",
                new OrtSession.SessionOptions()
        );

        // Load Hugging Face tokenizer
        tokenizer = HuggingFaceTokenizer.newInstance(
                Paths.get("models/all-MiniLM-L6-v2/tokenizer.json")
        );
    }

    public float[] embed(String text) throws Exception {

        // -----------------------------------------
        // 1. TOKENIZE
        // -----------------------------------------

        Encoding encoding = tokenizer.encode(text);

        long[] inputIds = encoding.getIds();
        long[] attentionMask = encoding.getAttentionMask();
        long[] typeIds = encoding.getTypeIds();

        // Add batch dimension
        long[][] inputIdsBatch = {inputIds};
        long[][] attentionMaskBatch = {attentionMask};
        long[][] typeIdsBatch = {typeIds};

        // -----------------------------------------
        // 2. CREATE ONNX INPUTS
        // -----------------------------------------

        Map<String, OnnxTensor> inputs = new HashMap<>();

        inputs.put(
                "input_ids",
                OnnxTensor.createTensor(environment, inputIdsBatch)
        );

        inputs.put(
                "attention_mask",
                OnnxTensor.createTensor(environment, attentionMaskBatch)
        );

        inputs.put(
                "token_type_ids",
                OnnxTensor.createTensor(environment, typeIdsBatch)
        );

        // -----------------------------------------
        // 3. RUN MINILM
        // -----------------------------------------

        try (OrtSession.Result result = session.run(inputs)) {

            // Model output:
            // [batch_size, sequence_length, 384]

            float[][][] lastHiddenState =
                    (float[][][]) result.get(0).getValue();

            // -----------------------------------------
            // 4. MEAN POOLING
            // -----------------------------------------

            float[] embedding =
                    meanPool(lastHiddenState, attentionMask);

            // -----------------------------------------
            // 5. NORMALIZATION
            // -----------------------------------------

            normalize(embedding);

            return embedding;
        }
        finally {

            // Release ONNX tensors
            for (OnnxTensor tensor : inputs.values()) {
                tensor.close();
            }
        }
    }

    private float[] meanPool(
            float[][][] tokenEmbeddings,
            long[] attentionMask) {

        int sequenceLength = tokenEmbeddings[0].length;
        int dimensions = tokenEmbeddings[0][0].length;

        float[] pooled = new float[dimensions];

        int validTokens = 0;

        for (int i = 0; i < sequenceLength; i++) {

            if (attentionMask[i] == 1) {

                validTokens++;

                for (int j = 0; j < dimensions; j++) {
                    pooled[j] += tokenEmbeddings[0][i][j];
                }
            }
        }

        // Divide by number of real tokens
        for (int j = 0; j < dimensions; j++) {
            pooled[j] /= validTokens;
        }

        return pooled;
    }

    private void normalize(float[] embedding) {

        double sum = 0.0;

        for (float value : embedding) {
            sum += value * value;
        }

        double norm = Math.sqrt(sum);

        for (int i = 0; i < embedding.length; i++) {
            embedding[i] /= (float) norm;
        }
    }

    @Override
    public void close() throws Exception {
        tokenizer.close();
        session.close();
        environment.close();
    }
}