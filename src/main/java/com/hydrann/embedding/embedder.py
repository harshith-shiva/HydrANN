import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer


MODEL_PATH = "models/all-MiniLM-L6-v2"
ONNX_PATH = f"{MODEL_PATH}/onnx/model.onnx"


class MiniLMEmbedder:

    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

        self.session = ort.InferenceSession(
            ONNX_PATH,
            providers=["CPUExecutionProvider"]
        )

        print("ONNX inputs:")
        for inp in self.session.get_inputs():
            print(inp.name, inp.shape, inp.type)

        print("\nONNX outputs:")
        for output in self.session.get_outputs():
            print(output.name, output.shape, output.type)

    def embed(self, text):

        encoded = self.tokenizer(
            text,
            padding=True,
            truncation=True,
            return_tensors="np"
        )

        inputs = {
            "input_ids": encoded["input_ids"].astype(np.int64),
            "attention_mask": encoded["attention_mask"].astype(np.int64)
        }

        # Add token_type_ids if the ONNX model expects them
        input_names = {
            inp.name for inp in self.session.get_inputs()
        }

        if "token_type_ids" in input_names:
            inputs["token_type_ids"] = encoded["token_type_ids"].astype(np.int64)

        outputs = self.session.run(
            None,
            inputs
        )

        token_embeddings = outputs[0]

        pooled = self.mean_pool(
            token_embeddings,
            encoded["attention_mask"]
        )
        return self.normalize(pooled)

        

    @staticmethod
    def mean_pool(token_embeddings, attention_mask):

        mask = attention_mask[:, :, None]

        mask = mask.astype(np.float32)

        summed = np.sum(
            token_embeddings * mask,
            axis=1
        )

        counts = np.clip(
            mask.sum(axis=1),
            a_min=1e-9,
            a_max=None
        )

        return summed / counts

    @staticmethod
    def normalize(embedding):

        norm = np.linalg.norm(
            embedding,
            axis=1,
            keepdims=True
        )

        return embedding / np.clip(
            norm,
            a_min=1e-12,
            a_max=None
        )

    @staticmethod
    def cosine_similarity(embedding1, embedding2):
        return np.sum(embedding1 * embedding2)