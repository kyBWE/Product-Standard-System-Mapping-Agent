from __future__ import annotations
import logging
import os

import numpy as np
import onnxruntime as ort

logger = logging.getLogger("ONNXEmbedder")

DEFAULT_MODEL_DIR = "models/bge-small-zh-v1.5"
DEFAULT_ONNX_FILE = "onnx/model_int8.onnx"
EMBEDDING_DIM = 512
MAX_SEQ_LEN = 128
CLS_TOKEN_ID = 101
SEP_TOKEN_ID = 102
PAD_TOKEN_ID = 0


class ONNXEmbedder:
    def __init__(self, model_dir: str = DEFAULT_MODEL_DIR, onnx_file: str = DEFAULT_ONNX_FILE):
        self._model_dir = model_dir
        onnx_path = os.path.join(model_dir, onnx_file)

        if not os.path.exists(onnx_path):
            raise FileNotFoundError(f"ONNX model not found: {onnx_path}")

        logger.info(f"Loading ONNX model: {onnx_path}")
        self._session = ort.InferenceSession(onnx_path)
        self._input_names = [i.name for i in self._session.get_inputs()]
        logger.info(f"ONNX inputs: {self._input_names}")

        self._tokenizer = None
        try:
            from tokenizers import Tokenizer
            tokenizer_path = os.path.join(model_dir, "tokenizer.json")
            if os.path.exists(tokenizer_path):
                self._tokenizer = Tokenizer.from_file(tokenizer_path)
                self._tokenizer.enable_truncation(max_length=MAX_SEQ_LEN - 2)
                self._tokenizer.enable_padding(length=MAX_SEQ_LEN)
                logger.info(f"WordPiece tokenizer loaded from: {tokenizer_path}")
            else:
                logger.warning(f"tokenizer.json not found: {tokenizer_path}, falling back to char-level")
        except Exception as e:
            logger.warning(f"Failed to load WordPiece tokenizer: {e}, falling back to char-level")

        if self._tokenizer is None:
            vocab_path = os.path.join(model_dir, "vocab.txt")
            if os.path.exists(vocab_path):
                with open(vocab_path, "r", encoding="utf-8") as f:
                    self._vocab = {line.strip(): idx for idx, line in enumerate(f)}
                logger.info(f"Char-level vocab loaded: {len(self._vocab)} tokens")
            else:
                self._vocab = {}
                logger.warning(f"vocab.txt not found: {vocab_path}")

    def _tokenize_wordpiece(self, text: str) -> dict[str, np.ndarray]:
        encoding = self._tokenizer.encode(text)
        ids = [CLS_TOKEN_ID] + encoding.ids[:MAX_SEQ_LEN - 2] + [SEP_TOKEN_ID]
        seq_len = len(ids)
        attention_mask = [1] * seq_len + [0] * (MAX_SEQ_LEN - seq_len)
        token_type_ids = [0] * MAX_SEQ_LEN
        ids += [PAD_TOKEN_ID] * (MAX_SEQ_LEN - seq_len)
        return {
            "input_ids": np.array([ids], dtype=np.int64),
            "attention_mask": np.array([attention_mask], dtype=np.int64),
            "token_type_ids": np.array([token_type_ids], dtype=np.int64),
        }

    def _tokenize_wordpiece_batch(self, texts: list[str]) -> dict[str, np.ndarray]:
        all_ids = []
        all_attention_mask = []
        all_token_type_ids = []
        for text in texts:
            encoding = self._tokenizer.encode(text)
            ids = [CLS_TOKEN_ID] + encoding.ids[:MAX_SEQ_LEN - 2] + [SEP_TOKEN_ID]
            seq_len = len(ids)
            attention_mask = [1] * seq_len + [0] * (MAX_SEQ_LEN - seq_len)
            token_type_ids = [0] * MAX_SEQ_LEN
            ids += [PAD_TOKEN_ID] * (MAX_SEQ_LEN - seq_len)
            all_ids.append(ids)
            all_attention_mask.append(attention_mask)
            all_token_type_ids.append(token_type_ids)
        return {
            "input_ids": np.array(all_ids, dtype=np.int64),
            "attention_mask": np.array(all_attention_mask, dtype=np.int64),
            "token_type_ids": np.array(all_token_type_ids, dtype=np.int64),
        }

    def _tokenize_char(self, text: str) -> dict[str, np.ndarray]:
        tokens = [self._vocab.get(c, 1) for c in text[: MAX_SEQ_LEN - 2]]
        input_ids = [CLS_TOKEN_ID] + tokens + [SEP_TOKEN_ID]
        seq_len = len(input_ids)
        attention_mask = [1] * seq_len + [0] * (MAX_SEQ_LEN - seq_len)
        token_type_ids = [0] * MAX_SEQ_LEN
        input_ids += [PAD_TOKEN_ID] * (MAX_SEQ_LEN - seq_len)
        return {
            "input_ids": np.array([input_ids], dtype=np.int64),
            "attention_mask": np.array([attention_mask], dtype=np.int64),
            "token_type_ids": np.array([token_type_ids], dtype=np.int64),
        }

    def _tokenize_char_batch(self, texts: list[str]) -> dict[str, np.ndarray]:
        all_input_ids = []
        all_attention_mask = []
        all_token_type_ids = []
        for text in texts:
            tokens = [self._vocab.get(c, 1) for c in text[: MAX_SEQ_LEN - 2]]
            input_ids = [CLS_TOKEN_ID] + tokens + [SEP_TOKEN_ID]
            seq_len = len(input_ids)
            attention_mask = [1] * seq_len + [0] * (MAX_SEQ_LEN - seq_len)
            token_type_ids = [0] * MAX_SEQ_LEN
            input_ids += [PAD_TOKEN_ID] * (MAX_SEQ_LEN - seq_len)
            all_input_ids.append(input_ids)
            all_attention_mask.append(attention_mask)
            all_token_type_ids.append(token_type_ids)
        return {
            "input_ids": np.array(all_input_ids, dtype=np.int64),
            "attention_mask": np.array(all_attention_mask, dtype=np.int64),
            "token_type_ids": np.array(all_token_type_ids, dtype=np.int64),
        }

    def _tokenize(self, text: str) -> dict[str, np.ndarray]:
        if self._tokenizer is not None:
            return self._tokenize_wordpiece(text)
        return self._tokenize_char(text)

    def _tokenize_batch(self, texts: list[str]) -> dict[str, np.ndarray]:
        if self._tokenizer is not None:
            return self._tokenize_wordpiece_batch(texts)
        return self._tokenize_char_batch(texts)

    def embed(self, text: str) -> np.ndarray:
        inputs = self._tokenize(text)
        outputs = self._session.run(None, inputs)
        last_hidden = outputs[0]
        attention_mask = inputs["attention_mask"].astype(np.float32)
        mask_expanded = attention_mask[:, :, np.newaxis]
        sum_embeddings = np.sum(last_hidden * mask_expanded, axis=1)
        sum_mask = np.clip(mask_expanded.sum(axis=1), a_min=1e-9, a_max=None)
        mean_embedding = sum_embeddings / sum_mask
        norm = np.linalg.norm(mean_embedding, axis=1, keepdims=True)
        normalized = mean_embedding / np.clip(norm, a_min=1e-9, a_max=None)
        return normalized[0]

    def embed_batch(self, texts: list[str], batch_size: int = 8) -> list[np.ndarray]:
        results = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            batch_inputs = self._tokenize_batch(batch)
            outputs = self._session.run(None, batch_inputs)
            last_hidden = outputs[0]
            attention_mask = batch_inputs["attention_mask"].astype(np.float32)
            mask_expanded = attention_mask[:, :, np.newaxis]
            sum_embeddings = np.sum(last_hidden * mask_expanded, axis=1)
            sum_mask = np.clip(mask_expanded.sum(axis=1), a_min=1e-9, a_max=None)
            mean_embeddings = sum_embeddings / sum_mask
            norms = np.linalg.norm(mean_embeddings, axis=1, keepdims=True)
            normalized = mean_embeddings / np.clip(norms, a_min=1e-9, a_max=None)
            results.extend([normalized[j] for j in range(len(batch))])
            if (i + batch_size) % 1000 == 0:
                logger.info(f"Embedding progress: {min(i + batch_size, len(texts))}/{len(texts)}")
        return results
