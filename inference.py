# inference_solver.py
# Inference-only Hangman solver with temperature scaling and candidate masking.

import os, json, logging
import numpy as np
import tensorflow as tf

from hangman_layers import TransformerBlock, GLUBlock, AttentionPool1D, PositionalEmbedding
from hangman_commons import (
    ALPHABET, LETTER_TO_INDEX, load_dictionary_words, group_words_by_length,
    encode_state_as_model_inputs, list_dictionary_candidates_for_mask
)

# ---------------- Logging ----------------
LOG_LEVEL = os.getenv("HANGMAN_LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s"
)
log = logging.getLogger("hangman.inference")

# ---------------- Defaults ----------------
DEFAULT_MODEL_PATH = "hangman_model.keras"
DEFAULT_META_PATH = DEFAULT_MODEL_PATH + ".meta.json"
DEFAULT_DICT_PATH = "words_250000_train.txt"


class HangmanSolver:
    """
    Neural Hangman solver with candidate filtering.

    Method:
        predict_next_letter(mask: str, guessed_letters: set[str],
                            tries_left: int, temperature: float = 1.0) -> str
    """

    def __init__(self,
                 model_path: str = DEFAULT_MODEL_PATH,
                 meta_path: str = DEFAULT_META_PATH,
                 dict_path: str = DEFAULT_DICT_PATH):
        # Load metadata
        with open(meta_path, "r") as f:
            meta = json.load(f)
        self.max_seq_len = int(meta["max_len"])

        # Load trained model with custom layers
        self.model = tf.keras.models.load_model(
            model_path,
            custom_objects={
                "TransformerBlock": TransformerBlock,
                "GLUBlock": GLUBlock,
                "AttentionPool1D": AttentionPool1D,
                "PositionalEmbedding": PositionalEmbedding,
            }
        )

        # Load and bucket dictionary words
        words = load_dictionary_words(dict_path)
        self.words_by_length = group_words_by_length(words)

        log.info("HangmanSolver loaded: model=%s max_len=%d dictionary_words=%d",
                 model_path, self.max_seq_len,
                 sum(len(v) for v in self.words_by_length.values()))

    # ---------------- Internal helpers ----------------
    @staticmethod
    def _apply_temperature_scaling(probs: np.ndarray, temperature: float) -> np.ndarray:
        """Apply temperature scaling to a probability distribution."""
        if temperature <= 0:
            return probs
        if abs(temperature - 1.0) < 1e-8:
            return probs
        adjusted = np.power(np.maximum(probs, 1e-12), 1.0 / float(temperature))
        return adjusted / adjusted.sum()

    # ---------------- Public API ----------------
    def predict_next_letter(self,
                            mask: str,
                            guessed_letters: set[str],
                            tries_left: int,
                            temperature: float = 1.0) -> str:
        """
        Predict the next letter for a given Hangman state.

        Args:
            mask: masked word (e.g., "s__ne")
            guessed_letters: set of letters already guessed
            tries_left: number of incorrect guesses remaining
            temperature: controls exploration (T>1 = smoother, T<1 = sharper)

        Returns:
            str: predicted letter
        """
        # Encode state into model inputs
        seq_onehot, guessed_vec, meta_vec = encode_state_as_model_inputs(
            mask, guessed_letters, tries_left, self.max_seq_len
        )
        probs = self.model.predict([seq_onehot[None], guessed_vec[None], meta_vec[None]], verbose=0)[0]

        # Suppress already-guessed letters
        for c in guessed_letters:
            if c in LETTER_TO_INDEX:
                probs[LETTER_TO_INDEX[c]] = 0.0

        # Restrict to letters consistent with candidate words
        candidates = list_dictionary_candidates_for_mask(mask, guessed_letters, self.words_by_length)
        if candidates:
            allowed = set("".join(candidates))
            for ch in ALPHABET:
                if ch not in allowed:
                    probs[LETTER_TO_INDEX[ch]] = 0.0

        # Apply temperature scaling
        probs = self._apply_temperature_scaling(probs, temperature)

        return ALPHABET[int(np.argmax(probs))]
