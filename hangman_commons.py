# hangman_commons.py
# Core utilities for Hangman dataset creation and model preprocessing.

import re
import numpy as np

# ---------------- Alphabet and Tokenization ----------------
ALPHABET = "abcdefghijklmnopqrstuvwxyz"
LETTER_TO_INDEX = {c: i for i, c in enumerate(ALPHABET)}

UNDERSCORE_TOKEN_INDEX = 26  # index for "_"
PAD_TOKEN_INDEX = 27         # index for padding
TOKEN_VOCAB_SIZE = 28        # 26 letters + underscore + pad
MAX_INCORRECT_TRIES = 6      # max number of wrong guesses allowed


# ---------------- Dictionary Utilities ----------------
def load_dictionary_words(filepath: str) -> list[str]:
    """Read dictionary file and return a list of valid lowercase words (a–z only)."""
    with open(filepath, "r") as f:
        words = [w.strip().lower() for w in f if w.strip()]
    return [w for w in words if all("a" <= c <= "z" for c in w)]


def group_words_by_length(words: list[str]) -> dict[int, list[str]]:
    """Group words into buckets keyed by their length."""
    buckets: dict[int, list[str]] = {}
    for word in words:
        buckets.setdefault(len(word), []).append(word)
    return buckets


# ---------------- Mask Handling ----------------
def mask_to_regex(mask: str) -> str:
    """Convert a mask like 'a__le' into regex '^a..le$' for matching."""
    return "^" + mask.replace("_", ".") + "$"


def find_candidate_words(mask: str, guessed_letters: set[str], words_by_length: dict[int, list[str]]) -> list[str]:
    """
    Return all words of correct length consistent with a given mask and guessed letters.
    - mask: current masked word (e.g., 'a__le').
    - guessed_letters: set of already guessed characters.
    - words_by_length: dict of word lists grouped by length.
    """
    regex = re.compile(mask_to_regex(mask))
    present = {ch for ch in mask if ch != "_"}
    absent = {g for g in guessed_letters if g not in present}

    candidates = []
    for word in words_by_length.get(len(mask), []):
        if regex.match(word) and not any(b in word for b in absent):
            candidates.append(word)
    return candidates


# ---------------- Feature Encoding ----------------
def encode_state_as_model_inputs(mask: str, guessed_letters: set[str], tries_left: int, max_len: int):
    """
    Encode current game state into model input features:
    - seq: one-hot sequence of shape (max_len, TOKEN_VOCAB_SIZE)
    - guessed_vector: binary vector of guessed letters (26,)
    - meta: normalized features [word_length/max_len, tries_left/MAX_INCORRECT_TRIES]
    """
    word_length = len(mask)

    # Encode sequence with letters, underscores, and padding
    sequence_matrix = np.zeros((max_len, TOKEN_VOCAB_SIZE), dtype=np.float32)
    for i in range(max_len):
        if i >= word_length:
            sequence_matrix[i, PAD_TOKEN_INDEX] = 1.0
        else:
            ch = mask[i]
            if ch == "_":
                sequence_matrix[i, UNDERSCORE_TOKEN_INDEX] = 1.0
            else:
                sequence_matrix[i, LETTER_TO_INDEX[ch]] = 1.0

    # Encode guessed letters
    guessed_vector = np.zeros((26,), dtype=np.float32)
    for c in guessed_letters:
        if c in LETTER_TO_INDEX:
            guessed_vector[LETTER_TO_INDEX[c]] = 1.0

    # Encode meta features
    tries_normalized = max(0, min(MAX_INCORRECT_TRIES, int(tries_left))) / MAX_INCORRECT_TRIES
    meta_features = np.array([word_length / max_len, tries_normalized], dtype=np.float32)

    return sequence_matrix, guessed_vector, meta_features


# ---------------- Frequency & Rarity Helpers ----------------
def compute_letter_frequencies(words: list[str]) -> np.ndarray:
    """
    Compute normalized frequency distribution over 26 letters.
    Counts each letter once per word (avoiding length bias).
    """
    counts = np.zeros(26, dtype=np.float64)
    for word in words:
        for ch in set(word):
            if ch in LETTER_TO_INDEX:
                counts[LETTER_TO_INDEX[ch]] += 1.0
    counts = counts + 1e-8
    return counts / counts.sum()


def compute_word_rarity(word: str, letter_frequencies: np.ndarray, exponent: float = 0.5) -> float:
    """
    Compute rarity weight for a word: higher for words with rare letters.
    - word: input word
    - letter_frequencies: vector of size (26,) with letter frequencies
    - exponent: controls scaling (0.5 = sqrt weighting, 1.0 = linear inverse frequency)
    """
    inverse_freqs = 1.0 / letter_frequencies
    letter_indices = [LETTER_TO_INDEX[c] for c in set(word)]
    base_score = np.mean(inverse_freqs[letter_indices])
    return float(base_score ** exponent)
