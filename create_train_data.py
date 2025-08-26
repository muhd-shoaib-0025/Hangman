# create_train_data.py
# Builds and writes the Hangman training dataset (serial or parallel with sharding).
# Unified tqdm progress bars (Words + Samples). Safe to run standalone or import.

import os, json, hashlib, logging, random, argparse, time, re, queue
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import Manager
import numpy as np
from tqdm import tqdm

from hangman_commons import (
    ALPHABET, LETTER_TO_INDEX, TOKEN_VOCAB_SIZE, MAX_INCORRECT_TRIES,
    load_dictionary_words, group_words_by_length, encode_state_as_model_inputs,
    compute_letter_frequencies, compute_word_rarity
)

# ---------------- Logging ----------------
LOG_LEVEL = os.getenv("HANGMAN_LOG_LEVEL", "INFO").upper()
logging.basicConfig(level=getattr(logging, LOG_LEVEL, logging.INFO),
                    format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("hangman.dataset")

# ---------------- Defaults ----------------
DATASET_VERSION = 4
DEFAULT_MAX_LEN = 32
DEFAULT_MAX_WORDS = 100_000
DEFAULT_SAMPLES_PER_WORD = 2
DEFAULT_MAX_STEPS_PER_WORD = 8
DEFAULT_MAX_SAMPLES = 1_000_000
DEFAULT_WORKERS = max(1, (os.cpu_count() or 1) // 2)
RANDOM_SEED = 42

# Label processing knobs
DEFAULT_SMOOTH_EPS = 0.05
DEFAULT_TOPM_LETTERS = 12
DEFAULT_RARITY_POWER = 0.5


# ---------------- Helpers ----------------
def sha1_of_file(filepath, chunk_bytes=1 << 20) -> str:
    h = hashlib.sha1()
    with open(filepath, "rb") as f:
        while True:
            block = f.read(chunk_bytes)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def get_dataset_paths(data_dir: str):
    os.makedirs(data_dir, exist_ok=True)
    return {
        "npz": os.path.join(data_dir, "train_dataset.npz"),
        "meta": os.path.join(data_dir, "dataset_meta.json"),
        "clean_dict": os.path.join(data_dir, "dictionary_clean.txt"),
        "shard_prefix": os.path.join(data_dir, "train_dataset.shard"),
    }


def make_dataset_metadata(config: dict) -> dict:
    meta = dict(config)
    meta["version"] = DATASET_VERSION
    meta["random_seed"] = RANDOM_SEED
    return meta


def apply_guesses_to_mask(word: str, guessed_letters: set[str]) -> str:
    return "".join([c if c in guessed_letters else "_" for c in word])


# ---------------- Target Distribution ----------------
def compute_next_letter_distribution(mask: str, guessed_letters: set[str],
                                     words_by_length: dict[int, list[str]],
                                     smooth_eps: float, topm_letters: int) -> np.ndarray:
    regex = re.compile("^" + mask.replace("_", ".") + "$")
    present_letters = {c for c in mask if c != "_"}
    absent_letters = {g for g in guessed_letters if g not in present_letters}
    candidates = [
        w for w in words_by_length.get(len(mask), [])
        if regex.match(w) and not any(b in w for b in absent_letters)
    ]

    probs = np.zeros((26,), dtype=np.float32)
    if not candidates:
        probs[:] = 1.0
    else:
        counts = Counter("".join(candidates))
        for ch, cnt in counts.items():
            probs[LETTER_TO_INDEX[ch]] = float(cnt)

        if topm_letters and topm_letters > 0:
            mask_unguessed = np.ones(26, dtype=bool)
            for ch in guessed_letters:
                if ch in LETTER_TO_INDEX:
                    mask_unguessed[LETTER_TO_INDEX[ch]] = False
            scores = probs.copy()
            scores[~mask_unguessed] = -1.0
            top_indices = np.argsort(scores)[-topm_letters:]
            keep_mask = np.zeros(26, dtype=bool)
            keep_mask[top_indices] = True
            probs[~keep_mask] = 0.0

        for ch in guessed_letters:
            if ch in LETTER_TO_INDEX:
                probs[LETTER_TO_INDEX[ch]] = 0.0

    if smooth_eps > 0:
        for i in range(26):
            if ALPHABET[i] not in guessed_letters:
                probs[i] += float(smooth_eps)

    total = probs.sum()
    if total == 0.0:
        for i in range(26):
            probs[i] = 0.0 if ALPHABET[i] in guessed_letters else 1.0
        total = probs.sum()
    probs /= total
    return probs


# ---------------- Sample Generation ----------------
def generate_samples_for_word(word: str, samples_per_word: int, max_steps_per_word: int,
                              max_len_used: int, words_by_length: dict[int, list[str]],
                              rng: np.random.Generator, smooth_eps: float, topm_letters: int):
    sequences, guessed_vectors, meta_features, target_distributions = [], [], [], []
    unique_letters = list(set(word))

    for _ in range(samples_per_word):
        guessed_letters, missed_count = set(), 0
        warm_start_count = rng.integers(0, 3)
        if unique_letters:
            warm_choices = rng.choice(unique_letters,
                                      size=min(warm_start_count, len(unique_letters)),
                                      replace=False)
            guessed_letters.update(warm_choices)

        steps = 0
        while steps < max_steps_per_word:
            steps += 1
            mask = apply_guesses_to_mask(word, guessed_letters)
            if "_" not in mask:
                break
            tries_left = MAX_INCORRECT_TRIES - missed_count
            target_probs = compute_next_letter_distribution(
                mask, guessed_letters, words_by_length, smooth_eps, topm_letters
            )
            seq_onehot, guess_vec, meta_vec = encode_state_as_model_inputs(
                mask, guessed_letters, tries_left, max_len_used
            )
            sequences.append(seq_onehot)
            guessed_vectors.append(guess_vec)
            meta_features.append(meta_vec)
            target_distributions.append(target_probs)

            next_idx = int(np.argmax(target_probs))
            next_letter = ALPHABET[next_idx]
            guessed_letters.add(next_letter)

            if next_letter not in word:
                missed_count += 1
                if missed_count >= MAX_INCORRECT_TRIES:
                    break

    return sequences, guessed_vectors, meta_features, target_distributions


# ---------------- Shard Worker ----------------
def write_shard_from_words(shard_id: int, shard_words: list[str], samples_per_word: int,
                           max_steps_per_word: int, max_len_used: int,
                           words_by_length: dict[int, list[str]], per_worker_cap: int,
                           shard_prefix: str, base_seed: int,
                           smooth_eps: float, topm_letters: int,
                           progress_q=None):
    rng = np.random.default_rng(base_seed + shard_id * 1_000_003)
    random.seed(base_seed + shard_id * 1_000_003)

    sequences, guesses, metas, targets = [], [], [], []
    total_generated = 0

    for word in shard_words:
        seqs, gvecs, metas_list, targ_dists = generate_samples_for_word(
            word, samples_per_word, max_steps_per_word, max_len_used,
            words_by_length, rng, smooth_eps, topm_letters
        )
        n = len(targ_dists)
        if n:
            sequences.extend(seqs)
            guesses.extend(gvecs)
            metas.extend(metas_list)
            targets.extend(targ_dists)
            total_generated += n

        if progress_q:
            try:
                progress_q.put(("word", 1))
                if n:
                    progress_q.put(("sample", n))
            except Exception:
                pass

        if total_generated >= per_worker_cap:
            break

    shard_path = f"{shard_prefix}{shard_id:03d}.npz"
    if total_generated == 0:
        np.savez_compressed(
            shard_path,
            X_sequence=np.zeros((0, max_len_used, TOKEN_VOCAB_SIZE), dtype=np.float32),
            X_guessed_letters=np.zeros((0, 26), dtype=np.float32),
            X_meta_features=np.zeros((0, 2), dtype=np.float32),
            Y_next_letter_distribution=np.zeros((0, 26), dtype=np.float32),
        )
        return shard_path, 0

    np.savez_compressed(
        shard_path,
        X_sequence=np.stack(sequences),
        X_guessed_letters=np.stack(guesses),
        X_meta_features=np.stack(metas),
        Y_next_letter_distribution=np.stack(targets),
    )
    return shard_path, int(len(targets))


# ---------------- Merge Shards ----------------
def merge_shards(shard_files: list[str], out_path_npz: str, max_cap: int) -> int:
    X_seq_list, X_guess_list, X_meta_list, Y_list = [], [], [], []
    total = 0

    for sf in sorted(shard_files):
        with np.load(sf) as z:
            xs, xg, xm, y = z["X_sequence"], z["X_guessed_letters"], z["X_meta_features"], z["Y_next_letter_distribution"]
        if xs.shape[0] == 0:
            continue
        remaining = max(0, max_cap - total)
        if remaining == 0:
            break
        take = min(remaining, xs.shape[0])
        X_seq_list.append(xs[:take])
        X_guess_list.append(xg[:take])
        X_meta_list.append(xm[:take])
        Y_list.append(y[:take])
        total += take

    if total == 0:
        X_sequence = np.zeros((0, 1, TOKEN_VOCAB_SIZE), dtype=np.float32)
        X_guessed_letters = np.zeros((0, 26), dtype=np.float32)
        X_meta_features = np.zeros((0, 2), dtype=np.float32)
        Y_next_letter_distribution = np.zeros((0, 26), dtype=np.float32)
    else:
        X_sequence = np.concatenate(X_seq_list, axis=0)
        X_guessed_letters = np.concatenate(X_guess_list, axis=0)
        X_meta_features = np.concatenate(X_meta_list, axis=0)
        Y_next_letter_distribution = np.concatenate(Y_list, axis=0)

    np.savez_compressed(
        out_path_npz,
        X_sequence=X_sequence,
        X_guessed_letters=X_guessed_letters,
        X_meta_features=X_meta_features,
        Y_next_letter_distribution=Y_next_letter_distribution,
    )
    return int(len(Y_next_letter_distribution))


# ---------------- Main Dataset Builder ----------------
def build_and_save_training_dataset(
    data_dir: str,
    dictionary_path: str,
    requested_max_len=DEFAULT_MAX_LEN,
    max_words=DEFAULT_MAX_WORDS,
    samples_per_word=DEFAULT_SAMPLES_PER_WORD,
    max_steps_per_word=DEFAULT_MAX_STEPS_PER_WORD,
    max_samples_cap=DEFAULT_MAX_SAMPLES,
    workers: int = 1,
    smooth_eps: float = DEFAULT_SMOOTH_EPS,
    topm_letters: int = DEFAULT_TOPM_LETTERS,
    rarity_power: float = DEFAULT_RARITY_POWER,
):
    t_start = time.time()
    paths = get_dataset_paths(data_dir)
    dict_abs_path = os.path.abspath(dictionary_path)
    dict_sha1 = sha1_of_file(dict_abs_path)

    words = load_dictionary_words(dict_abs_path)
    log.info("Dictionary entries=%d", len(words))
    max_len_used = min(requested_max_len, max(len(w) for w in words))
    words_by_length = group_words_by_length(words)

    rng = np.random.default_rng(RANDOM_SEED)
    freqs = compute_letter_frequencies(words)
    weights = np.array([compute_word_rarity(w, freqs) for w in words], dtype=np.float64)
    keys = rng.random(len(words)) ** (1.0 / (weights + 1e-12))
    ordered_words = [words[i] for i in np.argsort(keys)[::-1] if len(words[i]) <= max_len_used]
    if max_words is not None:
        ordered_words = ordered_words[:max_words]

    # --- Serial mode ---
    if workers <= 1:
        log.info("Running single-process dataset generation.")
        sequences, guesses, metas, targets = [], [], [], []
        total_samples = 0
        rng_local = np.random.default_rng(RANDOM_SEED)

        with tqdm(total=len(ordered_words), desc="Words", unit="word", dynamic_ncols=True) as word_bar, \
             tqdm(total=int(max_samples_cap), desc="Samples", unit="samp", dynamic_ncols=True) as samp_bar:
            for word in ordered_words:
                seqs, gvecs, mvecs, targs = generate_samples_for_word(
                    word, samples_per_word, max_steps_per_word,
                    max_len_used, words_by_length, rng_local,
                    smooth_eps, topm_letters
                )
                n = len(targs)
                if n:
                    sequences.extend(seqs); guesses.extend(gvecs)
                    metas.extend(mvecs); targets.extend(targs)
                    total_samples += n
                    samp_bar.update(min(n, max_samples_cap - total_samples))
                word_bar.update(1)
                if total_samples >= max_samples_cap:
                    break

        X_sequence = np.stack(sequences) if total_samples > 0 else np.zeros((0, max_len_used, TOKEN_VOCAB_SIZE))
        X_guessed = np.stack(guesses) if total_samples > 0 else np.zeros((0, 26))
        X_meta = np.stack(metas) if total_samples > 0 else np.zeros((0, 2))
        Y_targets = np.stack(targets) if total_samples > 0 else np.zeros((0, 26))

        np.savez_compressed(paths["npz"],
                            X_sequence=X_sequence,
                            X_guessed_letters=X_guessed,
                            X_meta_features=X_meta,
                            Y_next_letter_distribution=Y_targets)

        with open(paths["clean_dict"], "w") as f:
            f.write("\n".join(words))

        meta = make_dataset_metadata({
            "max_len_used": int(max_len_used),
            "max_words": None if max_words is None else int(max_words),
            "samples_per_word": int(samples_per_word),
            "max_steps_per_word": int(max_steps_per_word),
            "max_samples_cap": int(max_samples_cap),
            "dict_path": dict_abs_path,
            "dict_sha1": dict_sha1,
            "workers": 1,
            "shards": [],
            "actual_samples": int(len(Y_targets)),
            "smooth_eps": float(smooth_eps),
            "topm_letters": int(topm_letters),
            "rarity_power": float(rarity_power),
        })
        with open(paths["meta"], "w") as f:
            json.dump(meta, f, indent=2)

        log.info("Dataset written: %s (samples=%d, max_len=%d) in %.1fs",
                 paths["npz"], len(Y_targets), max_len_used, time.time() - t_start)
        return (X_sequence, X_guessed, X_meta), Y_targets, max_len_used

    # --- Parallel mode ---
    workers = max(1, int(workers))
    word_splits = [[] for _ in range(workers)]
    for i, w in enumerate(ordered_words):
        word_splits[i % workers].append(w)

    per_worker_cap = int(np.ceil(max_samples_cap / workers))
    log.info("Parallel generation with %d workers; per-worker cap=%d", workers, per_worker_cap)

    manager = Manager()
    progress_q = manager.Queue()

    with tqdm(total=len(ordered_words), desc="Words", unit="word", dynamic_ncols=True) as word_bar, \
         tqdm(total=int(max_samples_cap), desc="Samples", unit="samp", dynamic_ncols=True) as samp_bar:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            futures = [
                ex.submit(
                    write_shard_from_words,
                    sid, word_splits[sid], samples_per_word,
                    max_steps_per_word, max_len_used,
                    words_by_length, per_worker_cap, paths["shard_prefix"],
                    RANDOM_SEED, smooth_eps, topm_letters,
                    progress_q
                )
                for sid in range(workers)
            ]

            done_workers = 0
            while done_workers < len(futures):
                try:
                    msg, val = progress_q.get(timeout=0.1)
                    if msg == "word":
                        word_bar.update(val)
                    elif msg == "sample":
                        samp_bar.update(val)
                except queue.Empty:
                    pass
                done_workers = sum(f.done() for f in futures)

            shard_files = []
            shard_counts = []
            for f in futures:
                shard_path, n = f.result()
                shard_files.append(shard_path)
                shard_counts.append(n)

    log.info("Shard sample counts: %s; total=%d", shard_counts, sum(shard_counts))
    final_samples = merge_shards(shard_files, paths["npz"], max_samples_cap)

    with open(paths["clean_dict"], "w") as f:
        f.write("\n".join(words))

    meta = make_dataset_metadata({
        "max_len_used": int(max_len_used),
        "max_words": None if max_words is None else int(max_words),
        "samples_per_word": int(samples_per_word),
        "max_steps_per_word": int(max_steps_per_word),
        "max_samples_cap": int(max_samples_cap),
        "dict_path": dict_abs_path,
        "dict_sha1": dict_sha1,
        "workers": int(workers),
        "shards": [os.path.basename(s) for s in sorted(shard_files)],
        "actual_samples": int(final_samples),
        "smooth_eps": float(smooth_eps),
        "topm_letters": int(topm_letters),
        "rarity_power": float(rarity_power),
    })
    with open(paths["meta"], "w") as f:
        json.dump(meta, f, indent=2)

    with np.load(paths["npz"]) as z:
        X_sequence, X_guessed, X_meta, Y_targets = (
            z["X_sequence"], z["X_guessed_letters"], z["X_meta_features"], z["Y_next_letter_distribution"]
        )

    log.info("Dataset written: %s (samples=%d, max_len=%d) in %.1fs",
             paths["npz"], len(Y_targets), max_len_used, time.time() - t_start)
    return (X_sequence, X_guessed, X_meta), Y_targets, max_len_used

# ---------------- CLI ----------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build Hangman training dataset.")
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--dictionary", type=str, default="words_250000_train.txt")
    parser.add_argument("--max_len", type=int, default=DEFAULT_MAX_LEN)
    parser.add_argument("--max_words", type=int, default=DEFAULT_MAX_WORDS)
    parser.add_argument("--samples_per_word", type=int, default=DEFAULT_SAMPLES_PER_WORD)
    parser.add_argument("--max_steps_per_word", type=int, default=DEFAULT_MAX_STEPS_PER_WORD)
    parser.add_argument("--max_samples_cap", type=int, default=DEFAULT_MAX_SAMPLES)
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    parser.add_argument("--smooth_eps", type=float, default=DEFAULT_SMOOTH_EPS)
    parser.add_argument("--topm_letters", type=int, default=DEFAULT_TOPM_LETTERS)
    parser.add_argument("--rarity_power", type=float, default=DEFAULT_RARITY_POWER)
    args = parser.parse_args()

    build_and_save_training_dataset(
        data_dir=args.data_dir,
        dictionary_path=args.dictionary,
        requested_max_len=args.max_len,
        max_words=args.max_words,
        samples_per_word=args.samples_per_word,
        max_steps_per_word=args.max_steps_per_word,
        max_samples_cap=args.max_samples_cap,
        workers=args.workers,
        smooth_eps=args.smooth_eps,
        topm_letters=args.topm_letters,
        rarity_power=args.rarity_power,
    )
