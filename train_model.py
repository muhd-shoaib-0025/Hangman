# train_model.py
# CPU-only Hangman next-letter prediction model.
# Loads a cached dataset from --data_dir, trains the model, and saves artifacts.

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"   # Disable GPU usage
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import json, time, argparse, logging, random, math, tempfile
from pathlib import Path
import numpy as np
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Dense, Dropout, GlobalAveragePooling1D, Concatenate, Flatten, Reshape
from tensorflow.keras.optimizers.schedules import CosineDecay

# AdamW optimizer import with fallbacks
try:
    from tensorflow.keras.optimizers import AdamW
except Exception:
    try:
        from tensorflow.keras.optimizers.legacy import AdamW
    except Exception:
        try:
            from tensorflow_addons.optimizers import AdamW
        except Exception:
            AdamW = None

from hangman_commons import TOKEN_VOCAB_SIZE
from hangman_layers import TransformerBlock, GLUBlock, PositionalEmbedding

# ---------------- Logging ----------------
LOG_LEVEL = os.getenv("HANGMAN_LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s"
)
logger = logging.getLogger("hangman.train")

# ---------------- File Paths ----------------
MODEL_SAVE_PATH = "hangman_model.keras"
MODEL_META_PATH = MODEL_SAVE_PATH + ".meta.json"

# ---------------- Default Hyperparameters ----------------
EMBEDDING_DIM = 128
ATTENTION_HEADS = 1
FEEDFORWARD_DIM = 256
NUM_TRANSFORMER_BLOCKS = 2
NUM_GLU_BLOCKS = 1
DEFAULT_BATCH_SIZE = 1024
DEFAULT_NUM_EPOCHS = 50
DEFAULT_STEPS_PER_EXEC = 64

# ---------------- Reproducibility ----------------
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
tf.keras.utils.set_random_seed(RANDOM_SEED)
try:
    tf.config.experimental.enable_op_determinism(True)
except Exception:
    pass


# ---------------- Atomic Writers ----------------
def write_json_atomic(path: Path, obj: dict):
    """Write JSON to file atomically (safe against crashes)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=str(path.parent), suffix=".json", delete=False, mode="w") as tmp:
        json.dump(obj, tmp)
        tmp.flush(); os.fsync(tmp.fileno())
        tmp_name = tmp.name
    os.replace(tmp_name, path)


def write_npy_atomic(path: Path, array: np.ndarray):
    """Write NumPy .npy array atomically."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=str(path.parent), suffix=".npy", delete=False) as tmp:
        np.save(tmp, array)
        tmp.flush(); os.fsync(tmp.fileno())
        tmp_name = tmp.name
    os.replace(tmp_name, path)


def write_npz_atomic(path: Path, **arrays):
    """Write compressed NumPy .npz archive atomically."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=str(path.parent), suffix=".npz", delete=False) as tmp:
        np.savez_compressed(tmp, **arrays)
        tmp.flush(); os.fsync(tmp.fileno())
        tmp_name = tmp.name
    os.replace(tmp_name, path)


# ---------------- Dataset Handling ----------------
def dataset_file_paths(data_dir: str) -> dict[str, str]:
    """Return expected dataset file paths."""
    return {
        "npz": os.path.join(data_dir, "train_dataset.npz"),
        "meta": os.path.join(data_dir, "dataset_meta.json")
    }


def load_cached_dataset(data_dir: str):
    """Load cached dataset (features + labels + metadata)."""
    paths = dataset_file_paths(data_dir)
    if not (os.path.isfile(paths["npz"]) and os.path.isfile(paths["meta"])):
        return None

    npz = np.load(paths["npz"])
    features_sequence = npz["X_sequence"]
    features_guessed = npz["X_guessed_letters"]
    features_meta = npz["X_meta_features"]
    labels_distribution = npz["Y_next_letter_distribution"]

    with open(paths["meta"], "r") as f:
        metadata = json.load(f)

    max_sequence_length = int(metadata["max_len_used"])
    logger.info(
        "Loaded dataset from %s (samples=%d, max_len=%d)",
        data_dir, len(labels_distribution), max_sequence_length
    )
    return (features_sequence, features_guessed, features_meta), labels_distribution, max_sequence_length


# ---------------- Model Architecture ----------------
def build_hangman_model(max_seq_len: int) -> tf.keras.Model:
    """Build transformer-based Hangman next-letter prediction model."""
    input_sequence = Input(shape=(max_seq_len, TOKEN_VOCAB_SIZE), name="masked_word_sequence")
    input_guessed = Input(shape=(26,), name="guessed_letters")
    input_meta = Input(shape=(2,), name="meta_features")

    # Project one-hot tokens into embedding space
    token_projection = Dense(EMBEDDING_DIM, activation=None, name="linear_projection")(input_sequence)
    token_with_position = PositionalEmbedding(max_len=max_seq_len, embed_dim=EMBEDDING_DIM, name="positional_encoding")(token_projection)

    x = token_with_position
    for i in range(NUM_TRANSFORMER_BLOCKS):
        x = TransformerBlock(
            embed_dim=EMBEDDING_DIM,
            num_heads=ATTENTION_HEADS,
            ff_dim=FEEDFORWARD_DIM,
            dropout=0.1,
            name=f"transformer_block_{i+1}"
        )(x)

    pooled_features = GlobalAveragePooling1D(name="average_pooling")(x)

    concatenated_features = Concatenate(name="concatenate_features")(
        [pooled_features, input_guessed, input_meta]
    )

    expanded_features = Reshape((1, EMBEDDING_DIM + 26 + 2), name="expand_time_dimension")(concatenated_features)

    h = Flatten(name="flatten_features")(expanded_features)
    h = Dense(256, activation="relu", name="dense_hidden")(h)
    h = Dropout(0.1, name="dropout_hidden")(h)
    output_logits = Dense(26, activation="softmax", name="predicted_letter_distribution")(h)

    return Model(inputs=[input_sequence, input_guessed, input_meta], outputs=output_logits)


# ---------------- TF Dataset Creation ----------------
def create_tf_datasets(train_inputs, y_train, val_inputs, y_val, batch_size: int):
    """Build tf.data.Dataset for training and validation."""
    AUTOTUNE = tf.data.AUTOTUNE
    n_train = len(y_train)

    train_dataset = tf.data.Dataset.from_tensor_slices(((train_inputs[0], train_inputs[1], train_inputs[2]), y_train))
    train_dataset = train_dataset.shuffle(min(n_train, 1_000_000), seed=RANDOM_SEED, reshuffle_each_iteration=True)
    train_dataset = train_dataset.batch(batch_size, drop_remainder=False).cache().prefetch(AUTOTUNE)

    val_dataset = tf.data.Dataset.from_tensor_slices(((val_inputs[0], val_inputs[1], val_inputs[2]), y_val))
    val_dataset = val_dataset.batch(batch_size, drop_remainder=False).prefetch(AUTOTUNE)

    return train_dataset, val_dataset


# ---------------- Training ----------------
def train_and_save_model(
    data_dir: str,
    batch_size: int,
    epochs: int,
    lr: float = 1e-3,
    weight_decay: float = 1e-3,
    cosine_alpha: float = 0.1,
    model_path: str = MODEL_SAVE_PATH,
    meta_path: str = MODEL_META_PATH,
    steps_per_execution: int = DEFAULT_STEPS_PER_EXEC,
    no_jit: bool = False,
    artifacts_dir: str = "train_artifacts",
):
    """Main training loop: load dataset, train model, save results."""
    start_time = time.time()

    cached = load_cached_dataset(data_dir)
    if cached is None:
        raise FileNotFoundError(
            f"Dataset not found in '{data_dir}'. Build it first:\n"
            f"python create_train_data.py --data_dir {data_dir} --dictionary words_250000_train.txt "
            f"--max_len 32 --max_words 100000 --samples_per_word 2 --max_steps_per_word 8 "
            f"--max_samples_cap 1000000 --workers 4"
        )

    (features_seq, features_guessed, features_meta), labels_distribution, max_len_used = cached
    labels_sparse = np.argmax(labels_distribution, axis=1).astype(np.int32)

    # Shuffle
    permutation = np.random.permutation(len(labels_sparse))
    features_seq, features_guessed, features_meta, labels_sparse = (
        features_seq[permutation],
        features_guessed[permutation],
        features_meta[permutation],
        labels_sparse[permutation]
    )

    # Train/val split
    n_train = int(0.8 * len(labels_sparse))
    train_inputs = [features_seq[:n_train], features_guessed[:n_train], features_meta[:n_train]]
    val_inputs   = [features_seq[n_train:], features_guessed[n_train:], features_meta[n_train:]]
    y_train, y_val = labels_sparse[:n_train], labels_sparse[n_train:]
    logger.info("Dataset sizes: train=%d val=%d", len(y_train), len(y_val))

    train_ds, val_ds = create_tf_datasets(train_inputs, y_train, val_inputs, y_val, batch_size)

    steps_per_epoch = max(1, math.ceil(n_train / batch_size))
    total_steps = steps_per_epoch * max(epochs, 1)
    lr_schedule = CosineDecay(initial_learning_rate=lr, decay_steps=total_steps, alpha=cosine_alpha)
    lr_values = np.array([float(lr_schedule(s).numpy()) for s in range(total_steps)], dtype=np.float32)

    optimizer = (
        tf.keras.optimizers.Adam(learning_rate=lr_schedule)
        if AdamW is None else AdamW(learning_rate=lr_schedule, weight_decay=weight_decay)
    )

    model = build_hangman_model(max_len_used)
    compile_config = dict(
        optimizer=optimizer,
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy")],
        run_eagerly=False,
        steps_per_execution=int(steps_per_execution),
    )

    if no_jit:
        model.compile(**compile_config)
        logger.info("XLA JIT: disabled")
    else:
        try:
            model.compile(**compile_config, jit_compile=True)
            logger.info("XLA JIT: enabled")
        except TypeError:
            model.compile(**compile_config)
            logger.warning("XLA JIT unsupported; continuing without it.")

    artifacts_path = Path(artifacts_dir)
    artifacts_path.mkdir(parents=True, exist_ok=True)

    # Train
    t1 = time.time()
    history_obj = model.fit(train_ds, validation_data=val_ds, epochs=epochs, verbose=1)
    t2 = time.time()

    # Save model + metadata
    model.save(model_path)
    write_json_atomic(Path(meta_path), {"max_len": int(max_len_used)})

    # Save training artifacts
    history_dict = {k: [float(x) for x in v] for k, v in history_obj.history.items()}
    write_json_atomic(artifacts_path / "history.json", history_dict)
    write_npy_atomic(artifacts_path / "lr_values.npy", lr_values)

    # Save validation predictions
    try:
        y_val_proba = model.predict(val_ds, verbose=0)
        y_val_pred = y_val_proba.argmax(axis=1).astype(np.int32)
        y_true = y_val.astype(np.int32)
        if y_val_proba.ndim != 2 or y_val_proba.shape[1] != 26:
            raise ValueError("Unexpected y_val_proba shape")
    except Exception as e:
        logger.warning("Validation prediction failed (%s). Writing empty val_preds.npz.", str(e))
        y_true = np.zeros((0,), dtype=np.int32)
        y_val_pred = np.zeros((0,), dtype=np.int32)
        y_val_proba = np.zeros((0, 26), dtype=np.float32)

    write_npz_atomic(artifacts_path / "val_preds.npz",
                     y_true=y_true, y_pred=y_val_pred, y_proba=y_val_proba)

    # Write sentinel DONE file
    write_json_atomic(artifacts_path / "DONE", {
        "finished_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "epochs": int(epochs),
        "steps_per_epoch": int(steps_per_epoch),
    })

    logger.info("Saved model=%s meta=%s", model_path, meta_path)
    logger.info("Artifacts saved to %s (history.json, lr_values.npy, val_preds.npz, DONE)", str(artifacts_path))
    logger.info("Steps/epoch=%d total_steps=%d lr=%.2e wd=%.1e spe=%d jit=%s",
                steps_per_epoch, total_steps, lr, weight_decay, steps_per_execution, str(not no_jit))
    logger.info("Timing: prep=%.1fs train=%.1fs total=%.1fs", t1 - start_time, t2 - t1, time.time() - start_time)


# ---------------- CLI ----------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Hangman next-letter model (CPU only). Dataset must exist.")
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--batch", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--epochs", type=int, default=DEFAULT_NUM_EPOCHS)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-3)
    parser.add_argument("--cosine_alpha", type=float, default=0.1)
    parser.add_argument("--steps_per_execution", type=int, default=DEFAULT_STEPS_PER_EXEC)
    parser.add_argument("--no_jit", action="store_true")
    parser.add_argument("--artifacts_dir", type=str, default="train_artifacts")
    args = parser.parse_args()

    train_and_save_model(
        data_dir=args.data_dir,
        batch_size=args.batch,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        cosine_alpha=args.cosine_alpha,
        steps_per_execution=args.steps_per_execution,
        no_jit=args.no_jit,
        artifacts_dir=args.artifacts_dir,
    )
