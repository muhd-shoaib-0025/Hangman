# train_plots.py
# Matplotlib-only plotting utilities for Hangman report.
# Now saves both PNG (1200 dpi) and PDF for each plot.

import math
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import os, json, argparse

# ----------------------------
# Global style for high-quality figures
# ----------------------------
mpl.rcParams.update({
    "font.size": 12,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.dpi": 1200,     # interactive display
    "lines.linewidth": 2.0,
    "axes.grid": True,
    "grid.alpha": 0.3
})

_ALPHABET = "abcdefghijklmnopqrstuvwxyz"

# ----------------------------
# Save helper
# ----------------------------
def _save_all(savepath):
    """Save figure to both PNG (1200 dpi) and PDF."""
    if savepath:
        base, _ = os.path.splitext(savepath)
        # Save high-res PNG
        plt.savefig(base + ".png", dpi=1200, bbox_inches="tight")
        # Save vector PDF
        plt.savefig(base + ".pdf", bbox_inches="tight")
    plt.close()

def _to_np(x):
    return np.asarray(x)

# ----------------------------
# Dataset statistics
# ----------------------------
def plot_word_length_histogram(words, bins=None, savepath=None, title="Word length distribution"):
    L = np.array([len(w) for w in words], dtype=int)
    if bins is None:
        bins = np.arange(L.min(), L.max() + 2) - 0.5
    plt.figure()
    plt.hist(L, bins=bins)
    plt.xlabel("Word length")
    plt.ylabel("Count")
    plt.title(title)
    _save_all(savepath)

def plot_bigram_heatmap(words, alphabet=_ALPHABET, savepath=None, title="Bigram frequency (row->col)"):
    A = len(alphabet)
    idx = {c:i for i,c in enumerate(alphabet)}
    M = np.zeros((A, A), dtype=np.int64)
    for w in words:
        for i in range(len(w) - 1):
            a, b = w[i], w[i+1]
            ia, ib = idx.get(a), idx.get(b)
            if ia is not None and ib is not None:
                M[ia, ib] += 1
    rowsum = M.sum(axis=1, keepdims=True)
    rowsum[rowsum == 0] = 1
    N = M / rowsum
    plt.figure()
    plt.imshow(N, origin="lower", aspect="equal")
    plt.xticks(range(A), list(alphabet), rotation=90)
    plt.yticks(range(A), list(alphabet))
    plt.xlabel("Next letter")
    plt.ylabel("Current letter")
    plt.title(title)
    plt.colorbar(label="Row-normalized freq")
    _save_all(savepath)

# ----------------------------
# Training curves
# ----------------------------
def plot_training_curves(history_dict, savepath=None, title="Training vs validation"):
    plt.figure(figsize=(12,4))
    ax1 = plt.subplot(1,2,1)
    if "loss" in history_dict:
        ax1.plot(history_dict["loss"], label="train")
    if "val_loss" in history_dict:
        ax1.plot(history_dict["val_loss"], label="val")
    ax1.set_title("Loss"); ax1.set_xlabel("Epoch"); ax1.set_ylabel("Loss"); ax1.legend()
    ax2 = plt.subplot(1,2,2)
    acc_keys = [k for k in history_dict.keys() if "acc" in k and not k.startswith("val_")]
    val_acc_keys = [k for k in history_dict.keys() if k.startswith("val_") and "acc" in k]
    if acc_keys: ax2.plot(history_dict[acc_keys[0]], label="train")
    if val_acc_keys: ax2.plot(history_dict[val_acc_keys[0]], label="val")
    ax2.set_title("Accuracy"); ax2.set_xlabel("Epoch"); ax2.set_ylabel("Accuracy"); ax2.legend()
    plt.suptitle(title)
    _save_all(savepath)

def plot_lr_schedule(lr_values, savepath=None, title="Learning rate schedule"):
    lr_values = _to_np(lr_values)
    plt.figure()
    plt.plot(lr_values)
    plt.xlabel("Step"); plt.ylabel("LR"); plt.title(title)
    _save_all(savepath)

# ----------------------------
# Confusion, reliability, top-k
# ----------------------------
def plot_confusion_matrix(y_true, y_pred, labels=None, normalize=True, savepath=None, title="Confusion matrix"):
    y_true = _to_np(y_true).astype(int); y_pred = _to_np(y_pred).astype(int)
    C = np.zeros((26,26), dtype=np.int64)
    for a,b in zip(y_true, y_pred):
        C[a,b] += 1
    if normalize:
        rowsum = C.sum(axis=1, keepdims=True)
        rowsum[rowsum==0] = 1
        C = C / rowsum
    plt.figure(figsize=(6,5))
    plt.imshow(C, origin="lower", aspect="equal")
    if labels is None: labels = list(_ALPHABET)
    plt.xticks(range(26), labels, rotation=90)
    plt.yticks(range(26), labels)
    plt.xlabel("Predicted"); plt.ylabel("True")
    plt.title(title)
    plt.colorbar(label="Proportion" if normalize else "Count")
    _save_all(savepath)

def plot_topk_accuracy_curve(y_true, y_proba, ks=(1,2,3,5,10), savepath=None, title="Top-k accuracy"):
    y_true = _to_np(y_true).astype(int)
    y_proba = _to_np(y_proba)
    order = np.argsort(-y_proba, axis=1)
    accs = []
    for k in ks:
        topk = order[:, :k]
        hit = (topk == y_true[:, None]).any(axis=1)
        accs.append(hit.mean())
    plt.figure()
    plt.plot(ks, accs, marker="o")
    plt.xticks(ks); plt.ylim(0,1)
    plt.xlabel("k"); plt.ylabel("Accuracy"); plt.title(title)
    _save_all(savepath)

def plot_reliability_diagram(y_true, y_proba, class_index=None, n_bins=10, savepath=None, title="Reliability diagram"):
    y_true = _to_np(y_true).astype(int)
    y_proba = _to_np(y_proba)
    if class_index is None:
        pred_class = np.argmax(y_proba, axis=1)
        conf = y_proba[np.arange(len(y_true)), pred_class]
        correct = (pred_class == y_true).astype(int)
    else:
        conf = y_proba[:, class_index]
        correct = (y_true == class_index).astype(int)
    bins = np.linspace(0, 1, n_bins+1)
    idx = np.digitize(conf, bins) - 1
    prob_bin = np.zeros(n_bins); acc_bin = np.zeros(n_bins); cnt = np.zeros(n_bins)
    for i in range(n_bins):
        m = idx == i; cnt[i] = m.sum()
        if cnt[i] > 0:
            prob_bin[i] = conf[m].mean(); acc_bin[i] = correct[m].mean()
    plt.figure()
    plt.plot([0,1],[0,1], linestyle="--")
    plt.scatter(prob_bin[cnt>0], acc_bin[cnt>0])
    plt.xlabel("Confidence"); plt.ylabel("Accuracy"); plt.title(title)
    _save_all(savepath)

# ----------------------------
# Main
# ----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifacts_dir", type=str, default="train_artifacts")
    ap.add_argument("--dictionary", type=str, default="words_250000_train.txt")
    ap.add_argument("--plots_dir", type=str, default=None)
    args = ap.parse_args()

    plots_dir = args.plots_dir or os.path.join(args.artifacts_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    # Training curves
    with open(os.path.join(args.artifacts_dir, "history.json")) as f:
        hist = json.load(f)
    plot_training_curves(hist, savepath=os.path.join(plots_dir, "training_curves"))

    # LR schedule
    lr = np.load(os.path.join(args.artifacts_dir, "lr_values.npy"))
    plot_lr_schedule(lr, savepath=os.path.join(plots_dir, "lr_schedule"))

    # Validation performance
    vp = np.load(os.path.join(args.artifacts_dir, "val_preds.npz"))
    y_true, y_pred, y_proba = vp["y_true"], vp["y_pred"], vp["y_proba"]
    plot_confusion_matrix(y_true, y_pred, savepath=os.path.join(plots_dir, "confusion_matrix"))
    plot_topk_accuracy_curve(y_true, y_proba, savepath=os.path.join(plots_dir, "topk_accuracy"))
    plot_reliability_diagram(y_true, y_proba, savepath=os.path.join(plots_dir, "reliability"))

    # Dataset stats (optional)
    if os.path.isfile(args.dictionary):
        with open(args.dictionary) as f:
            words = [w.strip().lower() for w in f if w.strip()]
        plot_word_length_histogram(words, savepath=os.path.join(plots_dir, "word_length_hist"))
        plot_bigram_heatmap(words, savepath=os.path.join(plots_dir, "bigram_heatmap"))

if __name__ == "__main__":
    main()
