# patch.py
# Fix doubled "turn" in infer_guess_log.jsonl and regenerate plots.

import os, json, argparse, shutil, tempfile, time
import numpy as np
import matplotlib.pyplot as plt

# ---- lightweight re-use of plotting from inference_plots.py (if present)
try:
    from inference_plots import (
        plot_latency_hist, plot_conf_hist, plot_conf_vs_turn,
        plot_entropy_vs_turn, plot_hit_rate_by_turn, plot_letter_popularity
    )
    _HAVE_INF_PLOTS = True
except Exception:
    _HAVE_INF_PLOTS = False

def _load_jsonl(path):
    recs = []
    with open(path, "r") as f:
        for ln in f:
            ln = ln.strip()
            if ln:
                recs.append(json.loads(ln))
    return recs

def _write_jsonl_atomic(path, recs):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with tempfile.NamedTemporaryFile("w", delete=False, dir=os.path.dirname(path), suffix=".jsonl") as tmp:
        for r in recs:
            tmp.write(json.dumps(r) + "\n")
        tmp.flush(); os.fsync(tmp.fileno())
        tmpname = tmp.name
    os.replace(tmpname, path)

def _save(figpath):
    os.makedirs(os.path.dirname(figpath), exist_ok=True)
    plt.savefig(figpath, bbox_inches="tight")
    plt.close()

# Fallback plotting (if inference_plots.py isn't importable)
def _fallback_plots(recs, outdir):
    # confidence histogram
    conf = [r.get("conf", None) for r in recs if "conf" in r]
    if conf:
        plt.figure(); plt.hist(conf, bins=20, range=(0,1))
        plt.xlabel("Top-1 probability"); plt.ylabel("Count"); plt.title("Confidence histogram")
        _save(os.path.join(outdir, "confidence_hist.png"))

    # confidence vs turn
    tt = np.array([r["turn"] for r in recs if "turn" in r], int)
    cc = np.array([r["conf"] for r in recs if "turn" in r and "conf" in r], float)
    if tt.size:
        xs, ys = [], []
        for t in range(int(tt.max())+1):
            m = tt == t
            if m.any():
                xs.append(t); ys.append(float(cc[m].mean()))
        plt.figure(); plt.plot(xs, ys, marker="o")
        plt.xlabel("Turn"); plt.ylabel("Mean top-1 prob"); plt.ylim(0,1); plt.title("Confidence vs turn")
        _save(os.path.join(outdir, "confidence_vs_turn.png"))

    # letter usage
    from collections import Counter
    cnt = Counter(r.get("letter","") for r in recs if "letter" in r)
    if cnt:
        letters, vals = zip(*sorted(cnt.items()))
        plt.figure(); plt.bar(np.arange(len(letters)), vals)
        plt.xticks(range(len(letters)), letters)
        plt.xlabel("Guessed letter"); plt.ylabel("Count"); plt.title("Letter usage")
        _save(os.path.join(outdir, "letter_usage.png"))

def main():
    ap = argparse.ArgumentParser(description="Fix doubled turn indices in inference log and rebuild plots.")
    ap.add_argument("--artifacts_dir", default="inference_artifacts", help="Directory with infer_guess_log.jsonl")
    ap.add_argument("--log_name", default="infer_guess_log.jsonl")
    ap.add_argument("--backup", action="store_true", help="Write a timestamped .bak before patching")
    ap.add_argument("--just_patch", action="store_true", help="Only fix the log; skip plotting")
    args = ap.parse_args()

    log_path = os.path.join(args.artifacts_dir, args.log_name)
    if not os.path.isfile(log_path):
        raise FileNotFoundError(f"Missing log: {log_path}")

    # Load
    recs = _load_jsonl(log_path)
    if not recs:
        print("Log is empty; nothing to do.")
        return

    # Patch "turn" by halving (floor)
    changed = 0
    for r in recs:
        if "turn" in r and isinstance(r["turn"], (int, float)):
            old = int(r["turn"])
            new = int(old // 2)
            if new != old:
                r["turn"] = new
                changed += 1

    if changed == 0:
        print("No turn values required patching.")
    else:
        # Backup
        if args.backup:
            ts = time.strftime("%Y%m%d-%H%M%S")
            bak = f"{log_path}.{ts}.bak"
            shutil.copy2(log_path, bak)
            print(f"Backup written: {bak}")

        # Atomic write
        _write_jsonl_atomic(log_path, recs)
        print(f"Patched {changed} records in {log_path}")

    if args.just_patch:
        return

    # Rebuild plots
    outdir = os.path.join(args.artifacts_dir, "plots_infer")
    os.makedirs(outdir, exist_ok=True)

    # High-DPI for crisp images (optional)
    plt.rcParams["figure.dpi"] = 120
    plt.rcParams["savefig.dpi"] = 300

    if _HAVE_INF_PLOTS:
        # Use your existing plotting helpers
        from inference_plots import (
            plot_latency_hist, plot_conf_hist, plot_conf_vs_turn,
            plot_entropy_vs_turn, plot_hit_rate_by_turn, plot_letter_popularity
        )
        plot_latency_hist(recs, outdir)
        plot_conf_hist(recs, outdir)
        plot_conf_vs_turn(recs, outdir)
        plot_entropy_vs_turn(recs, outdir)
        plot_hit_rate_by_turn(recs, outdir)
        plot_letter_popularity(recs, outdir)
    else:
        # Minimal fallback (confidence/turn/usage)
        _fallback_plots(recs, outdir)

    print(f"Plots regenerated in {outdir}")

if __name__ == "__main__":
    main()
