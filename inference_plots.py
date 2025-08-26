# inference_plots.py
# Build plots from artifacts/infer_guess_log.jsonl written by guess().
import os, json, argparse
import numpy as np
import matplotlib.pyplot as plt

def _save(figpath):
    os.makedirs(os.path.dirname(figpath), exist_ok=True)
    plt.savefig(figpath, bbox_inches="tight"); plt.close()

def _load_jsonl(path):
    recs = []
    with open(path) as f:
        for ln in f:
            ln = ln.strip()
            if ln: recs.append(json.loads(ln))
    return recs

def plot_latency_hist(recs, outdir):
    lat = [r["lat_ms"] for r in recs if "lat_ms" in r]
    if not lat: return
    plt.figure(); plt.hist(lat, bins=30)
    plt.xlabel("Latency per guess (ms)"); plt.ylabel("Count"); plt.title("Inference latency")
    _save(os.path.join(outdir, "latency_hist.png"))

def plot_conf_hist(recs, outdir):
    conf = [r["conf"] for r in recs if "conf" in r]
    if not conf: return
    plt.figure(); plt.hist(conf, bins=20, range=(0,1))
    plt.xlabel("Top-1 probability"); plt.ylabel("Count"); plt.title("Confidence histogram")
    _save(os.path.join(outdir, "confidence_hist.png"))

def plot_conf_vs_turn(recs, outdir):
    turns = np.array([r["turn"] for r in recs if "conf" in r], int)
    confs = np.array([r["conf"] for r in recs if "conf" in r], float)
    if turns.size == 0: return
    xs, ys = [], []
    for t in range(int(turns.max())+1):
        m = turns == t
        if m.any():
            xs.append(t); ys.append(float(confs[m].mean()))
    plt.figure(); plt.plot(xs, ys, marker="o")
    plt.xlabel("Turn"); plt.ylabel("Mean top-1 prob"); plt.ylim(0,1); plt.title("Confidence vs turn")
    _save(os.path.join(outdir, "confidence_vs_turn.png"))

def plot_entropy_vs_turn(recs, outdir):
    turns = np.array([r["turn"] for r in recs if "entropy" in r], int)
    ents  = np.array([r["entropy"] for r in recs if "entropy" in r], float)
    if turns.size == 0: return
    xs, ys = [], []
    for t in range(int(turns.max())+1):
        m = turns == t
        if m.any():
            xs.append(t); ys.append(float(ents[m].mean()))
    plt.figure(); plt.plot(xs, ys, marker="o")
    plt.xlabel("Turn"); plt.ylabel("Entropy"); plt.title("Entropy vs turn")
    _save(os.path.join(outdir, "entropy_vs_turn.png"))

def plot_hit_rate_by_turn(recs, outdir):
    # last guess in a game may lack 'hit'; we ignore missing labels
    T, H = [], []
    for r in recs:
        if "hit" in r:
            T.append(int(r["turn"])); H.append(int(r["hit"]))
    if not T: return
    T = np.array(T); H = np.array(H)
    xs, ys = [], []
    for t in range(int(T.max())+1):
        m = T == t
        if m.any():
            xs.append(t); ys.append(float(H[m].mean()))
    plt.figure(); plt.plot(xs, ys, marker="o"); plt.ylim(0,1)
    plt.xlabel("Turn"); plt.ylabel("Hit rate"); plt.title("Per-turn hit rate")
    _save(os.path.join(outdir, "hit_rate_by_turn.png"))

def plot_letter_popularity(recs, outdir):
    from collections import Counter
    cnt = Counter(r["letter"] for r in recs if "letter" in r)
    if not cnt: return
    letters, vals = zip(*sorted(cnt.items()))
    plt.figure(); plt.bar(np.arange(len(letters)), vals)
    plt.xticks(range(len(letters)), letters)
    plt.xlabel("Guessed letter"); plt.ylabel("Count"); plt.title("Letter usage")
    _save(os.path.join(outdir, "letter_usage.png"))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifacts_dir", default="inference_artifacts")
    args = ap.parse_args()

    log_path = os.path.join(args.artifacts_dir, "infer_guess_log.jsonl")
    if not os.path.isfile(log_path):
        print(f"no log: {log_path}"); return
    outdir = os.path.join(args.artifacts_dir, "plots_infer"); os.makedirs(outdir, exist_ok=True)

    recs = _load_jsonl(log_path)
    if not recs:
        print("empty log"); return

    plot_latency_hist(recs, outdir)
    plot_conf_hist(recs, outdir)
    plot_conf_vs_turn(recs, outdir)
    plot_entropy_vs_turn(recs, outdir)
    plot_hit_rate_by_turn(recs, outdir)
    plot_letter_popularity(recs, outdir)

    print(f"wrote plots to {outdir}")

if __name__ == "__main__":
    main()
