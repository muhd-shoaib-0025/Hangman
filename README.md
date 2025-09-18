# Hangman — Neural Letter Predictor

Neural agent for Hangman using a small Transformer. Trains on simulated game states to predict the next letter.

---

## Challenge context

Build a Hangman player, use the provided dictionary, play 1,000 recorded games, and submit code plus a short strategy write-up. 

---

## Features

- Dataset built by simulating Hangman boards from a dictionary  
- Next-letter classifier over 26 letters  
- Fast scripts for data generation, training, and API/demo play

---

## Repo structure

```
Hangman/
├─ create_train_data.py        # Simulate games and build arrays
├─ train_model.py              # Model, training loop, logging
├─ hangman_api_user.py         # Simple API/demo runner
├─ requirements.txt
├─ report/                     # PDF with details and figures
└─ data/                       # Dictionary + generated tensors (gitignored if large)
```

---

## Quickstart

### 1) Setup

```bash
git clone https://github.com/muhd-shoaib-0025/Hangman.git
cd Hangman
python -m venv .venv
# Linux/macOS
source .venv/bin/activate
# Windows
# .venv\Scripts\activate
pip install -r requirements.txt
```

### 2) Build training data

Uses a word list like `words_250000_train.txt`.

```bash
python create_train_data.py   --data_dir data   --dictionary words_250000_train.txt   --max_len 16   --max_words 250000   --samples_per_word 4   --max_steps_per_word 4   --max_samples_cap 1000000   --workers 32
```

Output arrays (saved in `data/`):
- `Xsequence`: encoded masked word sequence per position  
- `Xguessed`: 26-dim binary vector for guessed letters  
- `Xmeta`: word length and tries remaining (normalized)  
- `Y`: soft next-letter distribution or argmax labels

### 3) Train

```bash
python train_model.py   --data_dir data   --batch 2048   --epochs 10   --lr 1e-3   --weight_decay 1e-3   --cosine_alpha 0.1   --steps_per_execution 128
```

Expected validation: ~58% Top-1 and ~87% Top-5 with stable convergence.

### 4) Inference / Demo (API)

```bash
# do NOT commit your key
export HANGMAN_API_KEY=your_key_here      # Windows: set HANGMAN_API_KEY=your_key_here
python hangman_api_user.py
```

Prints proposed letters with confidences across turns.

---

## Method

**Inputs**
- Masked word: one-hot per position over 28 symbols (26 letters + underscore + pad)  
- Guessed letters: 26-dim binary vector  
- Meta: normalized word length and tries remaining

**Targets**
- Categorical over 26 letters; training uses argmax labels from a soft target distribution built from candidate frequencies

**Data generation**
- Filter dictionary by mask consistency  
- Count per-letter frequencies over remaining candidates  
- Zero out already-guessed letters  
- Truncate to top-M and normalize to a distribution

**Model**
- Dense(128) projection → add simple positional encoding  
- 1× Transformer block (1 head, FF=256)  
- Global average pool → concat with guessed + meta → Dense(256)  
- Output: Softmax(26)  
- Optimizer: AdamW, cosine LR decay (1e-3 → ~1e-4), weight decay=1e-3, `alpha=0.1`  
- Batch size 2048, 10 epochs

**Notes on compute**
- Short runs were used due to time limits; larger models or more epochs can improve accuracy

---

## Reproducing analysis plots

Training curves, LR schedule, confusion matrix, calibration, Top-K, bigrams, word-length histograms, and gameplay metrics can be regenerated from logs produced by `train_model.py`.

---

