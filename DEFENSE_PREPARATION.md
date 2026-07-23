# Monet — Final Year Project Defense Preparation

> This document is your study guide for tomorrow's defense. It explains the
> project in plain language, then lists the questions an external examiner is
> most likely to ask, with model answers. Use it as the basis for your written
> report and your PowerPoint slides.

---

## 0. One-page cheat sheet (memorize these numbers)

| Item | Value |
|---|---|
| What it is | Browser extension that flags AI-generated YouTube Shorts |
| Browsers | Chrome + Firefox |
| Frames analysed per video | 8 captured, 3 go through the deep detector, 4 through texture |
| Deep detector model | SigLIP2-SO400M + DINOv2 ViT-Large ensemble (LoRA fine-tuned) |
| Fusion classifier | Random Forest (binary: AI vs not-AI) |
| Number of engineered features | 17 |
| Training data | 1,521 videos (540 real, 419 suspicious, 562 AI) |
| Best RF hyperparameters | n_estimators=400, max_depth=10, min_samples_split=5, min_samples_leaf=2, class_weight=balanced |
| Cross-validation | 5-fold Stratified — Accuracy 81.9%, F1 0.737, ROC-AUC 0.842 |
| Final binary accuracy | **89.3%** (at score ≥ 0.65) |
| AI precision / recall | 98.1% / 72.6% (at 0.65); 90.4% / 85.4% (at optimal 0.52) |
| Top features | vit_mean 0.27, vit_max 0.22, vit_std 0.12 (detector dominates) |

---

## 1. Project overview (the "elevator pitch")

**Monet** is a browser extension that helps viewers spot AI-generated
("synthetic") YouTube Shorts. When a Short loads, the extension grabs a few
frames, sends them to a local Python server, and the server returns an "AI
likelihood" score. The extension then draws a coloured border and a label
around the video — **Low AI**, **Suspicious**, or **Likely AI** — plus a
breakdown popup explaining why.

The key idea: **no single test reliably detects AI video**, so the system
combines several independent signals:

1. A **deep neural detector** (the strong signal).
2. **Classical computer-vision heuristics** (texture + semantics).
3. **Metadata** (title/description/YouTube AI-disclosure labels).
4. A **Random Forest** classifier that learns how to fuse all of the above
   into one calibrated score.

This "hybrid" design is the core contribution and the main thing to emphasise.

---

## 2. System architecture & data flow

```
YouTube Short playing in browser
        │
        │  extension captures 8 frames (480×854, JPEG)
        ▼
   WebSocket (JSON + base64 frames)
        │
        ▼
 Python FastAPI server  (server.py)
        │
        ├──► Texture analyzer   (4 frames, classical CV)
        ├──► Semantic analyzer  (1 frame,  classical CV)
        ├──► Deep AI detector   (3 frames, SigLIP2+DINOv2)
        ├──► Color + temporal + digital-content checks
        └──► Metadata check (title/description/disclosure)
                    │
                    │  17 features → Random Forest
                    ▼
            Final AI score (0–1) + label + reason
                    │
        ◄───────────┘  WebSocket reply
        │
   extension draws coloured border + breakdown popup
```

**Key design points to mention:**
- The three heavy analyzers (texture, semantics, detector) run **concurrently**
  with `asyncio.gather` so latency stays low.
- Communication is over a **WebSocket** on `127.0.0.1:8000` (everything stays
  local — no data leaves the user's machine, which is a privacy plus).
- The server is a **FastAPI** app served by **Uvicorn**.

---

## 3. The Machine Learning pipeline (step by step)

This is the heart of the project. Know it cold.

### Step 1 — Frame capture
The extension captures **8 frames** at 480×854 resolution while the Short
plays, and sends them as base64 JPEGs. (Honest limitation: capture is
time-based, so re-running on the same video can give slightly different scores.
See §9.)

### Step 2 — Feature extraction (produces 17 numbers per video)
Each analyzer turns pixels into a few summary numbers:

- **Deep detector** → `vit_mean`, `vit_max`, `vit_std` (AI probability over the
  sampled frames).
- **Texture analyzer** → `texture_mean`, `texture_max`, `texture_std`.
- **Color analyzer** → `color_mean`, `color_max`, `color_std`.
- **Semantic analyzer** → `semantic` (one score).
- **Metadata** → `metadata_score`, `metadata_hits`.
- **Temporal consistency** → `temporal_diff_mean`, `temporal_diff_std`,
  `brightness_flicker`, `saturation_flicker` (frame-to-frame changes).
- **Digital-content check** → `digital_penalty` (reduces the score for
  flat/digital/cartoon frames so they aren't false positives).

That's the **17-feature vector** that feeds the Random Forest.

### Step 3 — The Random Forest fusion classifier
The RF takes the 17 features and outputs a probability. This is then blended
with a hand-designed "manual score" to produce the final 0–1 score (see §7).

### Step 4 — Threshold → label
- score < 0.46 → **Low AI** (green)
- 0.46 ≤ score < 0.65 → **Suspicious** (orange)
- score ≥ 0.65 → **Likely AI** (red)

---

## 4. The deep AI detector (SigLIP2 + DINOv2)

This is the strongest single signal. Be ready to explain it.

**What it is:** an ensemble of two pre-trained vision transformers:
- **SigLIP2-SO400M** (patch14, 384px) — a vision-language model; good at
  semantic/high-level features.
- **DINOv2 ViT-Large** — a self-supervised vision model; very strong at
  fine-grained visual features and textures.

**Why combine both?** They are trained differently and make different kinds of
errors, so ensembling them gives better detection than either alone. Their
feature vectors are concatenated and fed to a small MLP "classification head"
that outputs one logit → `sigmoid(logit)` = AI probability.

**Fine-tuning method: LoRA (Low-Rank Adaptation).** Instead of re-training the
huge backbones (hundreds of millions of parameters), LoRA freezes the original
weights and injects tiny trainable "low-rank" matrices into the attention
layers (q/v projections). This trains far fewer parameters, needs less memory,
and avoids catastrophic forgetting.
- LoRA rank = 32, alpha = 64, dropout = 0.1.

**Where the weights come from:** the fine-tuned checkpoint
`Bombek1/ai-image-detector-siglip-dinov2` on HuggingFace (downloaded once and
cached locally). So I used transfer learning — I did **not** train the detector
from scratch.

**Inference:** 3 evenly-spaced frames are resized to 392×392, passed through
the model in a batch, and I report the **mean, max, and std** of the per-frame
AI probabilities. `max` catches the single strongest AI frame.

---

## 5. The classical analyzers (hand-crafted features)

These are traditional computer-vision checks. They matter because (a) they
catch artifacts the neural net might miss, and (b) they're cheap and
explainable.

### Texture analyzer (6 sub-checks, weighted sum)
1. **Frequency analysis** — FFT; AI images often lack high-frequency detail.
2. **Repetition** — patch cross-correlation; AI can repeat textures.
3. **Edge artifacts** — Canny + Sobel; looks for "halos" around edges.
4. **Color banding** — in LAB color space; gradient stepping.
5. **Compression artifacts** — 8×8 block boundary analysis.
6. **Detail consistency** — Laplacian sharpness map across a 4×4 grid.

### Semantic analyzer (5 sub-checks)
1. **Text** — OCR (optional) looking for garbled/nonsense words.
2. **Object counting** — Hough circles + contours + Hu moments; flags weird
   counts or repeated shapes.
3. **Symmetry** — left/right flip difference; flags unnaturally perfect
   symmetry.
4. **Watermarks** — corner analysis + FFT peaks.
5. **Context** — edge vs centre complexity; flags "subject pasted on
   background".

Each sub-check returns a 0–1 score; they're combined with fixed weights and a
small multiplier when several fire at once.

---

## 6. The Random Forest classifier

**What is a Random Forest?** An ensemble of many decision trees. Each tree is
trained on a random subset of data and features; the final prediction is the
majority vote (classification) or average (probability). It's robust, needs
little tuning, handles mixed feature types, and gives feature importances.

**Why I chose it:**
- Works well on **small/medium tabular datasets** (1,521 rows × 17 columns) —
  better than a deep net here, which would overfit.
- **Interpretable** (feature importances) — important for a defense.
- **Fast** to train and infer.
- Handles the **non-linear fusion** of heterogeneous signals well.

**Preprocessing:** `StandardScaler` (zero mean, unit variance). *Note:* tree
models don't strictly need scaling, but it's harmless and keeps the pipeline
consistent with the saved model.

**Hyperparameters (chosen by GridSearchCV):**
| Parameter | Value | Meaning |
|---|---|---|
| n_estimators | 400 | number of trees |
| max_depth | 10 | max depth of each tree (limits overfitting) |
| min_samples_split | 5 | min samples to split a node |
| min_samples_leaf | 2 | min samples at a leaf |
| class_weight | balanced | corrects class imbalance |
| random_state | 42 | reproducibility |

The grid searched: n_estimators {100,200,400} × max_depth {None,10,20} ×
min_samples_split {2,5} × min_samples_leaf {1,2}, optimising F1.

---

## 7. How the final score is computed (the fusion)

Be ready to walk through this — examiners love it.

1. Build the **manual score**: a fixed weighted blend of the signals.
   The deep detector dominates (weight 0.64), then metadata (0.12), texture
   (0.10), semantics (0.09), color (0.02), RF (0.03). A multiplier kicks in
   when several signals agree strongly (×1.18 / ×1.35 / ×1.55). The
   **digital penalty** then *reduces* the score for flat/digital content.
2. Get the **RF probability** (the model's own AI probability).
3. **Blend** (binary model, when RF confidence ≥ 0.45):
   `final = 0.30 × manual_score + 0.70 × rf_score`.
4. Apply optional **evidence floors** (raise the score when the detector or
   metadata is very strong), then clamp to [0, 1].
5. Map to a label with the 0.46 / 0.65 thresholds.

**Why blend at all? Why not just use the detector?** Because the detector
alone has a high false-positive rate (it often reads real footage as ~95% AI).
The blend averages out that noise using corroborating signals and the RF's
learned discrimination — this is what lifts accuracy from ~50% (detector alone)
to **89%**. *This is the single most important insight of the project.*

---

## 8. Training & evaluation

### Data
- **1,521 videos** at 480p, organised in three folders: `urls_real` (540),
  `urls_suspicious` (419), `urls_fake` (562). Source URLs are in `.txt` lists.
- For the **binary** model, real + suspicious are merged into "not-AI" (959)
  vs "AI" (562).

### Methodology
- **Stratified 5-fold cross-validation** (preserves class ratios in each fold).
- **GridSearchCV** for hyperparameter tuning (optimised F1).
- Honest CV with the best params: **Accuracy 81.9%, F1 0.737, ROC-AUC 0.842**.
- Final model trained on **all** data and saved as `random_forest.pkl`.
- A held-out 80/20 split (stratified) was used for threshold tuning + reporting.

### Final binary evaluation (on the full cached set)
At the deployed threshold (score ≥ 0.65):

| Metric | Value |
|---|---|
| Accuracy | 89.3% |
| AI precision | 98.1% |
| AI recall | 72.6% |
| AI F1 | 83.4% |

Confusion matrix (true ↓, pred →): not-AI 951 / 8, AI 154 / 408.

A threshold sweep showed the **optimal** cutoff is ~0.52 (accuracy 91.3%,
precision 90.4%, recall 85.4%). The 0.65 cutoff trades recall for very high
precision — a deliberate "be sure before you flag it" choice.

### Feature importance (top)
vit_mean 0.271 · vit_max 0.221 · vit_std 0.115 · saturation_flicker 0.053 ·
brightness_flicker 0.052 · texture_std 0.048 …

The detector features together account for ~61% of importance — confirming the
detector is the backbone, with the classical features adding corroborating
signal.

---

## 9. Honest limitations (examiners respect honesty — have these ready)

1. **Frame capture is not perfectly deterministic.** Frames are sampled on a
   timer during playback, so the same video can score slightly differently
   across runs. A fix is to seek to fixed timestamps — noted as future work.
2. **Detector false positives on real footage.** The raw detector often reads
   real videos as high-AI; the fusion exists precisely to suppress this.
3. **Dataset size (1,521 videos).** Larger, more diverse data would generalise
   better to new generators (Sora, Veo, etc.).
4. **Static metadata.** Detection relies partly on titles/descriptions, which
   can be missing or manipulated.
5. **Browser/DRM constraints.** Some platform restrictions can limit frame
   access.
6. **Binary model has no real "Suspicious" discrimination** — the middle band
   is an uncertainty zone, not a learned third class.

---

## 10. Likely defense questions + model answers

### A. Architecture / general
**Q: Walk me through your system.**
A: (Give §2 — browser captures frames → WebSocket → Python server runs 3
analyzers in parallel → 17 features → Random Forest → score → labelled border.)

**Q: Why a hybrid approach instead of just a neural network?**
A: A single detector has a high false-positive rate on real footage (≈50%
precision alone). Combining it with classical CV checks, metadata, and a
learned fusion model lifts precision to 98% and accuracy to 89%. The classical
features are also cheap and explainable.

**Q: Why a local server and not cloud?**
A: Privacy (frames never leave the machine), low latency, and no API costs.

### B. Machine learning fundamentals
**Q: What is a Random Forest and why this choice?**
A: Ensemble of decision trees trained on random subsets, combined by voting.
Chosen because it works well on small tabular data, resists overfitting,
is interpretable (feature importance), and trains fast. (See §6.)

**Q: What is cross-validation? Why stratified?**
A: K-fold CV splits data into K folds, trains on K-1, validates on 1, rotating
so every sample is validated once. **Stratified** keeps the class proportions
the same in each fold — essential here because classes are imbalanced
(959 vs 562).

**Q: What does class_weight='balanced' do?**
A: It weights each class inversely to its frequency so the majority class
(not-AI) doesn't dominate the loss. Prevents the model from trivially
predicting "not-AI" everywhere.

**Q: Explain precision, recall, F1.**
A: **Precision** = of all flagged AI, how many really are (TP/(TP+FP)).
**Recall** = of all real AI, how many we caught (TP/(TP+FN)). **F1** = their
harmonic mean, balancing both. For a warning tool, recall matters (don't miss
AI), but precision protects trust (don't cry wolf).

**Q: Why is accuracy alone misleading here?**
A: Class imbalance. Always guessing "not-AI" would score 63% accuracy while
detecting zero AI. F1/precision/recall and the confusion matrix show real
performance.

**Q: Explain your confusion matrix.**
A: 951 true not-AI, 8 false alarms, 154 missed AI, 408 correctly caught AI.
So precision 408/(408+8)=98%, recall 408/(408+154)=73%.

**Q: What is overfitting and how did you prevent it?**
A: Overfitting = memorising training data, failing on new data. Prevented via:
limiting tree depth (10), min_samples_leaf/split, balanced class weights,
cross-validation to measure generalisation, and the held-out test set.

**Q: Train/test split? Any data leakage?**
A: Stratified 80/20 split with a fixed random_state (42) for reproducibility.
Features are extracted once and cached; the split happens after extraction, on
the feature matrix, so there's no leakage between folds.

### C. Deep learning
**Q: What are SigLIP and DINOv2?**
A: SigLIP2 is a vision-language transformer (image+text contrastive training);
DINOv2 is a self-supervised vision transformer (no labels). Both produce rich
image features. (See §4.)

**Q: What is LoRA and why use it?**
A: Low-Rank Adaptation freezes the huge pre-trained weights and trains small
low-rank matrices inside the attention layers. Far fewer trainable params →
less memory/compute, no catastrophic forgetting. Rank 32, alpha 64.

**Q: Did you train the detector from scratch?**
A: No — transfer learning. I used a publicly fine-tuned checkpoint
(Bombek1/ai-image-detector-siglip-dinov2) and only run inference. Training that
detector was outside my scope; my contribution is the fusion system around it.

**Q: Why sigmoid at the output?**
A: The head outputs a raw logit; sigmoid maps it to a [0,1] probability of
"AI", which is what I report as the detector score.

**Q: Why 3 frames and max/mean/std?**
A: 3 keeps inference fast; `max` catches the single strongest AI frame,
`mean` averages out noise, `std` measures consistency — all useful features.

### D. Scoring & fusion
**Q: How is the final score computed?**
A: (Walk through §7 — manual weighted blend → digital penalty → RF blend
0.30/0.70 → evidence floors → thresholds.)

**Q: Why does your headline score differ from the detector's raw %?**
A: Deliberately. The raw detector is noisy (false positives on real video).
The headline is a *fused* score that the Random Forest and corroborating
signals have calibrated. Matching the raw detector would crash precision.

**Q: How did you choose 0.46 and 0.65?**
A: 0.65 was chosen for high precision (be sure before flagging); a threshold
sweep showed 0.52 maximises accuracy (91%). It's a tunable precision/recall
trade-off.

### E. Data & features
**Q: What are your 17 features?**
A: (List §3 Step 2 — detector stats, texture stats, color stats, semantic,
metadata, temporal consistency, digital penalty.)

**Q: Which features matter most?**
A: The detector features (vit_mean 0.27, vit_max 0.22, vit_std 0.12 — ~61%
combined), then flicker/temporal and texture. Metadata is near-zero because
most training clips have no AI-disclosure text.

**Q: What is the digital penalty?**
A: A reduction applied when a frame has very few unique colours (flat
digital/cartoon content), because the detector false-positives on such frames.

### F. Evaluation & results
**Q: What's your accuracy?**
A: 89.3% binary accuracy at the deployed threshold; 91.3% at the optimal
threshold. CV accuracy 81.9%, ROC-AUC 0.842.

**Q: Is 72.6% recall acceptable?**
A: It's conservative — we miss ~27% of AI but almost never false-alarm (98%
precision). For a tool that *accuses* content of being AI, high precision is
safer. The threshold can be lowered to raise recall if desired.

**Q: How would you improve it?**
A: Deterministic frame capture, larger/more diverse dataset covering newer
generators, a true 3-class model for the Suspicious band, and possibly a
video-level temporal model (e.g., 3D-CNN or VideoMAE) instead of frame
sampling.

### G. Possible "trick" / probing questions
**Q: Does the Random Forest need feature scaling?**
A: Strictly, no — trees split on thresholds and are scale-invariant. I keep
StandardScaler in the pipeline for consistency with the saved model and because
it doesn't hurt.

**Q: What's the difference between your binary and a 3-class model?**
A: Binary merges real+suspicious into "not-AI" and learns only the AI/not-AI
boundary — giving 89% accuracy and 98% AI precision. A 3-class model would
have to separate a fuzzy "suspicious" middle class, which is harder and lowers
overall accuracy, so I chose binary and treat the middle band as an
"uncertain" zone.

**Q: Could an adversary fool this?**
A: Partly — e.g., heavy compression or no metadata reduces signal. No detector
is bulletproof; this is a best-effort warning system, not a verdict.

---

## 11. Suggested PowerPoint structure (10–12 slides)

1. **Title + problem** — AI content flooding YouTube; viewers can't tell.
2. **Objective** — a browser extension that flags AI Shorts.
3. **System architecture** — the diagram in §2.
4. **The ML pipeline** — frames → features → RF → score (§3).
5. **Deep detector** — SigLIP2+DINOv2+LoRA (§4).
6. **Classical analyzers** — texture + semantics (§5).
7. **Random Forest + fusion** — the blend (§6, §7).
8. **Dataset** — 1,521 videos, 3 classes.
9. **Training methodology** — CV, GridSearch, hyperparameters.
10. **Results** — the metrics table + confusion matrix (§8).
11. **Limitations & future work** (§9).
12. **Demo / conclusion**.

---

## 12. Final tips for tomorrow

- **Lead with the hybrid idea** — it's your main contribution.
- **Know the numbers** in §0 by heart.
- **Be honest about limitations** (§9) — examiners reward self-awareness.
- **Don't overclaim** — say "the detector was pre-trained; I built the fusion
  system," not "I trained a state-of-the-art detector."
- If asked something you don't know, say: "That's outside my current scope, but
  I'd approach it by …" — then reason aloud.

Good luck.
