# Monet — Architecture & How It Works (in detail)

Monet is a browser-extension + local-server system that detects AI-generated
video content. It overlays a colored "AI score" badge on videos (targeting
YouTube Shorts) in real time, entirely on your own machine — no video data
leaves your computer.

This document explains the whole system end to end: the parts, how they talk
to each other, every analyzer, the fusion math, the training pipeline, and the
performance design.

---

## 1. Big picture

There are two pieces that run on your machine:

```
┌──────────────────────────────┐        WebSocket (ws://localhost:8000)
│  Browser Extension           │◄──────────────────────────────────────┐
│  (Chrome / Edge / Firefox)   │                                       │
│  - watches <video> elements  │                                       │
│  - captures 8 frames         │       ┌───────────────────────────────┴──┐
│  - collects page metadata    │       │  Local Python Server (FastAPI)     │
│  - draws the badge UI        │──────►│  server/server.py                  │
│                              │ JSON  │                                   │
└──────────────────────────────┘       │  MonetAnalyzer orchestrates:       │
                                       │   • Texture analyzer (CV/FFT)      │
                                       │   • Semantic analyzer (CV/OCR)     │
                                       │   • SigLIP2 + DINOv2 deep detector │
                                       │   • Heuristics (color/temporal/…)  │
                                       │   • Random Forest fusion model     │
                                       │   → final ai_score 0..1            │
                                       └────────────────────────────────────┘
```

- The **extension** is the "eyes + UI". It sees videos playing, grabs frames,
  and shows the result.
- The **server** is the "brain". It does all the heavy ML work and returns a
  score.
- They communicate over a **local WebSocket** on port 8000. Nothing goes to
  the internet except the models' one-time download.

---

## 2. End-to-end data flow (what happens when a Short plays)

This is the life of a single analysis, step by step.

### On the extension side (`extension/chrome/content.js`)

1. **A polling loop** (`watchVideos`) runs every 1 second and finds every
   `<video>` element on the page.
2. For each video it checks the aspect ratio (`height >= width`, i.e. portrait
   Shorts) and whether the video is actually visible in the viewport (YouTube
   pre-loads the next/previous Short off-screen — those are skipped).
3. When a qualifying video starts playing, `tryAnalyze()` → `analyze()` fires.
4. `analyze()` creates a purple **"Scanning…"** border + label around the
   video so the user sees immediately that work is happening.
5. **Frame capture**: it draws the video to an off-screen canvas 8 times,
   600 ms apart, at 480×854, JPEG quality 0.85, producing 8 base64 strings.
   (8 frames over ~4.2 s gives temporal artifacts a chance to show up.)
6. **Metadata collection** (`collectMetadata`): grabs title, channel,
   description, hashtags, visible page text, and crucially the
   **YouTube "Altered or synthetic content" disclosure** if present.
7. It sends one JSON message over the WebSocket:
   `{ type: "analyze", frames: [...8 base64...], videoId, metadata }`.

> **Short-circuit:** if the YouTube AI disclosure is detected, the extension
> marks the video **100% AI locally** and never even sends frames to the
> server. This is instant and reliable.

### On the server side (`server/server.py`)

8. The WebSocket handler receives the message, decodes the 8 base64 frames back
   into OpenCV BGR arrays.
9. It calls `MonetAnalyzer.analyze_video(frames, metadata)` (details in §5).
10. `analyze_video` runs the analyzers, builds a feature vector, asks the
    Random Forest for a fused score, applies override/floor logic, and returns
    a result dict containing `ai_score`, `label`, `confidence_level`,
    `detection_reason`, and a per-analyzer `breakdown`.
11. The server sends the result JSON back over the same WebSocket.

### Back on the extension

12. `onmessage` looks up the border by `videoId`, then `updateColor()`:
    - picks a color/label by the score band (red "Likely AI" ≥0.65, orange
      "Suspicious" ≥0.46, green "Low AI" <0.46),
    - sets an animated glowing border,
    - renders the floating label (`"73% Likely AI"`),
    - builds a **breakdown popup** (texture, color, semantics, detector,
      metadata, digital penalty, confidence) that opens on click.
13. The result is cached per video element so the border persists when you
    replay the same Short.

If the server never replies within 30 s, a stale-border cleaner removes the
"Scanning…" border and allows a re-analysis. If the WebSocket drops, the
extension auto-reconnects every 3 s.

---

## 3. The browser extension in detail

File: `extension/chrome/content.js` (Firefox copy is identical). A
Manifest V3 content script that runs on all URLs.

**Main object `Monet`** holds:
- `socket` — the WebSocket to the server.
- `borders: Map<id, element>` — pending/active border overlays.
- `videoResults: Map<videoEl, info>` — cached results so badges persist.

**Subsystems:**

| Method | Job |
|---|---|
| `init` | Wires up styles, connection, video watcher, stale-border watcher. |
| `connect` / `scheduleReconnect` | Opens `ws://localhost:8000/ws`, auto-reconnects every 3 s on close. |
| `watchVideos` | 1 s poll: detects new videos, SPA navigation, source changes; attaches `play` listeners. |
| `tryAnalyze` / `analyze` | Gate-keeping (skip already-analyzed, skip landscape, skip off-screen) + frame capture + send. |
| `collectMetadata` / `collectYoutubeAiDisclosure` | Scrapes title/channel/description/hashtags + the official YouTube AI disclosure pill. |
| `getContainer` | Finds the right ancestor element to attach the border to (Shorts shell / reel / article). |
| `_createBorderElement` | Builds the purple scanning border + "Scanning…" label. |
| `updateColor` | Recolors the border, sets the label text, builds the breakdown popup. |
| `watchStaleBorders` | 10 s poll: removes borders stuck "Scanning" >30 s. |

**Metadata / disclosure detection nuance:**
`collectYoutubeAiDisclosure` only matches explicit phrases like
"altered or synthetic content", "made with ai", etc., and deliberately
**excludes** titles, channel names, descriptions, and comments. An earlier
version matched a bare "AI"/"synthetic" anywhere and falsely flagged every
video — this is why the matcher is now phrase-based and location-restricted.

**Permissions** (`manifest.json`): `activeTab` + `<all_urls>` host permission
so the content script can read the page and draw overlays on any site.

---

## 4. The server in detail

File: `server/server.py`. A FastAPI app served by Uvicorn on `127.0.0.1:8000`.

**Startup (`MonetAnalyzer.__init__`):**
- Prints an animated boot sequence (the `type_text` + spinner helpers) — purely
  cosmetic.
- Loads four modules with a spinner and status check:
  1. `TextureAnalyzer`
  2. `SemanticAnalyzer`
  3. `SigLIPDinoV2Detector` (the deep model — by far the heaviest)
  4. `RandomForestClassifier` (loads `analyzers/random_forest.pkl`)
- Starts a "Ready & Waiting for extension" spinner that stops on first connect.

**WebSocket endpoint `/ws`:**
- Accepts the connection, then loops `receive_text()` → `json.loads`.
- For each `analyze` message: decodes frames (base64 → PIL → numpy → BGR),
  calls `analyze_video`, attaches the `videoId`, sends the result JSON back.

**Why a WebSocket (not HTTP)?** One persistent bidirectional channel lets the
server push results whenever ready and lets the extension queue multiple
videos. It also avoids per-request overhead.

---

## 5. `analyze_video` — the analysis pipeline

This is the heart of the server. Given ~8 frames + metadata, it produces the
final score. Conceptually:

```
frames ──┬─► Texture analyzer      ─► texture_{mean,max,std}
         ├─► Semantic analyzer     ─► semantic_score
         ├─► SigLIP+DINOv2         ─► vit_{mean,max,std}
         ├─► color heuristic       ─► color_{mean,max,std}
         ├─► digital heuristic     ─► digital_penalty
         ├─► temporal heuristic    ─► diff/flicker stats
metadata ┴─► metadata heuristic    ─► metadata_{score,hits}
                                        │
                   ┌────────────────────┴────────────────────┐
                   │  17-feature vector → Random Forest       │
                   │  + manual weighted score                 │
                   │  + evidence floors + overrides           │
                   └────────────────────┬─────────────────────┘
                                        ▼
                          final ai_score (0..1) + label + confidence
```

**Parallelism:** the three heavy/independent analyzers (texture, semantics,
deep detector) are run **concurrently** with `asyncio.to_thread` +
`asyncio.gather`. The cheap heuristics (color, digital, temporal, metadata)
run after, on the main coroutine. This matters for performance (see §9).

**Frame sampling (after the perf tuning):**
- Texture: 4 evenly-spaced frames.
- Deep detector: 3 evenly-spaced frames (configurable via
  `MONET_DETECTOR_FRAMES`).
- Heuristics: use all received frames.

**Outputs of each stage** are combined into a 17-element feature vector (see
`RandomForestClassifier.FEATURE_NAMES`):

```
texture_mean, texture_max, texture_std,
color_mean,   color_max,   color_std,
digital_penalty,
semantic,
vit_mean,     vit_max,      vit_std,
metadata_score, metadata_hits,
temporal_diff_mean, temporal_diff_std,
brightness_flicker, saturation_flicker
```

---

## 6. The analyzers in detail

### 6.1 Texture analyzer (`analyzers/texture.py`)

Classical computer-vision signals that AI image/video generators often get
"wrong". It runs **six sub-analyzers** on a grayscale frame, each returning a
0..1 score, then weights them (weights sum to 1.0) and applies an anomaly
multiplier if several sub-scores are high.

| Sub-analyzer | What it looks for | Weight |
|---|---|---|
| **Frequency** | FFT energy distribution. AI images often lack high-frequency detail (too smooth). | 0.20 |
| **Repetition** | Patch cross-correlation. Repeated texture blocks signal synthesis. | 0.15 |
| **Edge artifacts** | Canny + Sobel gradient ratio near edges. AI often leaves halos. | 0.20 |
| **Color banding** | L-channel gradient analysis in LAB space. Stepped gradients = 8-bit banding. | 0.15 |
| **Compression** | 8×8 block-boundary vs interior differences. Detects JPEG/block inconsistency. | 0.15 |
| **Detail consistency** | Laplacian-variance sharpness map on a 4×4 grid. Abrupt sharpness changes. | 0.15 |

If ≥3 sub-scores exceed 0.4 the final is boosted ×1.3; ≥2 → ×1.15.

### 6.2 Semantic analyzer (`analyzers/semantics.py`)

Higher-level "does the scene make sense" signals on the middle frame. **Five
sub-analyzers**, weights sum to 1.0, same anomaly-boost scheme.

| Sub-analyzer | What it does | Weight |
|---|---|---|
| **Text** | OCR (Tesseract) for garbled/nonsense text — a classic AI tell. **Disabled by default** (`MONET_OCR=1` to enable); falls back to edge-pattern text detection. | 0.25 |
| **Object counting** | Hough circles + contour shapes + Hu-moment matching. Flags odd counts / excessive repetition. | 0.25 |
| **Symmetry** | Left/right mirror difference. "Unnaturally perfect" symmetry hints at AI. | 0.20 |
| **Watermarks** | Corner std/edge ratios + bottom-strip FFT peaks. Detects SynthID-style watermarks. | 0.15 |
| **Context** | Edge-vs-center complexity ratio in HSV. "Subject pasted on background" look. | 0.15 |

### 6.3 The deep detector (`analyzers/siglip_dinov2_detector.py`)

The strongest single signal. An **ensemble of two large vision transformers**
fine-tuned to classify an image as AI-generated vs authentic.

**Architecture (`EnsembleAIDetector`):**
- **SigLIP2** (`google/siglip2-so400m-patch14-384`) — ~400M-param vision
  transformer, image size 384, patch 14. Its `pooler_output` is the SigLIP
  feature vector (1152-dim).
- **DINOv2** (`vit_large_patch14_dinov2.lvd142m` via `timm`) — ~300M-param
  self-supervised ViT-Large, image size 392. Produces a 1024-dim feature.
- The two feature vectors are **concatenated** (2176-dim) and fed to an
  `ClassificationHead`: `LayerNorm → Linear → GELU → Dropout → Linear → GELU
  → Dropout → Linear(→1)`. Output is a single logit → `sigmoid` = AI
  probability.

**Fine-tuning with LoRA** (parameter-efficient):
- SigLIP gets PEFT LoRA adapters on `q_proj`/`v_proj` (rank 32, alpha 64).
- DINOv2 gets custom `LoRALinear` wrappers on each block's `qkv` projection.
- Only the LoRA weights + classification head are trained; the base ViTs stay
  frozen. This is what's stored in `models/siglip_dinov2/pytorch_model.pt`
  (~2 GB).

**Inference path (`analyze_frame_stats`):**
- Sample N frames (default 3), batch them.
- Each frame → BGR→RGB → PIL → SigLIP processor (384) + DINOv2 transform (392).
- One forward pass per batch; collect per-frame AI probabilities.
- Returns `mean`, `max`, `std`, and a description. **`max` is used as the
  detector score** (`vit_score = vit_max`) — i.e. "the most AI-looking frame
  wins", which is robust because one fake frame is enough evidence.

**Device backends** (see §9): CUDA → Intel XPU (Arc) → Apple MPS → CPU, with
a warmup sanity check and automatic fallback to CPU if the accelerator fails.

### 6.4 The lightweight heuristics (in `server.py`)

These are cheap numpy/OpenCV computations, not learned models:

- **Color** (`_analyze_color`): mean HSV saturation per frame; high saturation
  bumps the score (over-saturated content is an AI tell).
- **Digital content** (`_detect_digital_content`): color-diversity ratio on the
  first frame. Very low unique-color counts → "digital/cartoon" content
  (digital *penalty*, which actually *reduces* the AI score — see §7).
- **Temporal** (`_compute_temporal_features`): frame-to-frame mean abs
  difference + brightness/saturation flicker (std across frames). AI video
  generators often flicker unnaturally.
- **Metadata** (`_analyze_metadata`): keyword scan over title/description/
  channel/hashtags/page-text. Strong terms ("ai generated", "sora", "runway"…)
  add to the score; weak terms ("cgi", "vfx") add a little. Capped at 0.85.

---

## 7. The fusion engine (`analyzers/random_forest.py`)

This turns the 17 features into one calibrated `ai_score`. It is deliberately
a **hybrid** of a learned model and hand-tuned logic, so the system stays
robust even when the RF is uncertain.

### Step A — Random Forest prediction
The feature vector is standardized (`StandardScaler`) and passed to a
`RandomForestClassifier` (100 trees, max depth 10, class-weight balanced),
trained on 3 classes (see §8). From the predicted probabilities:
- `rf_score` = weighted blend of the "suspicious" and "ai" class
  probabilities (`proba[1]*0.7 + proba[2]` for the 3-class model).
- `rf_confidence` = probability of the predicted class.

### Step B — Manual weighted score (a safety net)
A transparent linear combination that doesn't trust the RF alone
(`_manual_score`):

```
weighted = texture·0.10 + semantic·0.09 + color·0.02
         + vit·0.64 + metadata·0.12 + rf_score·0.03
```

(`vit` = a 0.45/0.55 blend of vit_mean and vit_max.) Note the deep detector
dominates (0.64) — it's the most reliable signal. Strong-signal counts (≥2 or
≥3 of texture/semantic/vit/metadata high) multiply this by 1.35–1.55. The
digital penalty then *reduces* it: `ai = weighted·(1 − digital_penalty·0.5)`.

### Step C — Blending RF + manual
- If the RF is confident (`rf_confidence ≥ 0.45`): final = 0.65·manual +
  0.35·rf (RF gets real weight but manual keeps it grounded).
- Otherwise: final = manual only (don't trust an unsure RF).

### Step D — Evidence floors (`_apply_evidence_floors`)
Hard minimums so strong individual signals can't be buried:
- **Detector authority:** `vit_max ≥ ai_threshold` → score ≥ `vit_max`
  (the dedicated detector is the strongest signal, so the headline never
  contradicts it); `vit_max ≥ suspicious_threshold` → ≥ `suspicious_threshold`.
- `vit_mean ≥ 0.55` → score ≥ 0.60.
- `vit_max ≥ 0.70` with a corroborating signal → ≥ 0.60.
- `metadata ≥ 0.70` → ≥ `ai_threshold`.
- Conversely, high digital penalty + low vit/metadata *caps* the score at
  0.49 (digital cartoons shouldn't read as AI).

### Step E — Overrides
- **YouTube AI disclosure** detected → force `max(score, 0.95)`,
  "STRONG AI EVIDENCE".

### Step F — Label + color
| Score | Label | Color |
|---|---|---|
| ≥ 0.65 | STRONG AI EVIDENCE | red `#ef4444` |
| ≥ 0.46 | MIXED AI EVIDENCE | orange `#f97316` |
| < 0.46 | LOW AI EVIDENCE | green `#22c55e` |

### Step G — Confidence level
A separate human-readable certainty, based on **agreement** between the RF
score and the deep detector:
- YouTube disclosure → **High**.
- RF and detector agree (both say AI, or both say real) → **High**.
- They disagree **and** the score is near 0.5 → **Low**.
- Otherwise → **Medium**.

---

## 8. The training pipeline (`train_model.py`, `evaluate_model.py`)

The Random Forest is trained offline from labeled videos, then the resulting
`random_forest.pkl` is shipped and used at runtime.

**Dataset:** three folders under `server/training_data/`:
- `urls_real/` → label **0 (real)**
- `urls_suspicious/` → label **1 (suspicious)**
- `urls_fake/` → label **2 (AI)**

**Feature extraction (`_extract_features_from_video`):**
For each training video, the exact same analyzers used at runtime (texture,
semantics, deep detector, heuristics) compute the **same 17-feature vector**.
This 1:1 match between training and inference features is essential — if you
change the analyzers at runtime you must retrain (or at least re-extract
features).

**Model:**
- `StandardScaler` + `RandomForestClassifier` inside an sklearn `Pipeline`.
- Hyperparameters tuned with `GridSearchCV` + `StratifiedKFold` cross-validation.
- Class-imbalanced data handled with `class_weight='balanced'` and optional
  per-class caps (`--limit`).

**Threshold tuning:**
The default decision thresholds (AI 0.65, suspicious 0.46) aren't fixed —
`train_model.py` / `evaluate_model.py` sweep candidate thresholds and pick the
pair that maximizes a target metric (accuracy/F1) on the validation set, then
store them inside the `.pkl` (`rf.ai_threshold`, `rf.suspicious_threshold`).

**Binary option:** `_train_binary` collapses real+suspicious into "not-AI" and
keeps AI as the positive class; the suspicious band then becomes a score range
rather than a learned class.

**Evaluation (`evaluate_model.py`):** scores a held-out set, prints per-class
precision/recall/F1, a confusion matrix, and runs a threshold sweep so you can
see how accuracy moves as you slide the cutoffs. It caches extracted features
to `.npz` files (the `features_*.npz` in the repo) so re-runs are fast.

---

## 9. Performance architecture

Getting per-video time down from ~25 s to a few seconds required several
layers of optimization. This is how they fit together.

**1. Frame-count reduction (biggest lever).**
The deep detector is by far the costliest stage (two large ViTs per frame).
Sampling 3 frames instead of 8 cuts ~60% of detector time. Texture uses 4
frames. Since `vit_score = vit_max`, you only need to catch the worst frame.

**2. Parallel execution.**
Texture, semantics, and the deep detector are independent, so they run
concurrently in three worker threads (`asyncio.to_thread` + `asyncio.gather`).
PyTorch and OpenCV both release the GIL during their heavy compute, so this is
real parallelism.

**3. Hardware acceleration.**
The detector auto-selects the best available backend:
`CUDA → Intel XPU (Arc, via IPEX) → Apple MPS → CPU`.
- On Apple Silicon it uses **MPS** (float32 — float16 hits a Metal assertion).
- On Intel Arc laptops it uses **XPU + bfloat16 + `ipex.optimize()`**.
- It runs a **warmup forward pass at startup** so first-kernel compilation
  happens once, not on the first real video, and falls back to CPU if the
  accelerator errors.

**4. Thread management.**
PyTorch CPU threads are capped (`MONET_TORCH_THREADS`, default cpu_count−3) so
the parallel OpenCV work and accelerator host-side dispatch aren't starved.

**5. Inference mode.**
Inference uses `torch.inference_mode()` (slightly faster than `no_grad`).

**Tuning knobs** (all env vars): `MONET_DEVICE`, `MONET_DETECTOR_FRAMES`,
`MONET_XPU_DTYPE`, `MONET_IPEX_OPTIMIZE`, `MONET_MPS_DTYPE`, `MONET_TORCH_THREADS`,
`MONET_BATCH_SIZE`, `MONET_OCR`.

---

## 10. The message protocol (WebSocket JSON)

**Extension → Server:**
```json
{
  "type": "analyze",
  "videoId": "monet-1700000000000",
  "frames": ["data:image/jpeg;base64,...", "...8 total..."],
  "metadata": {
    "title": "...", "channel": "...", "description": "...",
    "hashtags": ["..."], "pageText": "...",
    "youtubeAiDisclosure": "", "url": "https://youtube.com/..."
  }
}
```

**Server → Extension:**
```json
{
  "videoId": "monet-1700000000000",
  "ai_score": 0.73,
  "label": "STRONG AI EVIDENCE",
  "color": "#ef4444",
  "confidence_level": "High",
  "detection_reason": "AI Detector: 81% AI",
  "rf_confidence": 0.66,
  "processing_time_ms": 4830,
  "breakdown": {
    "texture_smoothness": { "score": 0.34, "max": 0.6, "std": 0.1, "desc": "..." },
    "color":              { "score": 0.4,  "max": 0.7, "std": 0.1 },
    "semantic":           { "score": 0.21, "desc": "..." },
    "ai_detector":        { "score": 0.81, "mean": 0.7, "std": 0.08, "desc": "..." },
    "metadata":           { "score": 0.0,  "hits": 0.0, "desc": "..." },
    "digital_penalty":    { "score": 0.0,  "desc": "Natural scene" }
  }
}
```

---

## 11. File map

```
extension/chrome|firefox/
  manifest.json        Manifest V3 — permissions + content script registration
  content.js           All extension logic (detection, capture, UI, WebSocket)
  styles.css           Extra overlay styling

server/
  server.py            FastAPI + WebSocket server; MonetAnalyzer orchestration
  requirements.txt     Python dependencies (+ optional IPEX/Tesseract notes)
  analyzers/
    texture.py             6-sub-analyzer CV/FFT texture analyzer
    semantics.py           5-sub-analyzer semantic analyzer (OCR optional)
    siglip_dinov2_detector.py   SigLIP2+DINOv2 LoRA ensemble + device backends
    random_forest.py       Fusion: RF + manual score + evidence floors
    random_forest.pkl      Trained RF model (loaded at runtime)
  train_model.py       Extract features from training videos, train + tune RF
  evaluate_model.py    Score held-out set, confusion matrix, threshold sweep
  training_data/       Labeled videos (urls_real/_suspicious/_fake) — not in git
  models/siglip_dinov2/  Trained detector weights (~2 GB) — not in git, auto-download
```

---

## 12. Key design decisions & trade-offs

- **Local-only.** Frames never leave the machine; only models download once.
  Trade-off: every user runs the full pipeline on their own hardware.
- **Hybrid fusion (RF + manual + floors).** A pure end-to-end model would be
  sleeker, but the manual score + evidence floors keep behavior predictable and
  let strong single signals (deep detector, metadata) dominate even when the RF
  is unsure. Trade-off: more moving parts to reason about.
- **`vit_max` as the detector signal.** One fake-looking frame is enough
  evidence; using the max is more robust than the mean. Trade-off: a single
  bad frame can tip the score.
- **Deep detector dominates the score (weight 0.64).** It's the most reliable
  signal; the CV heuristics are corroborating evidence. Trade-off: heavily
  dependent on that one model's quality.
- **YouTube disclosure override.** When the platform itself says "made with
  AI", trust it absolutely (force ≥0.95). This is the one ground-truth signal.
- **Digital penalty reduces the score.** Cartoons/flat-digital content can
  trigger "smooth texture" false positives; the penalty prevents digital art
  from being misread as AI video.
- **Frame count vs. speed.** Fewer frames = faster but slightly less robust
  max-score. Made tunable via env vars so each machine can pick its balance.
```
