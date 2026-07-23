import argparse
import asyncio
import json
import os
import sys
import warnings
from collections import Counter, defaultdict
from datetime import datetime

import numpy as np

os.environ["OBJC_DISABLE_INITIALIZE_FORK_SAFETY"] = "YES"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

warnings.filterwarnings("ignore", message=".*SymbolDatabase.GetPrototype.*")
warnings.filterwarnings("ignore", message=".*Mean of empty slice.*")
warnings.filterwarnings("ignore", message=".*invalid value encountered.*")

sys.path.append(os.path.dirname(os.path.abspath(__file__)))


LABELS = {
    0: "Authentic",
    1: "Suspicious",
    2: "AI",
}

FEATURE_NAMES = [
    'texture_mean', 'texture_max', 'texture_std',
    'color_mean', 'color_max', 'color_std',
    'digital_penalty',
    'semantic',
    'vit_mean', 'vit_max', 'vit_std',
    'metadata_score',
    'metadata_hits',
    'temporal_diff_mean', 'temporal_diff_std',
    'brightness_flicker', 'saturation_flicker',
]

# Cache schema version. Bumped when frame sampling / resolution changes so
# stale cached features are recomputed instead of reused.
CACHE_VERSION = "8frame_480p_temporal4"

# Must match train_model.FEATURE_CACHE_VERSION. This is the .npz feature cache
# that train_model.py writes; evaluate_model.py prefers it so evaluation never
# has to re-extract features from the source videos.
FEATURE_CACHE_VERSION = "temporal4_8frame"


def score_to_class(score, ai_threshold=0.65, suspicious_threshold=0.46):
    if score >= ai_threshold:
        return 2
    if score >= suspicious_threshold:
        return 1
    return 0


def load_sources(base_dir, limit_per_class=None):
    """Load evaluation examples from the local 480p video folders."""
    sources = [
        ("urls_real", 0, "real"),
        ("urls_suspicious", 1, "suspicious"),
        ("urls_fake", 2, "ai"),
    ]
    examples = []
    for folder_name, label, bucket in sources:
        folder = os.path.join(base_dir, "training_data", folder_name)
        video_files = _iter_video_files(folder)
        if limit_per_class:
            video_files = video_files[:limit_per_class]
        for path in video_files:
            video_id = os.path.splitext(os.path.basename(path))[0]
            examples.append({
                "url": _url_from_video_id(video_id),
                "video_path": path,
                "label": label,
                "bucket": bucket,
                "source": folder_name + "/",
            })
    return examples


def load_feature_cache(base_dir, cache_tag="full"):
    """Load the .npz feature cache written by train_model.py.

    Returns (X, y, cache_path). X/y are None if no usable cache exists
    (missing file, version mismatch, or feature-count mismatch).
    """
    cache_path = os.path.join(
        base_dir, "training_data", f"features_v{FEATURE_CACHE_VERSION}_{cache_tag}.npz"
    )
    if not os.path.exists(cache_path):
        return None, None, cache_path
    try:
        cached = np.load(cache_path, allow_pickle=True)
        version = str(cached["version"].item())
        n_features = int(cached["X"].shape[1])
        if version != FEATURE_CACHE_VERSION or n_features != len(FEATURE_NAMES):
            return None, None, cache_path
        return cached["X"], cached["y"], cache_path
    except Exception:
        return None, None, cache_path


def _detect_digital_content(img):
    pixels = img.reshape(-1, 3)
    unique_colors = len(np.unique(pixels, axis=0))
    total_pixels = img.shape[0] * img.shape[1]
    diversity_ratio = unique_colors / total_pixels

    if diversity_ratio < 0.05:
        return 0.3
    if diversity_ratio < 0.15:
        return 0.1
    return 0.0


def _analyze_color(frames):
    import cv2

    scores = []
    for frame in frames:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        saturation = hsv[:, :, 1].mean()
        if saturation > 150:
            scores.append(0.7)
        elif saturation > 120:
            scores.append(0.4)
        else:
            scores.append(0.1)
    return scores


def _compute_temporal_features(frames):
    """Frame-to-frame consistency features (must match train_model.py / server.py)."""
    import cv2

    if len(frames) < 2:
        return 0.0, 0.0, 0.0, 0.0
    diffs, brights, sats = [], [], []
    for i, f in enumerate(frames):
        gray = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
        brights.append(float(np.mean(gray)) / 255.0)
        hsv = cv2.cvtColor(f, cv2.COLOR_BGR2HSV)
        sats.append(float(np.mean(hsv[:, :, 1])) / 255.0)
        if i > 0:
            d = cv2.absdiff(f, frames[i - 1])
            diffs.append(float(np.mean(d)) / 255.0)
    return (
        float(np.mean(diffs)),
        float(np.std(diffs)),
        float(np.std(brights)),
        float(np.std(sats)),
    )


def _analyze_metadata(metadata):
    if not isinstance(metadata, dict):
        return 0.0, 0.0

    disclosure = str(metadata.get("youtubeAiDisclosure", "") or "").lower()
    import re
    if re.search(r'\bai\b|synthetic|altered', disclosure):
        return 1.0, 1.0

    disclosure_phrases = [
        "altered or synthetic content",
        "synthetic content",
        "ai-generated",
        "ai generated",
        "created with ai",
        "made with ai",
        "generated with ai"
    ]
    if any(phrase in disclosure for phrase in disclosure_phrases):
        return 1.0, 1.0

    fields = [
        metadata.get("title", ""),
        metadata.get("description", ""),
        metadata.get("channel", ""),
        " ".join(metadata.get("tags", []) or []),
        " ".join(metadata.get("categories", []) or []),
        metadata.get("youtubeAiDisclosure", "")
    ]
    text = " ".join(str(f) for f in fields if f).lower()
    if not text.strip():
        return 0.0, 0.0

    strong_terms = [
        "ai generated", "generated by ai", "made with ai", "text to video",
        "image to video", "ai video", "ai short", "synthetic video",
        "altered or synthetic content", "created with ai", "generated with ai"
    ]
    tool_terms = [
        "runway", "sora", "pika", "kling", "hailuo", "luma", "dream machine",
        "veo", "midjourney", "stable diffusion", "sdxl", "gen-3", "gen 3",
        "prompt", "comfyui"
    ]
    weak_terms = ["generated", "artificial", "deepfake", "cgi", "vfx"]

    hits = []
    score = 0.0
    for term in strong_terms:
        if term in text:
            hits.append(term)
            score += 0.35
    for term in tool_terms:
        if term in text:
            hits.append(term)
            score += 0.18
    for term in weak_terms:
        if term in text:
            hits.append(term)
            score += 0.08

    hit_count = min(len(set(hits)), 6)
    return min(score, 0.85), hit_count / 6


VIDEO_EXTENSIONS = {".mp4", ".webm", ".mkv", ".avi", ".mov", ".m4v", ".flv"}


def _is_video_file(path):
    return os.path.isfile(path) and os.path.splitext(path)[1].lower() in VIDEO_EXTENSIONS


def _iter_video_files(folder):
    if not os.path.isdir(folder):
        return []
    entries = sorted(os.listdir(folder))
    return [os.path.join(folder, name) for name in entries if _is_video_file(os.path.join(folder, name))]


def _url_from_video_id(video_id):
    return f"https://www.youtube.com/watch?v={video_id}"


def _sample_frames(video_path, num_sample=8):
    """Sample up to `num_sample` frames evenly across the whole video (matches train_model.py)."""
    import cv2

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    try:
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frames = []
        if total_frames > 0:
            count = min(num_sample, total_frames)
            indices = np.linspace(0, total_frames - 1, count, dtype=int)
            indices = list(dict.fromkeys(int(i) for i in indices))
            for idx in indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ok, frame = cap.read()
                if ok:
                    frames.append(frame)
        return frames
    finally:
        cap.release()


class FeatureExtractor:
    def __init__(self):
        from analyzers.texture import TextureAnalyzer
        from analyzers.semantics import SemanticAnalyzer
        from analyzers.random_forest import RandomForestClassifier
        from analyzers.siglip_dinov2_detector import SigLIPDinoV2Detector

        print("Initializing analyzers...")
        self.texture = TextureAnalyzer()
        self.semantics = SemanticAnalyzer()
        self.vit = SigLIPDinoV2Detector()
        self.rf = RandomForestClassifier()
        self.feature_names = RandomForestClassifier.FEATURE_NAMES

    async def extract(self, example):
        metadata = {}
        frames = _sample_frames(example["video_path"])
        if not frames:
            raise RuntimeError("No frames extracted")

        texture_scores = [self.texture.analyze(frame)[0] for frame in frames]
        color_scores = _analyze_color(frames)
        semantic_score, _ = await self.semantics.analyze(frames)
        vit_mean, vit_max, vit_std, _ = self.vit.analyze_frame_stats(frames)
        metadata_score, metadata_hits = _analyze_metadata(metadata)
        digital_penalty = _detect_digital_content(frames[0])

        temporal_diff_mean, temporal_diff_std, brightness_flicker, saturation_flicker = _compute_temporal_features(frames)

        features = self.rf.extract_features(
            np.mean(texture_scores), np.max(texture_scores), np.std(texture_scores),
            np.mean(color_scores), np.max(color_scores), np.std(color_scores),
            digital_penalty,
            semantic_score,
            vit_max,
            metadata_score=metadata_score,
            metadata_hits=metadata_hits,
            vit_mean=vit_mean,
            vit_max=vit_max,
            vit_std=vit_std,
            temporal_diff_mean=temporal_diff_mean,
            temporal_diff_std=temporal_diff_std,
            brightness_flicker=brightness_flicker,
            saturation_flicker=saturation_flicker,
        )[0]

        return {
            "features": features.tolist(),
            "metadata": metadata,
            "signals": dict(zip(self.feature_names, features.tolist())),
        }


def load_cache(path):
    if not path or not os.path.exists(path):
        return {}
    cache = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get("url"):
                cache[row["url"]] = row
    return cache


def append_cache(path, row):
    if not path:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def class_metrics(rows):
    by_label = {}
    for label in LABELS:
        tp = sum(1 for r in rows if r["actual"] == label and r["predicted"] == label)
        fp = sum(1 for r in rows if r["actual"] != label and r["predicted"] == label)
        fn = sum(1 for r in rows if r["actual"] == label and r["predicted"] != label)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        by_label[label] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": sum(1 for r in rows if r["actual"] == label),
        }
    accuracy = sum(1 for r in rows if r["actual"] == r["predicted"]) / len(rows) if rows else 0.0
    return accuracy, by_label


def threshold_sweep(scored_rows):
    candidates = []
    for suspicious_threshold in [0.38, 0.42, 0.44, 0.46, 0.48, 0.50, 0.54]:
        for ai_threshold in [0.55, 0.60, 0.63, 0.65, 0.67, 0.70]:
            if suspicious_threshold >= ai_threshold:
                continue
            rows = []
            for row in scored_rows:
                updated = dict(row)
                updated["predicted"] = score_to_class(row["score"], ai_threshold, suspicious_threshold)
                rows.append(updated)
            accuracy, metrics = class_metrics(rows)
            macro_f1 = sum(m["f1"] for m in metrics.values()) / len(metrics)
            ai_recall = metrics[2]["recall"]
            real_precision = metrics[0]["precision"]
            candidates.append({
                "suspicious_threshold": suspicious_threshold,
                "ai_threshold": ai_threshold,
                "accuracy": accuracy,
                "macro_f1": macro_f1,
                "ai_recall": ai_recall,
                "real_precision": real_precision,
            })
    return sorted(candidates, key=lambda r: (r["macro_f1"], r["ai_recall"], r["accuracy"]), reverse=True)


def ai_recall_thresholds(sweep, min_real_precision=0.70):
    filtered = [row for row in sweep if row["real_precision"] >= min_real_precision]
    if not filtered:
        filtered = sweep
    return sorted(
        filtered,
        key=lambda r: (r["ai_recall"], r["macro_f1"], r["accuracy"]),
        reverse=True,
    )


def write_report(path, rows, sweep, feature_importance):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    accuracy, metrics = class_metrics(rows)
    errors = [r for r in rows if r["actual"] != r["predicted"]]
    false_positives = [r for r in errors if r["actual"] == 0 and r["predicted"] in {1, 2}]
    false_negatives = [r for r in errors if r["actual"] == 2 and r["predicted"] != 2]
    bucket_counts = Counter(r["bucket"] for r in errors)

    lines = [
        "# Monet Classification Evaluation",
        "",
        f"Generated: {datetime.now().isoformat(timespec='seconds')}",
        f"Examples evaluated: {len(rows)}",
        f"Accuracy: {accuracy:.1%}",
        f"Errors: {len(errors)}",
        "",
        "## Class Metrics",
        "",
        "| Class | Precision | Recall | F1 | Support |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]

    for label, name in LABELS.items():
        metric = metrics[label]
        lines.append(
            f"| {name} | {metric['precision']:.1%} | {metric['recall']:.1%} | "
            f"{metric['f1']:.1%} | {metric['support']} |"
        )

    lines.extend([
        "",
        "## Best Thresholds",
        "",
        "| Suspicious | AI | Accuracy | Macro F1 | AI Recall | Real Precision |",
        "| ---: | ---: | ---: | ---: | ---: | ---: |",
    ])
    for row in sweep[:8]:
        lines.append(
            f"| {row['suspicious_threshold']:.2f} | {row['ai_threshold']:.2f} | "
            f"{row['accuracy']:.1%} | {row['macro_f1']:.1%} | "
            f"{row['ai_recall']:.1%} | {row['real_precision']:.1%} |"
        )

    lines.extend([
        "",
        "## Best Thresholds For Catching AI",
        "",
        "These favor AI recall while keeping real-video precision at roughly 70% or better when possible.",
        "",
        "| Suspicious | AI | Accuracy | Macro F1 | AI Recall | Real Precision |",
        "| ---: | ---: | ---: | ---: | ---: | ---: |",
    ])
    for row in ai_recall_thresholds(sweep)[:8]:
        lines.append(
            f"| {row['suspicious_threshold']:.2f} | {row['ai_threshold']:.2f} | "
            f"{row['accuracy']:.1%} | {row['macro_f1']:.1%} | "
            f"{row['ai_recall']:.1%} | {row['real_precision']:.1%} |"
        )

    lines.extend(["", "## Error Buckets", ""])
    if bucket_counts:
        for bucket, count in bucket_counts.most_common():
            lines.append(f"- {bucket}: {count}")
    else:
        lines.append("- No errors found in this run.")

    lines.extend(["", "## Feature Importance", ""])
    if feature_importance:
        for name, importance in sorted(feature_importance.items(), key=lambda item: item[1], reverse=True):
            lines.append(f"- {name}: {importance:.3f}")
    else:
        lines.append("- Feature importance unavailable until the model is trained with matching features.")

    lines.extend(["", "## False Positives", ""])
    if false_positives:
        for row in false_positives[:25]:
            lines.append(
                f"- {row['url']} | score {row['score']:.3f} | predicted {LABELS[row['predicted']]}"
            )
    else:
        lines.append("- None.")

    lines.extend(["", "## False Negatives", ""])
    if false_negatives:
        for row in false_negatives[:25]:
            lines.append(
                f"- {row['url']} | score {row['score']:.3f} | predicted {LABELS[row['predicted']]}"
            )
    else:
        lines.append("- None.")

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


# Binary evaluation: real + suspicious are merged into "not-AI", so the model
# is scored on the AI-vs-not-AI task it was trained for. The middle display band
# is treated as "uncertain", never as an error.

def _actual_binary(actual):
    """Map 3-class ground truth (0=real, 1=suspicious, 2=AI) -> binary (0=not-AI, 1=AI)."""
    return 1 if actual == 2 else 0


def binary_metrics(rows, threshold):
    tp = fp = tn = fn = 0
    for r in rows:
        actual = _actual_binary(r["actual"])
        predicted = 1 if r["score"] >= threshold else 0
        if predicted and actual:
            tp += 1
        elif predicted and not actual:
            fp += 1
        elif not predicted and not actual:
            tn += 1
        else:
            fn += 1
    total = tp + fp + tn + fn
    accuracy = (tp + tn) / total if total else 0.0
    ai_prec = tp / (tp + fp) if (tp + fp) else 0.0
    ai_rec = tp / (tp + fn) if (tp + fn) else 0.0
    ai_f1 = 2 * ai_prec * ai_rec / (ai_prec + ai_rec) if (ai_prec + ai_rec) else 0.0
    notai_prec = tn / (tn + fn) if (tn + fn) else 0.0
    notai_rec = tn / (tn + fp) if (tn + fp) else 0.0
    return {
        "threshold": threshold,
        "accuracy": accuracy,
        "ai_precision": ai_prec,
        "ai_recall": ai_rec,
        "ai_f1": ai_f1,
        "notai_precision": notai_prec,
        "notai_recall": notai_rec,
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "support_ai": tp + fn,
        "support_notai": tn + fp,
    }


def binary_threshold_sweep(rows):
    candidates = []
    for t in [0.40, 0.45, 0.50, 0.52, 0.55, 0.58, 0.60, 0.63, 0.65, 0.68, 0.70, 0.75, 0.80, 0.84]:
        candidates.append(binary_metrics(rows, t))
    return candidates


def write_binary_report(path, rows, sweep, feature_importance, deployed_threshold, uncertain_lo, uncertain_hi):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    deployed = binary_metrics(rows, deployed_threshold)

    false_positives = [r for r in rows if _actual_binary(r["actual"]) == 0 and r["score"] >= deployed_threshold]
    false_negatives = [r for r in rows if _actual_binary(r["actual"]) == 1 and r["score"] < deployed_threshold]
    uncertain = [r for r in rows if uncertain_lo <= r["score"] < uncertain_hi]

    lines = [
        "# Monet Classification Evaluation (BINARY: AI vs not-AI)",
        "",
        f"Generated: {datetime.now().isoformat(timespec='seconds')}",
        "Real + Suspicious are merged into 'not-AI'. The score band "
        f"[{uncertain_lo:.2f}, {uncertain_hi:.2f}) is the display 'Suspicious' zone "
        "(reported as uncertain, never as an error).",
        f"Examples evaluated: {len(rows)} (not-AI={deployed['support_notai']}, AI={deployed['support_ai']})",
        f"AI decision threshold: score >= {deployed_threshold}",
        "",
        "## Deployed-threshold metrics",
        "",
        f"- Accuracy: {deployed['accuracy']:.1%}",
        f"- AI precision: {deployed['ai_precision']:.1%}",
        f"- AI recall: {deployed['ai_recall']:.1%}",
        f"- AI F1: {deployed['ai_f1']:.1%}",
        f"- not-AI precision: {deployed['notai_precision']:.1%}",
        f"- not-AI recall: {deployed['notai_recall']:.1%}",
        f"- Uncertain band [{uncertain_lo:.2f}, {uncertain_hi:.2f}): {len(uncertain)} examples",
        "",
        "## Confusion matrix (rows=true, cols=pred)",
        "```",
        "           not-AI    AI",
        f"  not-AI  {deployed['tn']:>6}  {deployed['fp']:>4}",
        f"  AI      {deployed['fn']:>6}  {deployed['tp']:>4}",
        "```",
        "",
        "## Threshold sweep",
        "",
        "| Threshold | Accuracy | AI Precision | AI Recall | AI F1 | not-AI Recall |",
        "| ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in sweep:
        lines.append(
            f"| {row['threshold']:.2f} | {row['accuracy']:.1%} | {row['ai_precision']:.1%} | "
            f"{row['ai_recall']:.1%} | {row['ai_f1']:.1%} | {row['notai_recall']:.1%} |"
        )

    lines.extend(["", "## Feature Importance", ""])
    if feature_importance:
        for name, importance in sorted(feature_importance.items(), key=lambda item: item[1], reverse=True):
            lines.append(f"- {name}: {importance:.3f}")
    else:
        lines.append("- Feature importance unavailable until the model is trained with matching features.")

    lines.extend(["", "## False Positives (not-AI flagged as AI)", ""])
    if false_positives:
        for row in false_positives[:25]:
            lines.append(f"- {row['url']} | score {row['score']:.3f}")
    else:
        lines.append("- None.")

    lines.extend(["", "## False Negatives (AI missed)", ""])
    if false_negatives:
        for row in false_negatives[:25]:
            lines.append(f"- {row['url']} | score {row['score']:.3f}")
    else:
        lines.append("- None.")

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


async def main():
    parser = argparse.ArgumentParser(description="Evaluate Monet on local labeled videos.")
    parser.add_argument("--limit-per-class", type=int, default=None, help="Quick-run limit for each class.")
    parser.add_argument("--cache", default=None, help="Feature cache JSONL path (only used with --force-extract).")
    parser.add_argument("--report", default=None, help="Markdown report output path.")
    parser.add_argument("--force-extract", action="store_true",
                        help="Ignore the .npz feature cache and re-extract features from the source videos.")
    parser.add_argument("--cache-tag", default=None,
                        help="Feature-cache tag to load (default 'full', or 'limit<N>' matching train_model --limit).")
    parser.add_argument("--binary", action="store_true",
                        help="Evaluate as binary AI-vs-not-AI (merge real+suspicious into not-AI).")
    args = parser.parse_args()

    from analyzers.random_forest import RandomForestClassifier

    base_dir = os.path.dirname(os.path.abspath(__file__))
    report_path = args.report or os.path.join(base_dir, "training_data", "evaluation_report.md")
    model_path = os.path.join(base_dir, "analyzers", "random_forest.pkl")

    model = RandomForestClassifier(model_path=model_path)

    cache_tag = args.cache_tag or (f"limit{args.limit_per_class}" if args.limit_per_class else "full")
    bucket_for_label = {0: "real", 1: "suspicious", 2: "ai"}

    rows = []

    # ---- Prefer the .npz feature cache produced by train_model.py ----
    if not args.force_extract:
        X, y, npz_path = load_feature_cache(base_dir, cache_tag)
        if X is not None:
            print(f"Loaded feature cache ({len(y)} samples, {X.shape[1]} features) from:")
            print(f"  {npz_path}")
            print("(Use --force-extract to re-extract from the source videos instead.)\n")
            for i in range(len(y)):
                features = np.array([X[i]])
                score, _, _, confidence = model.predict(features)
                rows.append({
                    "url": f"cache/{i}",
                    "actual": int(y[i]),
                    "predicted": score_to_class(score),
                    "score": float(score),
                    "confidence": float(confidence),
                    "bucket": bucket_for_label.get(int(y[i]), "unknown"),
                    "source": "feature_cache.npz",
                    "signals": {},
                })
        else:
            print(f"No usable .npz feature cache for tag '{cache_tag}'; falling back to video extraction.\n")

    # ---- Fallback: extract features from the source videos ----
    if not rows:
        cache_path = args.cache or os.path.join(base_dir, "training_data", "feature_cache.jsonl")
        examples = load_sources(base_dir, args.limit_per_class)
        if not examples:
            print("No examples found.")
            return

        cache = load_cache(cache_path)
        extractor = FeatureExtractor()

        for idx, example in enumerate(examples, start=1):
            url = example["url"]
            print(f"[{idx}/{len(examples)}] {url}")
            try:
                cached = cache.get(url)
                if cached and cached.get("v") == CACHE_VERSION and len(cached.get("features", [])) == len(FEATURE_NAMES):
                    feature_row = cached
                else:
                    feature_row = await extractor.extract(example)
                    feature_row.update({"url": url, "v": CACHE_VERSION})
                    append_cache(cache_path, feature_row)

                features = np.array([feature_row["features"]])
                score, _, _, confidence = model.predict(features)
                predicted = score_to_class(score)

                rows.append({
                    "url": url,
                    "actual": example["label"],
                    "predicted": predicted,
                    "score": float(score),
                    "confidence": float(confidence),
                    "bucket": example["bucket"],
                    "source": example["source"],
                    "signals": feature_row.get("signals", {}),
                })
            except Exception as e:
                print(f"  skipped: {e}")

    if not rows:
        print("No examples could be evaluated.")
        return

    if args.binary:
        deployed_threshold = model.ai_threshold
        sweep = binary_threshold_sweep(rows)
        write_binary_report(
            report_path, rows, sweep, model.get_feature_importance(),
            deployed_threshold, model.suspicious_threshold, model.ai_threshold,
        )
        deployed = binary_metrics(rows, deployed_threshold)
        print(f"\nBinary evaluation (AI vs not-AI) — threshold {deployed_threshold}")
        print(f"  Examples     : {len(rows)} (not-AI={deployed['support_notai']}, AI={deployed['support_ai']})")
        print(f"  Accuracy     : {deployed['accuracy']:.1%}")
        print(f"  AI precision : {deployed['ai_precision']:.1%}")
        print(f"  AI recall    : {deployed['ai_recall']:.1%}")
        print(f"  AI F1        : {deployed['ai_f1']:.1%}")
        print(f"Report saved to: {report_path}")
    else:
        sweep = threshold_sweep(rows)
        write_report(report_path, rows, sweep, model.get_feature_importance())

        accuracy, metrics = class_metrics(rows)
        print(f"\nEvaluated {len(rows)} examples")
        print(f"Accuracy: {accuracy:.1%}")
        print(f"AI recall: {metrics[2]['recall']:.1%}")
        print(f"Report saved to: {report_path}")


if __name__ == "__main__":
    asyncio.run(main())
