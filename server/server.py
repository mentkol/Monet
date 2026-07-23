import os, sys, warnings

os.environ["OBJC_DISABLE_INITIALIZE_FORK_SAFETY"] = "YES"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# Cap PyTorch CPU threads so parallel analyzer threads (texture/semantics via
# OpenCV) have room on the machine. Tunable via MONET_TORCH_THREADS.
try:
    import torch as _torch
    _default_torch_threads = max(1, (os.cpu_count() or 8) - 3)
    _torch.set_num_threads(int(os.environ.get("MONET_TORCH_THREADS", _default_torch_threads)))
except Exception:
    pass

warnings.filterwarnings("ignore", message=".*SymbolDatabase.GetPrototype.*")
warnings.filterwarnings("ignore", message=".*Mean of empty slice.*")
warnings.filterwarnings("ignore", message=".*invalid value encountered.*")
warnings.filterwarnings("ignore", message=".*use_fast.*")
warnings.filterwarnings("ignore", message=".*slow image processor.*")
warnings.filterwarnings("ignore", message=".*Non-default generation parameters.*")

try:
    null_fd = os.open(os.devnull, os.O_RDWR)
    save_stderr = os.dup(2)
    os.dup2(null_fd, 2)
except Exception:
    pass

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import asyncio
import json
import numpy as np
from PIL import Image
import io
import base64
import cv2
import time

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from analyzers.texture import TextureAnalyzer
from analyzers.semantics import SemanticAnalyzer
from analyzers.random_forest import RandomForestClassifier
from analyzers.siglip_dinov2_detector import SigLIPDinoV2Detector

try:
    os.dup2(save_stderr, 2)
    os.close(null_fd)
    os.close(save_stderr)
except Exception:
    pass

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class ImpossibleSceneClassifier:
    def __init__(self):
        pass
    
    def check_text_description(self, img):
        try:
            import pytesseract
            text = pytesseract.image_to_string(img).lower()
            
            impossible_phrases = [
                'cat driving', 'dog driving', 'animal driving',
                'buffalo walking', 'cow standing', 'dog walking on two',
                'cat pilot', 'monkey typing'
            ]
            
            for phrase in impossible_phrases:
                if phrase in text:
                    return 0.9, f"Text indicates: {phrase}"
            
            return 0.0, "No suspicious text"
        except:
            return 0.0, "OCR unavailable"
    
    def analyze(self, img):
        return self.check_text_description(img)

def type_text(text, speed=0.03, prefix="", suffix="", end="\n"):
    sys.stdout.write(prefix)
    for char in text:
        sys.stdout.write(char)
        sys.stdout.flush()
        time.sleep(speed)
    sys.stdout.write(suffix + end)
    sys.stdout.flush()

SPINNER_CODE = """
import sys, time
name = sys.argv[1]
indent = sys.argv[2] if len(sys.argv) > 2 else "  "
style = sys.argv[3] if len(sys.argv) > 3 else "\\033[96m"
frames = ['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏']
i = 0
try:
    while True:
        sys.stdout.write(f"\\r{indent}{style}{frames[i]}\\033[0m {name}")
        sys.stdout.flush()
        i = (i + 1) % len(frames)
        time.sleep(0.08)
except KeyboardInterrupt:
    pass
"""

class MonetAnalyzer:
    def __init__(self):
        print()
        type_text("MONET AI Detection Tool", speed=0.04, prefix="\033[1m\033[95m", suffix="\033[0m")
        time.sleep(0.2)
        
        type_text("Initializing core systems...", speed=0.04, prefix="\033[2m", end="")
        for _ in range(3):
            for dots in ["   ", ".  ", ".. ", "..."]:
                sys.stdout.write(f"\b\b\b{dots}")
                sys.stdout.flush()
                time.sleep(0.15)
        sys.stdout.write("\033[0m\n\n")
        
        type_text("System Status:", speed=0.05, prefix="\033[1m", suffix="\033[0m")
        time.sleep(0.2)
        
        _real_stdout = sys.stdout
        _real_stderr = sys.stderr
        
        def load_module(name, init_func, check_func=None):
            import subprocess
            p = subprocess.Popen([sys.executable, "-c", SPINNER_CODE, name])

            sys.stdout = open(os.devnull, 'w')
            sys.stderr = open(os.devnull, 'w')
            
            try:
                null_fd = os.open(os.devnull, os.O_RDWR)
                save_err = os.dup(2)
                os.dup2(null_fd, 2)
            except Exception:
                pass

            try:
                time.sleep(0.35)
                module = init_func()
                ok = check_func(module) if check_func else True
            except Exception:
                module = None
                ok = False
            finally:
                p.terminate()
                p.wait()
                
                try:
                    os.dup2(save_err, 2)
                    os.close(null_fd)
                    os.close(save_err)
                except Exception:
                    pass
                
                sys.stdout.close()
                sys.stderr.close()
                sys.stdout = _real_stdout
                sys.stderr = _real_stderr
                
            status = "\033[92m✓\033[0m" if ok else "\033[91m✗\033[0m"
            print(f"\r  {status} {name}          ", flush=True)
            return module

        self.texture = load_module("Texture", lambda: TextureAnalyzer())
        self.semantics = load_module("Semantics", lambda: SemanticAnalyzer())
        
        self.impossible_classifier = ImpossibleSceneClassifier()
        self.ai_detector = load_module("AI Detector", lambda: SigLIPDinoV2Detector(), lambda m: m.loaded)
        
        model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "analyzers", "random_forest.pkl")
        self.rf_classifier = load_module("RF Model", lambda: RandomForestClassifier(model_path=model_path), lambda m: m.is_trained)
            
        print()
        import subprocess
        self.waiting_spinner = subprocess.Popen([
            sys.executable, "-c", SPINNER_CODE, 
            "\033[1m\033[96mReady & Waiting for extension\033[0m", 
            "",
            "\033[1m\033[96m"
        ])
    
    def _detect_digital_content(self, img):
        pixels = img.reshape(-1, 3)
        unique_colors = len(np.unique(pixels, axis=0))
        total_pixels = img.shape[0] * img.shape[1]
        diversity_ratio = unique_colors / total_pixels
        
        if diversity_ratio < 0.05:
            return 0.3, "Digital content detected"
        elif diversity_ratio < 0.15:
            return 0.1, "Limited color palette"
        return 0.0, "Natural scene"
    
    def _analyze_color(self, frames):
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

    def _compute_temporal_features(self, frames):
        """Frame-to-frame consistency features (must match train_model.py / evaluate_model.py)."""
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

    def _analyze_metadata(self, metadata):
        if not isinstance(metadata, dict):
            return 0.0, 0.0, "No metadata"

        if self._youtube_disclosure_says_ai(metadata):
            return 1.0, 1.0, "YouTube AI disclosure detected"

        fields = [
            metadata.get("title", ""),
            metadata.get("description", ""),
            metadata.get("channel", ""),
            " ".join(metadata.get("hashtags", []) or []),
            metadata.get("youtubeAiDisclosure", ""),
            metadata.get("pageText", "")
        ]
        text = " ".join(str(f) for f in fields if f).lower()
        if not text.strip():
            return 0.0, 0.0, "No metadata"

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
        normalized_hits = hit_count / 6
        score = min(score, 0.85)

        if hit_count:
            return score, normalized_hits, f"Metadata hints: {', '.join(list(dict.fromkeys(hits))[:3])}"
        return 0.0, 0.0, "No AI metadata hints"

    @staticmethod
    def _youtube_disclosure_says_ai(metadata):
        if not isinstance(metadata, dict):
            return False

        disclosure = str(metadata.get("youtubeAiDisclosure", "") or "").lower()
        if not disclosure.strip():
            return False

        import re
        if re.search(r'\bai\b|synthetic|altered', disclosure):
            return True

        phrases = [
            "altered or synthetic content",
            "synthetic content",
            "ai-generated",
            "ai generated",
            "created with ai",
            "made with ai",
            "generated with ai"
        ]
        return any(phrase in disclosure for phrase in phrases)
    
    @staticmethod
    def _safe(val, default=0.0):
        import math
        v = float(val)
        return default if math.isnan(v) or math.isinf(v) else v
    
    async def analyze_video(self, frames, metadata=None):
        if len(frames) == 0:
            return {"error": "No frames received"}
        
        start = time.time()
        texture_n = min(4, len(frames))
        texture_indices = np.linspace(0, len(frames) - 1, texture_n, dtype=int)
        texture_indices = list(dict.fromkeys(int(i) for i in texture_indices))
        texture_frames = [frames[i] for i in texture_indices]

        async def _texture_job():
            def _work():
                texture_results = [self.texture.analyze(frame) for frame in texture_frames]
                texture_scores = [score for score, _ in texture_results]
                return (
                    float(np.mean(texture_scores)),
                    float(np.max(texture_scores)),
                    float(np.std(texture_scores)),
                    [desc for _, desc in texture_results],
                )
            return await asyncio.to_thread(_work)

        async def _semantics_job():
            return await asyncio.to_thread(lambda: self.semantics.analyze_sync(frames))

        async def _detector_job():
            return await asyncio.to_thread(lambda: self.ai_detector.analyze_frame_stats(frames))

        (texture_mean, texture_max, texture_std, texture_descs), \
            (semantic_score, semantic_desc), \
            (vit_mean, vit_max, vit_std, vit_desc) = await asyncio.gather(
                _texture_job(), _semantics_job(), _detector_job()
            )
        vit_score = vit_max

        color_scores = self._analyze_color(frames)
        color_mean = np.mean(color_scores)
        color_max = np.max(color_scores)
        color_std = np.std(color_scores)
        
        digital_penalty, digital_desc = self._detect_digital_content(frames[0])

        metadata_score, metadata_hits, metadata_desc = self._analyze_metadata(metadata)

        (temporal_diff_mean, temporal_diff_std,
         brightness_flicker, saturation_flicker) = self._compute_temporal_features(texture_frames)

        features = self.rf_classifier.extract_features(
            texture_mean, texture_max, texture_std,
            color_mean, color_max, color_std,
            digital_penalty,
            semantic_score,
            vit_score,
            metadata_score=metadata_score,
            metadata_hits=metadata_hits,
            vit_mean=vit_mean,
            vit_max=vit_max,
            vit_std=vit_std,
            temporal_diff_mean=temporal_diff_mean,
            temporal_diff_std=temporal_diff_std,
            brightness_flicker=brightness_flicker,
            saturation_flicker=saturation_flicker,
        )
        
        final_score, label, color, rf_confidence = self.rf_classifier.predict(features)

        youtube_disclosure_override = self._youtube_disclosure_says_ai(metadata)
        if youtube_disclosure_override:
            final_score = max(final_score, 0.95)
            label = "STRONG AI EVIDENCE"
            color = "#ef4444"
        
        rf_says_ai = final_score >= 0.50
        detector_says_ai = vit_score >= 0.50
        rf_says_real = final_score < 0.28
        detector_says_real = vit_score < 0.20
        
        if youtube_disclosure_override:
            confidence_level = "High"
        elif (rf_says_ai and detector_says_ai) or (rf_says_real and detector_says_real):
            confidence_level = "High"
        elif rf_says_ai != detector_says_ai and abs(final_score - 0.5) < 0.15:
            confidence_level = "Low"
        else:
            confidence_level = "Medium"
        
        score_reasons = [
            (texture_mean, texture_descs[-1], "Texture"),
            (semantic_score, semantic_desc, "Semantics"),
            (vit_score, vit_desc, "AI Detector"),
            (metadata_score, metadata_desc, "Metadata")
        ]
        
        highest = max(score_reasons, key=lambda x: x[0])
        
        if youtube_disclosure_override:
            detection_reason = "Metadata: YouTube AI disclosure detected"
        elif highest[0] >= 0.5:
            detection_reason = f"{highest[2]}: {highest[1]}"
        elif final_score >= 0.42:
            detection_reason = "Multiple weak signals"
        else:
            detection_reason = "No significant anomalies"
            
        elapsed = int((time.time() - start) * 1000)
        
        if final_score >= 0.65:
            term_color = '\033[91m'
        elif final_score >= 0.46:
            term_color = '\033[93m'
        else:
            term_color = '\033[92m'

        dim = '\033[2m'
        reset = '\033[0m'
        bold = '\033[1m'
        
        print(f"\n{dim}[{elapsed}ms]{reset} {bold}{term_color}{label}{reset} (Score: {final_score:.2f})")
        print(f"  ├─ Confidence : {confidence_level}")
        print(f"  └─ Reason     : {detection_reason}")
        
        s = self._safe
        
        return {
            "ai_score": round(s(final_score), 3),
            "label": label,
            "color": color,
            "processing_time_ms": elapsed,
            "detection_reason": detection_reason,
            "confidence_level": confidence_level,
            "rf_confidence": round(s(rf_confidence), 3),
            "breakdown": {
                "texture_smoothness": {
                    "score": round(s(texture_mean), 2),
                    "max": round(s(texture_max), 2),
                    "std": round(s(texture_std), 2),
                    "desc": texture_descs[-1]
                },
                "color": {
                    "score": round(s(color_mean), 2),
                    "max": round(s(color_max), 2),
                    "std": round(s(color_std), 2)
                },
                "semantic": {
                    "score": round(s(semantic_score), 2),
                    "desc": semantic_desc
                },
                "ai_detector": {
                    "score": round(s(vit_score), 2),
                    "mean": round(s(vit_mean), 2),
                    "std": round(s(vit_std), 2),
                    "desc": vit_desc
                },
                "metadata": {
                    "score": round(s(metadata_score), 2),
                    "hits": round(s(metadata_hits), 2),
                    "desc": metadata_desc
                },
                "digital_penalty": {
                    "score": round(s(digital_penalty), 2),
                    "desc": digital_desc
                }
            }
        }

analyzer = MonetAnalyzer()

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    import asyncio
    await websocket.accept()
    
    if hasattr(analyzer, "waiting_spinner") and analyzer.waiting_spinner:
        analyzer.waiting_spinner.terminate()
        analyzer.waiting_spinner.wait()
        sys.stdout.write("\r\033[1m\033[92m")
        analyzer.waiting_spinner = None
    else:
        sys.stdout.write("\033[1m\033[92m")
        
    sys.stdout.flush()
    
    for char in "Extension connected                          ":
        sys.stdout.write(char)
        sys.stdout.flush()
        await asyncio.sleep(0.02)
    sys.stdout.write("\033[0m\n")
    sys.stdout.flush()
    
    try:
        while True:
            try:
                message = await websocket.receive_text()
                data = json.loads(message)
                
                if data.get("type") == "analyze":
                    video_id = data.get("videoId", "unknown")
                    metadata = data.get("metadata", {})
                    
                    frames = []
                    for b64 in data.get("frames", []):
                        try:
                            img_data = b64.split(",")[1]
                            img_bytes = base64.b64decode(img_data)
                            img = Image.open(io.BytesIO(img_bytes))
                            arr = np.array(img)
                            frames.append(cv2.cvtColor(arr, cv2.COLOR_RGB2BGR))
                        except Exception as e:
                            print(f"\033[91m⚠️ Decode error: {e}\033[0m")
                    
                    if frames:
                        result = await analyzer.analyze_video(frames, metadata=metadata)
                        result["videoId"] = video_id
                        
                        try:
                            safe_json = json.dumps(result, allow_nan=False)
                            await websocket.send_text(safe_json)
                        except Exception as e:
                            print(f"\033[91m⚠️ Send error: {e}\033[0m")
                            break
                    else:
                        try:
                            await websocket.send_json({
                                "videoId": video_id,
                                "error": "No valid frames"
                            })
                        except:
                            break

            except WebSocketDisconnect:
                sys.stdout.write("\033[1m\033[91m")
                for char in "Extension disconnected":
                    sys.stdout.write(char)
                    sys.stdout.flush()
                    await asyncio.sleep(0.02)
                sys.stdout.write("\033[0m\n")
                sys.stdout.flush()
                break
            except json.JSONDecodeError:
                print("Invalid JSON")
                continue
            except Exception as e:
                print(f"Error: {e}")
                continue
                
    except Exception as e:
        print(f"WebSocket error: {e}")
    finally:
        try:
            await websocket.close()
        except:
            pass

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="error")
