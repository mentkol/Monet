"""
Vision Transformer (ViT) Detector for AI-generated content.
Uses ONNX Runtime for fast inference with the Organika/sdxl-detector model
(Swin Transformer fine-tuned for modern diffusion model detection).
"""

import numpy as np
import cv2
import os
import time

class ViTDetector:
    def __init__(self, model_name="Organika/sdxl-detector", cache_dir=None):
        """
        Initialize the ViT detector.
        
        Downloads and exports the model to ONNX on first run, then uses
        ONNX Runtime for fast inference on subsequent runs.
        
        Args:
            model_name: Hugging Face model identifier
            cache_dir: Optional cache directory for the ONNX model
        """
        self.model_name = model_name
        self.pipeline = None
        self.processor = None
        self.session = None
        self.loaded = False
        
        # Default cache dir
        if cache_dir is None:
            cache_dir = os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                "..", "models", "vit_onnx"
            )
        self.cache_dir = os.path.abspath(cache_dir)
        
        self._load_model()
    
    def _load_model(self):
        """Load or download the ONNX model."""
        try:
            from optimum.onnxruntime import ORTModelForImageClassification
            from transformers import AutoImageProcessor
            
            print(f"Loading ViT model: {self.model_name}")
            start = time.time()
            
            onnx_model_path = self.cache_dir
            
            # Check if ONNX model is already cached locally
            if os.path.exists(os.path.join(onnx_model_path, "model.onnx")):
                print("  Loading cached ONNX model...")
                self.session = ORTModelForImageClassification.from_pretrained(
                    onnx_model_path,
                    local_files_only=True
                )
            else:
                print("  Downloading and exporting to ONNX (one-time)...")
                os.makedirs(onnx_model_path, exist_ok=True)
                self.session = ORTModelForImageClassification.from_pretrained(
                    self.model_name,
                    export=True
                )
                # Save locally for fast loading next time
                self.session.save_pretrained(onnx_model_path)
                print(f"  Model cached at: {onnx_model_path}")
            
            # Load the image processor (handles resize, normalize, etc.)
            processor_path = os.path.join(onnx_model_path, "preprocessor_config.json")
            if os.path.exists(processor_path):
                self.processor = AutoImageProcessor.from_pretrained(onnx_model_path)
            else:
                self.processor = AutoImageProcessor.from_pretrained(self.model_name)
                self.processor.save_pretrained(onnx_model_path)
            
            elapsed = time.time() - start
            self.loaded = True
            print(f"  ViT model loaded in {elapsed:.1f}s")
            
        except Exception as e:
            print(f"  ⚠️  ViT model failed to load: {e}")
            print("  ViT detector will return neutral scores (0.0)")
            self.loaded = False
    
    def _preprocess(self, frame):
        """
        Convert BGR OpenCV frame to PIL Image for the processor.
        
        Args:
            frame: BGR numpy array from OpenCV
            
        Returns:
            PIL Image in RGB format
        """
        from PIL import Image
        
        # BGR → RGB
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return Image.fromarray(rgb)
    
    def analyze_frame(self, frame):
        """
        Analyze a single frame for AI-generated content.
        
        Args:
            frame: BGR numpy array from OpenCV
            
        Returns:
            tuple: (ai_probability, description)
                - ai_probability: float 0-1
                - description: string explaining the result
        """
        if not self.loaded:
            return 0.0, "ViT unavailable"
        
        try:
            import torch
            
            pil_image = self._preprocess(frame)
            
            # Process through the image processor
            inputs = self.processor(images=pil_image, return_tensors="pt")
            
            # Run ONNX inference
            with torch.no_grad():
                outputs = self.session(**inputs)
            
            # Get probabilities via softmax
            logits = outputs.logits
            probs = torch.nn.functional.softmax(logits, dim=-1)[0]
            
            # Model outputs: typically index 0 = "artificial", index 1 = "human"
            # But check the model's label mapping
            id2label = self.session.config.id2label
            
            # Find the AI/artificial class probability
            ai_prob = 0.0
            label_desc = "unknown"
            
            for idx, label in id2label.items():
                label_lower = label.lower()
                if any(keyword in label_lower for keyword in ['artificial', 'ai', 'fake', 'generated']):
                    ai_prob = float(probs[int(idx)])
                    label_desc = label
                    break
            
            if ai_prob >= 0.7:
                desc = f"ViT: {ai_prob:.0%} AI ({label_desc})"
            elif ai_prob >= 0.4:
                desc = f"ViT: {ai_prob:.0%} possibly AI"
            else:
                desc = f"ViT: {ai_prob:.0%} likely authentic"
            
            return ai_prob, desc
            
        except Exception as e:
            print(f"  ViT inference error: {e}")
            return 0.0, f"ViT error: {str(e)[:50]}"
    
    def analyze_frames(self, frames):
        """
        Analyze multiple frames and return the maximum AI probability.
        
        Uses a max strategy: analyzes 2 key frames (first + middle) and
        returns the highest AI probability found. This catches both
        thumbnail tricks and mid-video AI artifacts.
        
        Args:
            frames: list of BGR numpy arrays
            
        Returns:
            tuple: (max_ai_probability, description)
        """
        if not self.loaded:
            return 0.0, "ViT unavailable"
        
        if not frames:
            return 0.0, "No frames"
        
        # Select up to 8 evenly spaced frames. Returning aggregate stats is less
        # brittle than treating one high/low frame as the whole video.
        sample_count = min(8, len(frames))
        key_indices = np.linspace(0, len(frames) - 1, sample_count, dtype=int)
        key_indices = list(dict.fromkeys(int(i) for i in key_indices))
        
        scores = []
        descs = []
        
        for idx in key_indices:
            score, desc = self.analyze_frame(frames[idx])
            scores.append(score)
            descs.append(desc)
        
        # Keep backward-compatible behavior: primary score is still the max.
        max_score = max(scores)
        max_idx = scores.index(max_score)
        
        return max_score, descs[max_idx]

    def analyze_frame_stats(self, frames):
        """
        Analyze multiple frames and return mean/max/std statistics.

        Returns:
            tuple: (mean_probability, max_probability, std_probability, description)
        """
        if not self.loaded:
            return 0.0, 0.0, 0.0, "ViT unavailable"

        if not frames:
            return 0.0, 0.0, 0.0, "No frames"

        sample_count = min(8, len(frames))
        key_indices = np.linspace(0, len(frames) - 1, sample_count, dtype=int)
        key_indices = list(dict.fromkeys(int(i) for i in key_indices))

        scores = []
        descs = []
        for idx in key_indices:
            score, desc = self.analyze_frame(frames[idx])
            scores.append(score)
            descs.append(desc)

        if not scores:
            return 0.0, 0.0, 0.0, "No ViT scores"

        max_idx = int(np.argmax(scores))
        return float(np.mean(scores)), float(np.max(scores)), float(np.std(scores)), descs[max_idx]
