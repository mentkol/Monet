import os
import cv2
import numpy as np
from typing import Tuple, List, Optional


class SemanticAnalyzer:
    def __init__(self):
        print("Loading semantic analyzer...")

        # OCR (Tesseract) is disabled by default: it adds ~0.3s/video for
        # marginal signal. Re-enable by setting MONET_OCR=1.
        self.ocr_available = False
        if os.environ.get("MONET_OCR", "0") == "1":
            try:
                import pytesseract
                pytesseract.get_tesseract_version()
                self.ocr_available = True
                print("  OCR: Tesseract available")
            except:
                print("  OCR: Tesseract not available (text detection limited)")
        else:
            print("  OCR: disabled (set MONET_OCR=1 to enable)")

        self.weights = {
            'text': 0.25,
            'counting': 0.25,
            'symmetry': 0.20,
            'watermarks': 0.15,
            'context': 0.15
        }

    def analyze_sync(self, frames):
        """Synchronous wrapper for use inside worker threads."""
        return self._run_analyze(frames)

    async def analyze(self, frames: List[np.ndarray]) -> Tuple[float, str]:
        return self._run_analyze(frames)

    def _run_analyze(self, frames):
        if len(frames) == 0:
            return 0.1, "No frames"

        img = frames[len(frames) // 2]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        results = {}
        flags = []

        results['text'], text_desc = self._analyze_text(img, gray)
        results['counting'], count_desc = self._analyze_object_counting(img, gray)
        results['symmetry'], sym_desc = self._analyze_symmetry(gray)
        results['watermarks'], wm_desc = self._analyze_watermarks(img, gray)
        results['context'], ctx_desc = self._analyze_context(img)

        for key, desc in [('text', text_desc), ('counting', count_desc),
                          ('symmetry', sym_desc), ('watermarks', wm_desc),
                          ('context', ctx_desc)]:
            if results[key] > 0.4:
                flags.append(desc)

        final_score = sum(results[key] * self.weights[key] for key in self.weights)

        anomaly_count = sum(1 for v in results.values() if v > 0.4)
        if anomaly_count >= 3:
            final_score = min(final_score * 1.3, 1.0)
        elif anomaly_count >= 2:
            final_score = min(final_score * 1.15, 1.0)

        desc = " | ".join(flags[:3]) if flags else "Normal semantics"

        return min(final_score, 1.0), desc

    def _analyze_text(self, img: np.ndarray, gray: np.ndarray) -> Tuple[float, str]:
        if not self.ocr_available:
            return self._analyze_text_patterns(gray)

        try:
            import pytesseract

            text = pytesseract.image_to_string(img, config='--psm 11')

            if len(text.strip()) < 3:
                return 0.1, "No text detected"

            words = text.split()

            if len(words) == 0:
                return 0.1, "No readable words"

            garbled_count = 0
            for word in words:
                word = word.strip()
                if len(word) < 2:
                    continue

                consonant_streak = 0
                vowel_streak = 0
                vowels = set('aeiouAEIOU')

                for char in word:
                    if char.isalpha():
                        if char in vowels:
                            vowel_streak += 1
                            consonant_streak = 0
                        else:
                            consonant_streak += 1
                            vowel_streak = 0

                        if consonant_streak >= 4 or vowel_streak >= 3:
                            garbled_count += 1
                            break

                if len(word) > 3:
                    for i in range(len(word) - 2):
                        if word[i] == word[i+1] == word[i+2]:
                            garbled_count += 1
                            break

            garbled_ratio = garbled_count / len(words) if len(words) > 0 else 0

            if garbled_ratio > 0.5:
                return 0.85, "Garbled text detected"
            elif garbled_ratio > 0.3:
                return 0.6, "Some nonsense text"
            elif garbled_ratio > 0.15:
                return 0.35, "Minor text issues"

            return 0.1, "Normal text"

        except Exception as e:
            return 0.1, "OCR error"

    def _analyze_text_patterns(self, gray: np.ndarray) -> Tuple[float, str]:
        edges = cv2.Canny(gray, 50, 150)

        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 1))
        horizontal = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, horizontal_kernel)

        contours, _ = cv2.findContours(horizontal, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        text_like_regions = 0
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = w / max(h, 1)

            if aspect_ratio > 3 and w > 30:
                text_like_regions += 1

        if text_like_regions > 5:
            return 0.2, "Text regions detected (no OCR)"

        return 0.1, "No obvious text"

    def _analyze_object_counting(self, img: np.ndarray,
                                  gray: np.ndarray) -> Tuple[float, str]:
        h, w = gray.shape
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        issues = []

        circles = cv2.HoughCircles(
            gray, cv2.HOUGH_GRADIENT, 1, 20,
            param1=50, param2=30, minRadius=10, maxRadius=100
        )

        if circles is not None:
            radii = circles[0][:, 2]

            size_groups = {}
            for r in radii:
                bucket = int(r / 10) * 10
                size_groups[bucket] = size_groups.get(bucket, 0) + 1

            for bucket, count in size_groups.items():
                if count == 3:
                    issues.append(f"odd_circle_count_{count}")

        edges = cv2.Canny(gray, 30, 100)

        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        elongated_objects = []
        for contour in contours:
            if len(contour) < 5:
                continue

            rect = cv2.minAreaRect(contour)
            width, height = rect[1]

            if min(width, height) > 0:
                aspect = max(width, height) / min(width, height)
                if aspect > 4:
                    elongated_objects.append((width, height, rect[0]))

        if len(elongated_objects) > 8:
            positions = [obj[2] for obj in elongated_objects]

            quadrant_counts = [0, 0, 0, 0]
            for pos in positions:
                qx = 0 if pos[0] < w/2 else 1
                qy = 0 if pos[1] < h/2 else 1
                quadrant_counts[qy * 2 + qx] += 1

            if max(quadrant_counts) > 7:
                issues.append("too_many_elongated_objects")

        similar_contours = self._find_repeated_shapes(contours)
        if similar_contours > 6:
            issues.append("excessive_object_repetition")

        if len(issues) >= 2:
            return 0.75, "Multiple counting anomalies"
        elif len(issues) == 1:
            return 0.5, issues[0]

        return 0.1, "Normal object counts"

    def _find_repeated_shapes(self, contours: List) -> int:
        if len(contours) < 4:
            return 0

        moments = []
        for contour in contours:
            if cv2.contourArea(contour) > 100:
                hu = cv2.HuMoments(cv2.moments(contour)).flatten()
                moments.append(hu)

        if len(moments) < 4:
            return 0

        similar_count = 0
        for i in range(len(moments)):
            for j in range(i + 1, len(moments)):
                m1 = -np.sign(moments[i]) * np.log10(np.abs(moments[i]) + 1e-10)
                m2 = -np.sign(moments[j]) * np.log10(np.abs(moments[j]) + 1e-10)

                diff = np.sum(np.abs(m1[:4] - m2[:4]))

                if diff < 1.0:
                    similar_count += 1

        return similar_count

    def _analyze_symmetry(self, gray: np.ndarray) -> Tuple[float, str]:
        h, w = gray.shape

        mid = w // 2
        left_half = gray[:, :mid]
        right_half = cv2.flip(gray[:, mid:2*mid], 1)

        min_w = min(left_half.shape[1], right_half.shape[1])
        left_half = left_half[:, :min_w]
        right_half = right_half[:, :min_w]

        diff = cv2.absdiff(left_half, right_half)

        mean_diff = np.mean(diff)

        left_edges = cv2.Canny(left_half, 50, 150)
        right_edges = cv2.Canny(right_half, 50, 150)
        edge_diff = cv2.absdiff(left_edges, right_edges)
        edge_asymmetry = np.mean(edge_diff)

        if mean_diff < 5 and edge_asymmetry < 10:
            return 0.6, "Unnaturally perfect symmetry"

        if mean_diff < 8:
            return 0.4, "Highly symmetric (possibly AI)"

        return 0.1, "Natural symmetry"

    def _analyze_watermarks(self, img: np.ndarray,
                            gray: np.ndarray) -> Tuple[float, str]:
        h, w = gray.shape

        corner_size = min(h, w) // 6

        corners = [
            gray[:corner_size, :corner_size],
            gray[:corner_size, -corner_size:],
            gray[-corner_size:, :corner_size],
            gray[-corner_size:, -corner_size:]
        ]

        watermark_indicators = 0

        for corner in corners:
            local_std = np.std(corner)

            if 5 < local_std < 25:
                edges = cv2.Canny(corner, 30, 100)
                edge_ratio = np.sum(edges > 0) / edges.size

                if 0.02 < edge_ratio < 0.15:
                    watermark_indicators += 1

        bottom_strip = gray[-h//10:, :]

        strip_edges = cv2.Canny(bottom_strip, 50, 150)
        strip_edge_ratio = np.sum(strip_edges > 0) / strip_edges.size

        if 0.05 < strip_edge_ratio < 0.2:
            fft = np.fft.fft2(bottom_strip)
            fft_mag = np.abs(np.fft.fftshift(fft))

            peak_threshold = np.percentile(fft_mag, 99)
            peaks = np.sum(fft_mag > peak_threshold)

            if peaks > 20:
                watermark_indicators += 1

        if watermark_indicators >= 3:
            return 0.8, "Likely AI watermark"
        elif watermark_indicators >= 2:
            return 0.5, "Possible watermark"
        elif watermark_indicators >= 1:
            return 0.3, "Faint watermark signs"

        return 0.1, "No watermarks detected"

    def _analyze_context(self, img: np.ndarray) -> Tuple[float, str]:
        h, w = img.shape[:2]
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        edge_pixels = np.concatenate([
            hsv[0, :, :].reshape(-1, 3),
            hsv[-1, :, :].reshape(-1, 3),
            hsv[:, 0, :].reshape(-1, 3),
            hsv[:, -1, :].reshape(-1, 3)
        ])

        edge_sat_std = np.std(edge_pixels[:, 1])
        edge_val_std = np.std(edge_pixels[:, 2])

        center_h, center_w = h // 4, w // 4
        center = hsv[center_h:3*center_h, center_w:3*center_w]
        center_sat_std = np.std(center[:, :, 1])
        center_val_std = np.std(center[:, :, 2])

        edge_uniformity = edge_sat_std + edge_val_std
        center_complexity = center_sat_std + center_val_std

        if edge_uniformity < 30 and center_complexity > 60:
            complexity_ratio = center_complexity / max(edge_uniformity, 1)

            if complexity_ratio > 4:
                return 0.6, "Subject pasted on background"
            elif complexity_ratio > 2.5:
                return 0.4, "Possible background mismatch"

        return 0.1, "Consistent context"
