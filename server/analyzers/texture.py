import cv2
import numpy as np
from typing import Tuple


class TextureAnalyzer:
    def __init__(self):
        print("Loading texture analyzer...")
        
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        
        self.weights = {
            'frequency': 0.20,
            'repetition': 0.15,
            'edges': 0.20,
            'banding': 0.15,
            'compression': 0.15,
            'consistency': 0.15
        }
    
    def analyze(self, img: np.ndarray) -> Tuple[float, str]:
        h, w = img.shape[:2]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        results = {}
        flags = []
        
        results['frequency'], freq_desc = self._analyze_frequency(gray)
        results['repetition'], rep_desc = self._analyze_repetition(gray)
        results['edges'], edge_desc = self._analyze_edge_artifacts(gray)
        results['banding'], band_desc = self._analyze_color_banding(img)
        results['compression'], comp_desc = self._analyze_compression(gray)
        results['consistency'], cons_desc = self._analyze_detail_consistency(gray)
        
        for key, desc in [('frequency', freq_desc), ('repetition', rep_desc),
                          ('edges', edge_desc), ('banding', band_desc),
                          ('compression', comp_desc), ('consistency', cons_desc)]:
            if results[key] > 0.4:
                flags.append(desc)
        
        final_score = sum(results[key] * self.weights[key] for key in self.weights)
        
        anomaly_count = sum(1 for v in results.values() if v > 0.4)
        if anomaly_count >= 3:
            final_score = min(final_score * 1.3, 1.0)
        elif anomaly_count >= 2:
            final_score = min(final_score * 1.15, 1.0)
        
        desc = " | ".join(flags[:3]) if flags else "Normal texture"
        
        return min(final_score, 1.0), desc
    
    def _analyze_frequency(self, gray: np.ndarray) -> Tuple[float, str]:
        h, w = gray.shape
        
        f_transform = np.fft.fft2(gray)
        f_shift = np.fft.fftshift(f_transform)
        magnitude = np.abs(f_shift)
        magnitude = np.log1p(magnitude)
        
        center_y, center_x = h // 2, w // 2
        
        y_coords, x_coords = np.ogrid[:h, :w]
        dist_from_center = np.sqrt((y_coords - center_y)**2 + (x_coords - center_x)**2)
        
        max_dist = np.sqrt(center_y**2 + center_x**2)
        
        low_freq_mask = dist_from_center < max_dist * 0.15
        mid_freq_mask = (dist_from_center >= max_dist * 0.15) & (dist_from_center < max_dist * 0.5)
        high_freq_mask = dist_from_center >= max_dist * 0.5
        
        low_energy = np.mean(magnitude[low_freq_mask])
        mid_energy = np.mean(magnitude[mid_freq_mask])
        high_energy = np.mean(magnitude[high_freq_mask])
        
        if high_energy < low_energy * 0.1:
            return 0.75, "Missing high-frequency detail"
        elif high_energy < low_energy * 0.2:
            return 0.5, "Low high-frequency content"
        elif high_energy < low_energy * 0.3:
            return 0.3, "Slightly smooth"
        
        return 0.1, "Natural frequency spectrum"
    
    def _analyze_repetition(self, gray: np.ndarray) -> Tuple[float, str]:
        h, w = gray.shape
        
        patch_size = min(64, h // 4, w // 4)
        if patch_size < 16:
            return 0.1, "Image too small"
        
        patches = []
        positions = []
        
        for y in range(0, h - patch_size, patch_size):
            for x in range(0, w - patch_size, patch_size):
                patch = gray[y:y+patch_size, x:x+patch_size]
                patches.append(patch.flatten())
                positions.append((y, x))
        
        if len(patches) < 4:
            return 0.1, "Insufficient patches"
        
        patches = np.array(patches)
        
        similarities = []
        for i in range(len(patches)):
            for j in range(i + 1, min(len(patches), i + 10)):
                p1 = patches[i] - np.mean(patches[i])
                p2 = patches[j] - np.mean(patches[j])
                
                std1, std2 = np.std(p1), np.std(p2)
                if std1 > 0 and std2 > 0:
                    corr = np.dot(p1, p2) / (std1 * std2 * len(p1))
                    
                    y1, x1 = positions[i]
                    y2, x2 = positions[j]
                    if abs(y1 - y2) > patch_size or abs(x1 - x2) > patch_size:
                        similarities.append(corr)
        
        if len(similarities) == 0:
            return 0.1, "Cannot compute similarity"
        
        high_sim_count = sum(1 for s in similarities if s > 0.85)
        high_sim_ratio = high_sim_count / len(similarities)
        
        if high_sim_ratio > 0.4:
            return 0.8, "Repetitive texture patterns"
        elif high_sim_ratio > 0.2:
            return 0.5, "Some texture repetition"
        elif high_sim_ratio > 0.1:
            return 0.3, "Minor repetition"
        
        return 0.1, "Natural texture variation"
    
    def _analyze_edge_artifacts(self, gray: np.ndarray) -> Tuple[float, str]:
        edges = cv2.Canny(gray, 50, 150)
        
        if np.sum(edges > 0) < 100:
            return 0.1, "Few edges to analyze"
        
        kernel = np.ones((5, 5), np.uint8)
        edge_neighborhood = cv2.dilate(edges, kernel, iterations=2)
        edge_neighborhood = edge_neighborhood - edges
        
        neighborhood_mask = edge_neighborhood > 0
        
        if np.sum(neighborhood_mask) < 100:
            return 0.1, "Insufficient edge data"
        
        gray_float = gray.astype(float)
        
        grad_x = cv2.Sobel(gray_float, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray_float, cv2.CV_64F, 0, 1, ksize=3)
        grad_mag = np.sqrt(grad_x**2 + grad_y**2)
        
        edge_mask = edges > 0
        edge_gradient = np.mean(grad_mag[edge_mask])
        neighbor_gradient = np.mean(grad_mag[neighborhood_mask])
        
        gradient_ratio = neighbor_gradient / max(edge_gradient, 1)
        
        if gradient_ratio > 0.5:
            return 0.75, "Edge halos detected"
        elif gradient_ratio > 0.35:
            return 0.5, "Possible edge artifacts"
        elif gradient_ratio > 0.25:
            return 0.3, "Minor edge irregularities"
        
        return 0.1, "Clean edges"
    
    def _analyze_color_banding(self, img: np.ndarray) -> Tuple[float, str]:
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l_channel = lab[:, :, 0].astype(float)
        
        h, w = l_channel.shape
        
        grad_x = np.diff(l_channel, axis=1)
        grad_y = np.diff(l_channel, axis=0)
        
        banding_score = 0.0
        
        for row_idx in range(0, h, h // 10):
            row_diff = grad_x[row_idx, :]
            
            near_zero = np.sum(np.abs(row_diff) < 2)
            jumps = np.sum(np.abs(row_diff) > 5)
            
            total = len(row_diff)
            if total > 0:
                if near_zero > total * 0.6 and jumps > total * 0.05:
                    banding_score += 0.1
        
        for col_idx in range(0, w, w // 10):
            col_diff = grad_y[:, col_idx]
            
            near_zero = np.sum(np.abs(col_diff) < 2)
            jumps = np.sum(np.abs(col_diff) > 5)
            
            total = len(col_diff)
            if total > 0:
                if near_zero > total * 0.6 and jumps > total * 0.05:
                    banding_score += 0.1
        
        if banding_score > 0.8:
            return 0.8, "Severe color banding"
        elif banding_score > 0.5:
            return 0.55, "Noticeable banding"
        elif banding_score > 0.3:
            return 0.35, "Minor banding"
        
        return 0.1, "Smooth gradients"
    
    def _analyze_compression(self, gray: np.ndarray) -> Tuple[float, str]:
        h, w = gray.shape
        
        block_size = 8
        block_boundaries_h = []
        block_boundaries_v = []
        
        for y in range(block_size, h - block_size, block_size):
            boundary_diff = np.abs(
                gray[y, :].astype(float) - gray[y-1, :].astype(float)
            )
            interior_diff = np.abs(
                gray[y+2, :].astype(float) - gray[y+1, :].astype(float)
            )
            
            block_boundaries_h.append(np.mean(boundary_diff) / max(np.mean(interior_diff), 1))
        
        for x in range(block_size, w - block_size, block_size):
            boundary_diff = np.abs(
                gray[:, x].astype(float) - gray[:, x-1].astype(float)
            )
            interior_diff = np.abs(
                gray[:, x+2].astype(float) - gray[:, x+1].astype(float)
            )
            
            block_boundaries_v.append(np.mean(boundary_diff) / max(np.mean(interior_diff), 1))
        
        if len(block_boundaries_h) == 0 or len(block_boundaries_v) == 0:
            return 0.1, "Cannot analyze compression"
        
        h_mean = np.mean(block_boundaries_h)
        v_mean = np.mean(block_boundaries_v)
        h_std = np.std(block_boundaries_h)
        v_std = np.std(block_boundaries_v)
        
        if h_mean < 1.1 and v_mean < 1.1:
            return 0.4, "No compression artifacts"
        elif h_std > h_mean * 0.5 or v_std > v_mean * 0.5:
            return 0.6, "Inconsistent compression"
        
        return 0.1, "Normal compression"
    
    def _analyze_detail_consistency(self, gray: np.ndarray) -> Tuple[float, str]:
        h, w = gray.shape
        
        grid_size = 4
        sharpness_map = []
        
        for row in range(grid_size):
            row_sharpness = []
            for col in range(grid_size):
                y1, y2 = row * h // grid_size, (row + 1) * h // grid_size
                x1, x2 = col * w // grid_size, (col + 1) * w // grid_size
                
                region = gray[y1:y2, x1:x2]
                
                laplacian = cv2.Laplacian(region, cv2.CV_64F)
                sharpness = np.var(laplacian)
                row_sharpness.append(sharpness)
            
            sharpness_map.append(row_sharpness)
        
        sharpness_map = np.array(sharpness_map)
        
        max_sharpness = np.max(sharpness_map)
        min_sharpness = np.min(sharpness_map)
        
        if max_sharpness < 1:
            return 0.5, "Uniformly blurry"
        
        sharpness_range = max_sharpness / max(min_sharpness, 1)
        
        abrupt_transitions = 0
        for row in range(grid_size):
            for col in range(grid_size):
                current = sharpness_map[row, col]
                
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nr, nc = row + dr, col + dc
                    if 0 <= nr < grid_size and 0 <= nc < grid_size:
                        neighbor = sharpness_map[nr, nc]
                        ratio = max(current, neighbor) / max(min(current, neighbor), 1)
                        if ratio > 5:
                            abrupt_transitions += 1
        
        if abrupt_transitions > 6:
            return 0.75, "Abrupt detail transitions"
        elif sharpness_range > 20:
            return 0.55, "Inconsistent detail levels"
        elif abrupt_transitions > 3:
            return 0.4, "Some detail inconsistency"
        
        return 0.1, "Consistent detail"