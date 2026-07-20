import mediapipe as mp
import cv2
import numpy as np
from typing import Tuple, List, Optional


class BiometricAnalyzer:
    def __init__(self):
        print("Loading biometrics analyzer...")
        
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(static_image_mode=True, max_num_hands=2)
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(static_image_mode=True)
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=True, max_num_faces=3, min_detection_confidence=0.5
        )
        
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        self.eye_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_eye.xml'
        )
        
        self.weights = {
            'hands': 0.25,
            'face_symmetry': 0.20,
            'proportions': 0.25,
            'teeth': 0.15,
            'ears': 0.15
        }
    
    async def analyze(self, img: np.ndarray) -> Tuple[float, str]:
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        results = {}
        flags = []
        
        results['hands'], hands_desc = self._analyze_hands(img_rgb)
        results['face_symmetry'], sym_desc = self._analyze_face_symmetry(img_rgb, gray)
        results['proportions'], prop_desc = self._analyze_body_proportions(img_rgb)
        results['teeth'], teeth_desc = self._analyze_teeth(img, gray)
        results['ears'], ears_desc = self._analyze_ears(img_rgb, gray)
        
        for key, desc in [('hands', hands_desc), ('face_symmetry', sym_desc),
                          ('proportions', prop_desc), ('teeth', teeth_desc),
                          ('ears', ears_desc)]:
            if results[key] > 0.4:
                flags.append(desc)
        
        final_score = sum(results[key] * self.weights[key] for key in self.weights)
        
        anomaly_count = sum(1 for v in results.values() if v > 0.4)
        if anomaly_count >= 3:
            final_score = min(final_score * 1.3, 1.0)
        elif anomaly_count >= 2:
            final_score = min(final_score * 1.15, 1.0)
        
        desc = " | ".join(flags[:3]) if flags else "Normal biometrics"
        
        return min(final_score, 1.0), desc
    
    def _analyze_hands(self, img_rgb: np.ndarray) -> Tuple[float, str]:
        hand_results = self.hands.process(img_rgb)
        
        if not hand_results.multi_hand_landmarks:
            return 0.1, "No hands detected"
        
        issues = []
        
        for hand_landmarks in hand_results.multi_hand_landmarks:
            landmarks = hand_landmarks.landmark
            
            finger_count = self._count_extended_fingers(landmarks)
            if finger_count > 5:
                issues.append(f"extra_fingers_{finger_count}")
            elif finger_count < 0:
                issues.append("missing_fingers")
            
            finger_issues = self._check_finger_proportions(landmarks)
            if finger_issues:
                issues.extend(finger_issues)
            
            thumb_issue = self._check_thumb_position(landmarks)
            if thumb_issue:
                issues.append(thumb_issue)
            
            angle_issues = self._check_joint_angles(landmarks)
            if angle_issues:
                issues.extend(angle_issues)
        
        if len(issues) >= 3:
            return 0.9, "Severe hand anomalies"
        elif len(issues) >= 2:
            return 0.7, "Multiple hand issues"
        elif len(issues) == 1:
            return 0.5, issues[0]
        
        return 0.1, "Normal hands"
    
    def _count_extended_fingers(self, landmarks) -> int:
        tips = [8, 12, 16, 20]
        pips = [6, 10, 14, 18]
        
        extended = 0
        for tip, pip in zip(tips, pips):
            if landmarks[tip].y < landmarks[pip].y:
                extended += 1
        
        if landmarks[4].x < landmarks[3].x:
            extended += 1
        
        return extended
    
    def _check_finger_proportions(self, landmarks) -> List[str]:
        issues = []
        
        fingers = {
            'index': [5, 6, 7, 8],
            'middle': [9, 10, 11, 12],
            'ring': [13, 14, 15, 16],
            'pinky': [17, 18, 19, 20]
        }
        
        for finger_name, indices in fingers.items():
            segments = []
            for i in range(len(indices) - 1):
                p1 = landmarks[indices[i]]
                p2 = landmarks[indices[i + 1]]
                length = np.sqrt((p2.x - p1.x)**2 + (p2.y - p1.y)**2 + (p2.z - p1.z)**2)
                segments.append(length)
            
            if len(segments) == 3 and min(segments) > 0:
                if segments[2] > segments[0] * 1.5:
                    issues.append(f"{finger_name}_wrong_proportions")
        
        return issues
    
    def _check_thumb_position(self, landmarks) -> Optional[str]:
        thumb_base = landmarks[1]
        index_base = landmarks[5]
        
        dist = np.sqrt((thumb_base.x - index_base.x)**2 + (thumb_base.y - index_base.y)**2)
        
        if dist < 0.05:
            return "thumb_wrong_position"
        
        return None
    
    def _check_joint_angles(self, landmarks) -> List[str]:
        issues = []
        
        finger_chains = [
            [5, 6, 7, 8],
            [9, 10, 11, 12],
            [13, 14, 15, 16],
            [17, 18, 19, 20]
        ]
        
        for chain in finger_chains:
            for i in range(len(chain) - 2):
                p1 = landmarks[chain[i]]
                p2 = landmarks[chain[i + 1]]
                p3 = landmarks[chain[i + 2]]
                
                v1 = np.array([p1.x - p2.x, p1.y - p2.y])
                v2 = np.array([p3.x - p2.x, p3.y - p2.y])
                
                if np.linalg.norm(v1) > 0 and np.linalg.norm(v2) > 0:
                    cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
                    cos_angle = np.clip(cos_angle, -1, 1)
                    angle = np.arccos(cos_angle) * 180 / np.pi
                    
                    if angle < 80:
                        issues.append("impossible_joint_bend")
                        break
        
        return issues
    
    def _analyze_face_symmetry(self, img_rgb: np.ndarray, 
                                gray: np.ndarray) -> Tuple[float, str]:
        face_results = self.face_mesh.process(img_rgb)
        
        if not face_results.multi_face_landmarks:
            return 0.1, "No faces detected"
        
        max_asymmetry = 0.0
        issues = []
        
        for face_landmarks in face_results.multi_face_landmarks:
            landmarks = face_landmarks.landmark
            
            pairs = [
                (33, 263, "eye_outer"),
                (133, 362, "eye_inner"),
                (61, 291, "mouth"),
                (70, 300, "eyebrow")
            ]
            
            nose = landmarks[1]
            
            for left_idx, right_idx, name in pairs:
                left = landmarks[left_idx]
                right = landmarks[right_idx]
                
                left_dist = abs(left.x - nose.x)
                right_dist = abs(right.x - nose.x)
                
                y_diff = abs(left.y - right.y)
                
                if max(left_dist, right_dist) > 0:
                    asymmetry = abs(left_dist - right_dist) / max(left_dist, right_dist)
                    if asymmetry > 0.25:
                        issues.append(f"asymmetric_{name}")
                        max_asymmetry = max(max_asymmetry, asymmetry)
                
                if name in ["eye_outer", "eye_inner"] and y_diff > 0.03:
                    issues.append("misaligned_eyes")
                    max_asymmetry = max(max_asymmetry, y_diff * 5)
        
        if len(issues) >= 3:
            return 0.85, "Severe face asymmetry"
        elif len(issues) >= 2:
            return 0.6, "Face asymmetry detected"
        elif len(issues) == 1:
            return 0.4, issues[0]
        elif max_asymmetry > 0.15:
            return 0.35, "Minor asymmetry"
        
        return 0.1, "Symmetric face"
    
    def _analyze_body_proportions(self, img_rgb: np.ndarray) -> Tuple[float, str]:
        pose_results = self.pose.process(img_rgb)
        
        if not pose_results.pose_landmarks:
            return 0.1, "No body detected"
        
        landmarks = pose_results.pose_landmarks.landmark
        issues = []
        
        left_upper_arm = self._get_segment_length(landmarks, 11, 13)
        left_forearm = self._get_segment_length(landmarks, 13, 15)
        right_upper_arm = self._get_segment_length(landmarks, 12, 14)
        right_forearm = self._get_segment_length(landmarks, 14, 16)
        
        if left_upper_arm > 0 and left_forearm > 0:
            ratio = left_upper_arm / left_forearm
            if ratio < 0.5 or ratio > 2.0:
                issues.append("wrong_arm_proportions")
        
        if right_upper_arm > 0 and right_forearm > 0:
            ratio = right_upper_arm / right_forearm
            if ratio < 0.5 or ratio > 2.0:
                issues.append("wrong_arm_proportions")
        
        left_thigh = self._get_segment_length(landmarks, 23, 25)
        left_shin = self._get_segment_length(landmarks, 25, 27)
        right_thigh = self._get_segment_length(landmarks, 24, 26)
        right_shin = self._get_segment_length(landmarks, 26, 28)
        
        if left_thigh > 0 and left_shin > 0:
            ratio = left_thigh / left_shin
            if ratio < 0.6 or ratio > 1.8:
                issues.append("wrong_leg_proportions")
        
        nose_y = landmarks[0].y
        left_hip_y = landmarks[23].y
        right_hip_y = landmarks[24].y
        
        if nose_y > max(left_hip_y, right_hip_y) * 1.3:
            issues.append("impossible_pose")
        
        if left_upper_arm > 0 and right_upper_arm > 0:
            arm_ratio = left_upper_arm / right_upper_arm
            if arm_ratio < 0.6 or arm_ratio > 1.6:
                issues.append("asymmetric_limbs")
        
        if len(issues) >= 3:
            return 0.85, "Severe proportion errors"
        elif len(issues) >= 2:
            return 0.6, "Body proportion issues"
        elif len(issues) == 1:
            return 0.45, issues[0]
        
        return 0.1, "Normal proportions"
    
    def _get_segment_length(self, landmarks, idx1: int, idx2: int) -> float:
        p1 = landmarks[idx1]
        p2 = landmarks[idx2]
        
        if p1.visibility < 0.5 or p2.visibility < 0.5:
            return 0.0
        
        return np.sqrt((p2.x - p1.x)**2 + (p2.y - p1.y)**2)
    
    def _analyze_teeth(self, img: np.ndarray, gray: np.ndarray) -> Tuple[float, str]:
        faces = self.face_cascade.detectMultiScale(gray, 1.1, 4, minSize=(60, 60))
        
        if len(faces) == 0:
            return 0.1, "No faces for teeth analysis"
        
        issues = []
        
        for (fx, fy, fw, fh) in faces[:2]:
            mouth_y = fy + int(fh * 0.65)
            mouth_h = int(fh * 0.25)
            mouth_x = fx + int(fw * 0.25)
            mouth_w = int(fw * 0.5)
            
            if mouth_y + mouth_h > img.shape[0] or mouth_x + mouth_w > img.shape[1]:
                continue
            
            mouth_region = img[mouth_y:mouth_y+mouth_h, mouth_x:mouth_x+mouth_w]
            mouth_gray = gray[mouth_y:mouth_y+mouth_h, mouth_x:mouth_x+mouth_w]
            
            if mouth_region.size == 0:
                continue
            
            hsv = cv2.cvtColor(mouth_region, cv2.COLOR_BGR2HSV)
            
            lower_white = np.array([0, 0, 180])
            upper_white = np.array([180, 60, 255])
            teeth_mask = cv2.inRange(hsv, lower_white, upper_white)
            
            teeth_pixels = cv2.countNonZero(teeth_mask)
            
            if teeth_pixels < 50:
                continue
            
            contours, _ = cv2.findContours(teeth_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            valid_contours = [c for c in contours if cv2.contourArea(c) > 20]
            
            if len(valid_contours) > 15:
                issues.append("too_many_teeth")
            elif len(valid_contours) == 1 and teeth_pixels > 500:
                issues.append("merged_teeth")
            
            if len(valid_contours) >= 3:
                areas = [cv2.contourArea(c) for c in valid_contours]
                area_std = np.std(areas)
                area_mean = np.mean(areas)
                
                if area_std > area_mean * 1.5:
                    issues.append("uneven_teeth")
        
        if len(issues) >= 2:
            return 0.75, "Multiple teeth anomalies"
        elif len(issues) == 1:
            return 0.5, issues[0]
        
        return 0.1, "Normal teeth"
    
    def _analyze_ears(self, img_rgb: np.ndarray, gray: np.ndarray) -> Tuple[float, str]:
        face_results = self.face_mesh.process(img_rgb)
        
        if not face_results.multi_face_landmarks:
            return 0.1, "No faces for ear analysis"
        
        issues = []
        
        for face_landmarks in face_results.multi_face_landmarks:
            landmarks = face_landmarks.landmark
            
            try:
                left_ear = landmarks[234]
                right_ear = landmarks[454]
                nose = landmarks[1]
                
                left_dist = abs(left_ear.x - nose.x)
                right_dist = abs(right_ear.x - nose.x)
                
                if max(left_dist, right_dist) > 0:
                    ear_asymmetry = abs(left_dist - right_dist) / max(left_dist, right_dist)
                    if ear_asymmetry > 0.3:
                        issues.append("asymmetric_ears")
                
                y_diff = abs(left_ear.y - right_ear.y)
                if y_diff > 0.05:
                    issues.append("misaligned_ears")
                
                left_eye = landmarks[33]
                
                ear_face_ratio_y = abs(left_ear.y - nose.y) / abs(left_eye.y - nose.y) if abs(left_eye.y - nose.y) > 0 else 0
                
                if ear_face_ratio_y > 2.0 or ear_face_ratio_y < 0.3:
                    issues.append("wrong_ear_placement")
                    
            except (IndexError, AttributeError):
                continue
        
        if len(issues) >= 2:
            return 0.7, "Multiple ear anomalies"
        elif len(issues) == 1:
            return 0.45, issues[0]
        
        return 0.1, "Normal ears"