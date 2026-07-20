import numpy as np
from sklearn.ensemble import RandomForestClassifier as SKRandomForest
from sklearn.preprocessing import StandardScaler
import pickle
import os

class RandomForestClassifier:
    FEATURE_NAMES = [
        'texture_mean', 'texture_max', 'texture_std',
        'bio_mean', 'bio_max', 'bio_std',
        'color_mean', 'color_max', 'color_std',
        'digital_penalty',
        'semantic',
        'vit_mean', 'vit_max', 'vit_std',
        'metadata_score',
        'metadata_hits'
    ]

    def __init__(self, model_path=None):
        self.scaler = StandardScaler()
        self.model = None
        self.is_trained = False
        
        if model_path and os.path.exists(model_path):
            try:
                self.load(model_path)
            except:
                print("Failed to load model.")
        else:
            print("No trained model found. Please run training script.")
        
        print("Random Forest classifier loaded")

    def extract_features(self, texture_mean, texture_max, texture_std,
                         bio_mean, bio_max, bio_std,
                         color_mean, color_max, color_std,
                         digital_penalty, semantic_score,
                         vit_score=0.0,
                         metadata_score=0.0, metadata_hits=0.0,
                         vit_mean=None, vit_max=None, vit_std=0.0):
        if vit_mean is None:
            vit_mean = vit_score
        if vit_max is None:
            vit_max = vit_score

        features = np.array([[
            texture_mean, texture_max, texture_std,
            bio_mean, bio_max, bio_std,
            color_mean, color_max, color_std,
            digital_penalty,
            semantic_score,
            vit_mean,
            vit_max,
            vit_std,
            metadata_score,
            metadata_hits
        ]])
        
        return features

    def _expected_feature_count(self):
        if hasattr(self.scaler, "n_features_in_"):
            return int(self.scaler.n_features_in_)
        if self.model is not None and hasattr(self.model, "n_features_in_"):
            return int(self.model.n_features_in_)
        return len(self.FEATURE_NAMES)

    def _manual_score(self, features, rf_score=0.0):
        texture = features[0, 0]
        biometrics = features[0, 3]
        color_score = features[0, 6]
        digital_penalty = features[0, 9]
        semantic = features[0, 10]
        vit_mean = features[0, 11] if features.shape[1] > 11 else 0.0
        vit_max = features[0, 12] if features.shape[1] > 12 else vit_mean
        metadata_score = features[0, 14] if features.shape[1] > 14 else 0.0

        vit = (vit_mean * 0.45) + (vit_max * 0.55)

        weighted_score = (
            texture * 0.10 +
            biometrics * 0.12 +
            semantic * 0.09 +
            color_score * 0.02 +
            vit * 0.52 +
            metadata_score * 0.12 +
            rf_score * 0.03
        )

        strong_signals = sum([
            1 if biometrics >= 0.3 else 0,
            1 if texture >= 0.35 else 0,
            1 if semantic >= 0.3 else 0,
            1 if vit >= 0.4 else 0,
            1 if metadata_score >= 0.35 else 0
        ])

        if strong_signals >= 4:
            weighted_score = min(weighted_score * 1.55, 1.0)
        elif strong_signals >= 3:
            weighted_score = min(weighted_score * 1.35, 1.0)
        elif strong_signals >= 2:
            weighted_score = min(weighted_score * 1.18, 1.0)

        ai_score = weighted_score * (1 - digital_penalty * 0.5)
        return min(max(ai_score, 0.0), 1.0)

    def _apply_evidence_floors(self, features, ai_score):
        texture = features[0, 0]
        biometrics = features[0, 3]
        semantic = features[0, 10]
        vit_mean = features[0, 11] if features.shape[1] > 11 else 0.0
        vit_max = features[0, 12] if features.shape[1] > 12 else vit_mean
        metadata_score = features[0, 14] if features.shape[1] > 14 else 0.0
        digital_penalty = features[0, 9]

        corroborating_signals = sum([
            1 if texture >= 0.35 else 0,
            1 if biometrics >= 0.30 else 0,
            1 if semantic >= 0.30 else 0,
            1 if metadata_score >= 0.35 else 0,
        ])

        if vit_mean >= 0.55:
            ai_score = max(ai_score, 0.60)
        elif vit_max >= 0.70 and corroborating_signals >= 1:
            ai_score = max(ai_score, 0.60)
        elif vit_max >= 0.55 and corroborating_signals >= 2:
            ai_score = max(ai_score, 0.56)

        if metadata_score >= 0.70:
            ai_score = max(ai_score, 0.58)
        elif metadata_score >= 0.35 and vit_max >= 0.45:
            ai_score = max(ai_score, 0.55)

        if digital_penalty >= 0.30 and vit_max < 0.55 and metadata_score < 0.35:
            ai_score = min(ai_score, 0.49)

        return min(max(ai_score, 0.0), 1.0)
    
    def predict(self, features):
        if not self.is_trained:
            raise RuntimeError("Model not trained. Call train() first.")
        expected_features = self._expected_feature_count()
        rf_confidence = 0.0
        rf_score = 0.0

        if expected_features == features.shape[1]:
            features_scaled = self.scaler.transform(features)
            proba = self.model.predict_proba(features_scaled)[0]
            predicted_class = self.model.predict(features_scaled)[0]
            rf_confidence = proba[predicted_class]
            rf_score = proba[1] * 0.7 + proba[2]
        
        CONFIDENCE_THRESHOLD = 0.45
        
        manual_score = self._manual_score(features, rf_score)

        if rf_confidence >= CONFIDENCE_THRESHOLD:
            digital_penalty = features[0, 9]
            rf_ai_score = rf_score * (1 - digital_penalty * 0.5)
            ai_score = (manual_score * 0.65) + (rf_ai_score * 0.35)
        else:
            ai_score = manual_score

        ai_score = self._apply_evidence_floors(features, ai_score)
        
        ai_score = min(max(ai_score, 0.0), 1.0)
        
        if ai_score >= 0.58:
            label = "STRONG AI EVIDENCE"
            color = "#ef4444"
        elif ai_score >= 0.32:
            label = "MIXED AI EVIDENCE"
            color = "#f97316"
        else:
            label = "LOW AI EVIDENCE"
            color = "#22c55e"
        
        return ai_score, label, color, rf_confidence
    
    def train(self, X, y):
        X = np.array(X)
        y = np.array(y)
        
        self.scaler.fit(X)
        X_scaled = self.scaler.transform(X)
        
        self.model = SKRandomForest(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            class_weight='balanced'
        )
        self.model.fit(X_scaled, y)
        self.is_trained = True
        
        print(f"Model trained on {len(y)} samples")
    
    def save(self, path):
        data = {
            'model': self.model,
            'scaler': self.scaler,
            'is_trained': self.is_trained,
            'feature_names': self.FEATURE_NAMES
        }
        with open(path, 'wb') as f:
            pickle.dump(data, f)
        print(f"Model saved to {path}")
    
    def load(self, path):
        with open(path, 'rb') as f:
            data = pickle.load(f)
        self.model = data['model']
        self.scaler = data['scaler']
        self.is_trained = data['is_trained']
        self.feature_names = data.get('feature_names', self.FEATURE_NAMES[:self._expected_feature_count()])
        print(f"Model loaded from {path}")
    
    def get_feature_importance(self):
        if not self.is_trained:
            return None
        
        feature_names = getattr(self, 'feature_names', self.FEATURE_NAMES)
        importances = self.model.feature_importances_
        
        return dict(zip(feature_names, importances))
