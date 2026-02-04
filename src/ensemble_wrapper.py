"""
Ensemble model wrapper for predictions.
"""

import numpy as np


class EnsembleStackingWrapper:
    """Wrapper for ensemble stacking model to provide predict/predict_proba interface."""
    
    def __init__(self, base_models, meta_learner):
        """
        Initialize ensemble wrapper.
        
        Parameters
        ----------
        base_models : dict
            Dictionary of base models {name: model}
        meta_learner : estimator
            Meta-learner model (LogisticRegression)
        """
        self.base_models = base_models
        self.meta_learner = meta_learner
        self.type = 'ensemble_stacking'
    
    def _generate_meta_features(self, X):
        """Generate meta-features from base model predictions."""
        meta_features = []
        for name, model in self.base_models.items():
            proba = model.predict_proba(X)
            meta_features.append(proba)
        return np.hstack(meta_features)
    
    def predict(self, X):
        """Predict class labels."""
        meta_features = self._generate_meta_features(X)
        return self.meta_learner.predict(meta_features)
    
    def predict_proba(self, X):
        """Predict class probabilities."""
        meta_features = self._generate_meta_features(X)
        return self.meta_learner.predict_proba(meta_features)
