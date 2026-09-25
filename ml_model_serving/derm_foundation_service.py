"""
Derm Foundation Service - Advanced AI Analysis using Google's Derm Foundation model.

This service uses Google's Derm Foundation model from Hugging Face to generate
dermatology-specific embeddings, which are then classified using a trained classifier.

STATUS: LEGACY, AND NOT CURRENTLY WIRED UP.
Google's model card now says Derm Foundation "is now legacy, but remains available
for existing applications", and points new development at MedSigLIP instead:
https://developers.google.com/health-ai-developer-foundations/derm-foundation/model-card

Nothing here runs today either way. `DERM_FOUNDATION_AVAILABLE` only turns true once
a trained classifier and scaler load, and neither artefact is in this repository -
`scripts/train_derm_classifier.py` is what would produce them. So this is scaffolding.
Before building on it, decide whether to point it at MedSigLIP or remove it; do not
add to it as it stands.

LEGAL NOTICE:
This tool is not a medical device and does not provide medical diagnosis.
Uses Google Health AI Developer Foundations under their Terms of Use, which define
Clinical Use as "any use in diagnosis or treatment of patients" and make the
developer solely responsible for validating any downstream application.
Always consult a qualified healthcare professional for medical advice.

Requirements:
- Accept Google Health AI Developer Foundations Terms of Use
- Hugging Face account with access to google/derm-foundation
"""

import logging
import os
import pickle
import time
from io import BytesIO

import numpy as np
import tensorflow as tf
from PIL import Image

logger = logging.getLogger(__name__)

# Feature flags
DERM_FOUNDATION_AVAILABLE = False
DERM_MODEL_LOADED = False


class DermFoundationService:
    """
    Singleton service for advanced melanoma analysis using Google's Derm Foundation.

    The Derm Foundation model produces 6144-dimensional embeddings from skin images,
    which are then classified using a trained neural network classifier.

    Expected accuracy improvement: +10-15% over Xception baseline.
    """

    _instance = None
    _derm_model = None
    _infer_fn = None
    _classifier = None
    _scaler = None

    # Paths
    _classifier_path = os.environ.get(
        'DERM_CLASSIFIER_PATH',
        './cnn-models/derm-foundation/classifier.h5'
    )
    _scaler_path = os.environ.get(
        'DERM_SCALER_PATH',
        './cnn-models/derm-foundation/scaler.pkl'
    )

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(DermFoundationService, cls).__new__(cls)
            cls._load_models()
        return cls._instance

    @classmethod
    def _load_models(cls):
        """Load Derm Foundation model and classifier."""
        global DERM_FOUNDATION_AVAILABLE, DERM_MODEL_LOADED

        try:
            # Check if Hugging Face access is configured
            hf_token = os.environ.get('HUGGINGFACE_TOKEN')
            if hf_token:
                from huggingface_hub import login
                login(token=hf_token)

            # Load Derm Foundation from Hugging Face
            logger.info("Loading Derm Foundation model from Hugging Face...")
            start_time = time.time()

            from huggingface_hub import from_pretrained_keras
            cls._derm_model = from_pretrained_keras("google/derm-foundation")
            cls._infer_fn = cls._derm_model.signatures["serving_default"]

            load_time = time.time() - start_time
            logger.info(f"Derm Foundation model loaded in {load_time:.2f}s")

            # Load classifier (required for melanoma predictions — embeddings alone are not a diagnosis)
            if os.path.exists(cls._classifier_path):
                logger.info(f"Loading classifier from {cls._classifier_path}")
                cls._classifier = tf.keras.models.load_model(cls._classifier_path)
                logger.info("Classifier loaded successfully")
            else:
                logger.warning(
                    f"Classifier not found at {cls._classifier_path}. "
                    "Advanced prediction endpoints will return model_unavailable until "
                    "classifier.h5 and scaler.pkl are installed (see cnn-models/derm-foundation/README.md)."
                )
                cls._classifier = None

            # Load scaler (required with classifier)
            if os.path.exists(cls._scaler_path):
                logger.info(f"Loading scaler from {cls._scaler_path}")
                with open(cls._scaler_path, 'rb') as f:
                    cls._scaler = pickle.load(f)
                logger.info("Scaler loaded successfully")
            else:
                logger.warning(f"Scaler not found at {cls._scaler_path}")
                cls._scaler = None

            # Embedding model loaded; ready for predictions only when classifier artifacts exist
            DERM_MODEL_LOADED = True
            DERM_FOUNDATION_AVAILABLE = cls._classifier is not None and cls._scaler is not None
            if DERM_FOUNDATION_AVAILABLE:
                logger.info("Derm Foundation service initialized successfully (embeddings + classifier)")
            else:
                logger.warning(
                    "Derm Foundation embeddings loaded but classifier artifacts missing — "
                    "predict-advanced/compare-models will report model_unavailable"
                )

        except ImportError as e:
            logger.warning(f"Derm Foundation dependencies not available: {e}")
            logger.warning("Install with: pip install huggingface_hub")
            DERM_FOUNDATION_AVAILABLE = False
            DERM_MODEL_LOADED = False

        except Exception as e:
            logger.error(f"Failed to load Derm Foundation: {e}")
            logger.error("Ensure you have accepted Google's terms and have Hugging Face access")
            DERM_FOUNDATION_AVAILABLE = False
            DERM_MODEL_LOADED = False

    @classmethod
    def is_available(cls) -> bool:
        """True only when embeddings + trained classifier + scaler are ready."""
        return (
            DERM_FOUNDATION_AVAILABLE
            and DERM_MODEL_LOADED
            and cls._classifier is not None
            and cls._scaler is not None
        )

    @classmethod
    def unavailable_payload(cls, detail: str = None) -> dict:
        """Stable JSON shape for 503 / soft clients when the model cannot serve predictions."""
        return {
            'success': False,
            'error': detail or 'Derm Foundation model not available',
            'code': 'model_unavailable',
            'model_type': 'derm_foundation',
            'hint': (
                'Install cnn-models/derm-foundation/classifier.h5 and scaler.pkl '
                '(see README in that folder) and set HUGGINGFACE_TOKEN.'
            ),
        }
    @classmethod
    def warmup(cls):
        """Warm up the model with a dummy prediction."""
        if not cls.is_available():
            logger.warning("Derm Foundation not available for warmup")
            return

        logger.info("Warming up Derm Foundation model...")
        try:
            # Create a dummy 448x448 RGB image
            dummy_image = Image.new('RGB', (448, 448), color='white')
            buf = BytesIO()
            dummy_image.save(buf, 'PNG')
            dummy_bytes = buf.getvalue()

            # Run a dummy inference
            cls._get_embedding(dummy_bytes)
            logger.info("Derm Foundation warmup complete")
        except Exception as e:
            logger.error(f"Warmup failed: {e}")

    @classmethod
    def _preprocess_image(cls, image_data: bytes) -> bytes:
        """
        Preprocess image for Derm Foundation model.

        Requirements:
        - PNG format
        - 448x448 pixels
        - RGB format
        """
        img = Image.open(BytesIO(image_data))

        # Convert to RGB if necessary
        if img.mode != 'RGB':
            img = img.convert('RGB')

        # Resize to 448x448
        img = img.resize((448, 448), Image.Resampling.LANCZOS)

        # Convert to PNG bytes
        buf = BytesIO()
        img.save(buf, 'PNG')
        return buf.getvalue()

    @classmethod
    def _get_embedding(cls, image_bytes: bytes) -> np.ndarray:
        """
        Extract 6144-dimensional embedding from Derm Foundation.

        Args:
            image_bytes: PNG image as bytes

        Returns:
            6144-dimensional numpy array
        """
        # Create TFRecord format input
        input_tensor = tf.train.Example(
            features=tf.train.Features(
                feature={
                    'image/encoded': tf.train.Feature(
                        bytes_list=tf.train.BytesList(value=[image_bytes])
                    )
                }
            )
        ).SerializeToString()

        # Run inference
        output = cls._infer_fn(inputs=tf.constant([input_tensor]))

        # Extract embedding
        embedding = output['embedding'].numpy().flatten()
        return embedding

    def predict(self, image_data: bytes, threshold: float = 0.5) -> dict:
        """
        Make advanced melanoma prediction using Derm Foundation.

        Args:
            image_data: Image as bytes (JPEG or PNG)
            threshold: Classification threshold (default 0.5)

        Returns:
            dict with prediction results:
            - melanoma_probability: float (0-1)
            - prediction: 'Melanoma' or 'NotMelanoma'
            - confidence: float (0-1)
            - model_type: 'derm_foundation'
            - embedding_available: bool
        """
        if not self.is_available():
            return self.unavailable_payload()

        start_time = time.time()

        try:
            # Preprocess image
            png_bytes = self._preprocess_image(image_data)

            # Get embedding
            embedding = self._get_embedding(png_bytes)
            embedding_time = time.time() - start_time

            # Scale embedding and classify (artifacts guaranteed by is_available)
            embedding_scaled = self._scaler.transform(embedding.reshape(1, -1))
            prob = self._classifier.predict(embedding_scaled, verbose=0)[0, 0]

            # Note: 0 = Melanoma, 1 = NotMelanoma in training — invert for melanoma probability
            melanoma_prob = 1 - float(prob)
            total_time = time.time() - start_time

            result = {
                'success': True,
                'model_type': 'derm_foundation',
                'embedding_available': True,
                'embedding_dim': int(len(embedding)),
                'melanoma_probability': float(melanoma_prob),
                'prediction': 'Melanoma' if melanoma_prob >= threshold else 'NotMelanoma',
                'confidence': float(max(melanoma_prob, 1 - melanoma_prob)),
                'threshold': threshold,
                'processing_time_ms': int(total_time * 1000),
            }

            logger.info(
                f"Derm Foundation prediction: {result['prediction']} "
                f"(prob={melanoma_prob:.3f}, time={total_time*1000:.0f}ms)"
            )
            return result

        except Exception as e:
            logger.error(f"Derm Foundation prediction failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'code': 'prediction_failed',
                'model_type': 'derm_foundation',
            }

    def compare_with_baseline(
        self,
        image_data: bytes,
        baseline_prediction: dict,
        threshold: float = 0.5
    ) -> dict:
        """
        Compare Derm Foundation prediction with baseline (Xception) prediction.

        Args:
            image_data: Image as bytes
            baseline_prediction: Prediction from baseline model (Xception)
            threshold: Classification threshold

        Returns:
            Comparison result with both predictions and analysis
        """
        # Get Derm Foundation prediction
        derm_result = self.predict(image_data, threshold)

        if not derm_result.get('success'):
            payload = {
                'success': False,
                'error': derm_result.get('error', 'Derm Foundation prediction failed'),
                'code': derm_result.get('code', 'model_unavailable'),
                'baseline': baseline_prediction,
            }
            if derm_result.get('hint'):
                payload['hint'] = derm_result['hint']
            return payload

        # Extract baseline values
        baseline_prob = baseline_prediction.get('melanomaProbability', 0)
        baseline_pred = baseline_prediction.get('predictionResult', 'Unknown')

        # Extract Derm Foundation values
        derm_prob = derm_result.get('melanoma_probability', 0)
        derm_pred = derm_result.get('prediction', 'Unknown')

        # Determine agreement
        predictions_agree = baseline_pred == derm_pred
        probability_diff = abs(derm_prob - baseline_prob)

        # Combined confidence (weighted average, Derm Foundation weighted higher)
        combined_prob = (baseline_prob * 0.4 + derm_prob * 0.6)
        combined_pred = 'Melanoma' if combined_prob >= threshold else 'NotMelanoma'

        # Risk assessment
        if combined_prob >= 0.7:
            risk_level = 'High'
            recommendation = 'Consult a dermatologist promptly for professional evaluation.'
        elif combined_prob >= 0.4:
            risk_level = 'Moderate'
            recommendation = 'Schedule a dermatologist appointment for evaluation.'
        else:
            risk_level = 'Low'
            recommendation = 'Continue regular self-monitoring. Recheck in 1-3 months.'

        return {
            'success': True,
            'baseline': {
                'model': 'Xception',
                'melanoma_probability': baseline_prob,
                'prediction': baseline_pred
            },
            'advanced': {
                'model': 'DermFoundation',
                'melanoma_probability': derm_prob,
                'prediction': derm_pred,
                'confidence': derm_result.get('confidence', 0)
            },
            'combined': {
                'melanoma_probability': combined_prob,
                'prediction': combined_pred,
                'risk_level': risk_level,
                'recommendation': recommendation
            },
            'analysis': {
                'predictions_agree': predictions_agree,
                'probability_difference': probability_diff,
                'confidence_boost': derm_prob > baseline_prob if derm_pred == 'Melanoma' else derm_prob < baseline_prob
            },
            'processing_time_ms': derm_result.get('processing_time_ms', 0),
            'disclaimer': (
                'This analysis is for informational purposes only and does not constitute '
                'medical advice or diagnosis. Always consult a qualified healthcare '
                'professional for medical concerns.'
            )
        }
