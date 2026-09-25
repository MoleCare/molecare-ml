"""
Comprehensive Mole Analysis Service

Combines ML-based classification with ABCDE symptom analysis
for complete mole assessment.

Features:
1. Melanoma probability prediction (ML model)
2. ABCDE criteria analysis (computer vision)
3. Mole comparison over time (evolution tracking)
4. Risk stratification and recommendations
"""

import base64
import logging
from dataclasses import asdict, dataclass
from datetime import datetime
from io import BytesIO
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
from PIL import Image

from .abcde_analyzer import ABCDEAnalyzer, ABCDEScore, analyze_mole_comparison

logger = logging.getLogger(__name__)


@dataclass
class ComprehensiveAnalysisResult:
    """Complete mole analysis result."""
    # ML Classification
    melanoma_probability: float
    ml_prediction: str
    ml_confidence: float

    # ABCDE Analysis
    abcde_score: ABCDEScore

    # Combined Assessment
    combined_risk_level: str
    combined_risk_score: float

    # Recommendations
    recommendations: List[str]
    urgent_referral: bool

    # Metadata
    analysis_timestamp: str
    analysis_version: str = "1.0.0"


class MoleAnalysisService:
    """
    Comprehensive mole analysis service combining ML and ABCDE analysis.

    Usage:
        service = MoleAnalysisService(ml_model_path='model.h5')

        # Single image analysis
        result = service.analyze_mole(image_bytes)
        print(f"Risk Level: {result.combined_risk_level}")
        print(f"Melanoma Probability: {result.melanoma_probability:.1%}")

        # Compare two images over time
        comparison = service.compare_moles(image1, image2, days=30)
    """

    def __init__(
        self,
        ml_model_path: Optional[str] = None,
        use_derm_foundation: bool = False,
        pixels_per_mm: float = 10.0
    ):
        """
        Initialize the mole analysis service.

        Args:
            ml_model_path: Path to melanoma classification model
            use_derm_foundation: Whether to use Google Derm Foundation
            pixels_per_mm: Calibration for size estimation
        """
        self.ml_model = None
        self.use_derm_foundation = use_derm_foundation
        self.abcde_analyzer = ABCDEAnalyzer(pixels_per_mm=pixels_per_mm)

        if ml_model_path:
            self._load_ml_model(ml_model_path)

        if use_derm_foundation:
            self._load_derm_foundation()

    def _load_ml_model(self, model_path: str):
        """Load the melanoma classification model."""
        try:
            import tensorflow as tf
            self.ml_model = tf.keras.models.load_model(model_path)
            logger.info(f"Loaded ML model from {model_path}")
        except Exception as e:
            logger.warning(f"Could not load ML model: {e}")

    def _load_derm_foundation(self):
        """Load Google Derm Foundation model.

        Legacy: Google points new development at MedSigLIP. See the status note in
        `derm_foundation_service` and the README before extending this path.
        """
        try:
            from huggingface_hub import from_pretrained_keras
            self.derm_model = from_pretrained_keras("google/derm-foundation")
            self.derm_infer = self.derm_model.signatures["serving_default"]
            logger.info("Loaded Derm Foundation model")
        except Exception as e:
            logger.warning(f"Could not load Derm Foundation: {e}")
            self.use_derm_foundation = False

    def analyze_mole(
        self,
        image_input: Any,
        include_abcde: bool = True,
        include_ml: bool = True,
    ) -> ComprehensiveAnalysisResult:
        """
        Perform comprehensive mole analysis.

        Args:
            image_input: Image as bytes, base64 string, numpy array, or file path
            include_abcde: Include ABCDE symptom analysis
            include_ml: Include ML-based classification

        Returns:
            ComprehensiveAnalysisResult with all analysis data
        """
        # Load and preprocess image
        image = self._load_image(image_input)

        # ML Classification
        ml_result = {'probability': 0.5, 'prediction': 'Unknown', 'confidence': 0.0}
        if include_ml and (self.ml_model or self.use_derm_foundation):
            ml_result = self._run_ml_classification(image)

        # ABCDE Analysis
        abcde_score = ABCDEScore(
            asymmetry_score=0.0,
            border_score=0.0,
            color_score=0.0,
            diameter_mm=0.0,
            recommendations=[]
        )
        if include_abcde:
            abcde_score = self.abcde_analyzer.analyze(image)

        # Combine results
        combined_result = self._combine_assessments(ml_result, abcde_score)

        return ComprehensiveAnalysisResult(
            melanoma_probability=ml_result['probability'],
            ml_prediction=ml_result['prediction'],
            ml_confidence=ml_result['confidence'],
            abcde_score=abcde_score,
            combined_risk_level=combined_result['risk_level'],
            combined_risk_score=combined_result['risk_score'],
            recommendations=combined_result['recommendations'],
            urgent_referral=combined_result['urgent_referral'],
            analysis_timestamp=datetime.utcnow().isoformat(),
        )

    def compare_moles(
        self,
        earlier_image: Any,
        later_image: Any,
        time_delta_days: int = 30
    ) -> Dict[str, Any]:
        """
        Compare two images of the same mole over time.

        Args:
            earlier_image: First (earlier) image
            later_image: Second (later) image
            time_delta_days: Days between images

        Returns:
            Comparison results with evolution assessment
        """
        image1 = self._load_image(earlier_image)
        image2 = self._load_image(later_image)

        # Run comparison
        comparison = analyze_mole_comparison(image1, image2, time_delta_days)

        # Add ML predictions if available
        if self.ml_model or self.use_derm_foundation:
            ml1 = self._run_ml_classification(image1)
            ml2 = self._run_ml_classification(image2)

            comparison['earlier_ml'] = ml1
            comparison['later_ml'] = ml2
            comparison['ml_probability_change'] = ml2['probability'] - ml1['probability']

            # If ML prediction changed significantly, flag it
            if comparison['ml_probability_change'] > 0.15:
                comparison['significant_changes'].append('ml_probability_increase')
                comparison['evolution_message'] += (
                    f" ML-detected melanoma probability increased by "
                    f"{comparison['ml_probability_change']:.1%}."
                )

        return comparison

    def _load_image(self, image_input: Any) -> np.ndarray:
        """Load image from various input formats."""
        if isinstance(image_input, np.ndarray):
            return image_input

        if isinstance(image_input, bytes):
            nparr = np.frombuffer(image_input, np.uint8)
            return cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if isinstance(image_input, str):
            # Check if base64
            if len(image_input) > 200 and '/' not in image_input[:50]:
                try:
                    image_bytes = base64.b64decode(image_input)
                    nparr = np.frombuffer(image_bytes, np.uint8)
                    return cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                except Exception:
                    pass

            # Assume file path
            return cv2.imread(image_input)

        raise ValueError(f"Unsupported image input type: {type(image_input)}")

    def _run_ml_classification(self, image: np.ndarray) -> Dict[str, Any]:
        """Run ML-based melanoma classification."""
        if self.use_derm_foundation:
            return self._classify_with_derm_foundation(image)
        elif self.ml_model:
            return self._classify_with_local_model(image)
        return {'probability': 0.5, 'prediction': 'Unknown', 'confidence': 0.0}

    def _classify_with_local_model(self, image: np.ndarray) -> Dict[str, Any]:
        """Classify using local ML model."""
        import tensorflow as tf

        # Get model input shape
        input_shape = self.ml_model.input_shape[1:3]

        # Preprocess
        img = cv2.resize(image, input_shape)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        img = np.expand_dims(img, axis=0)

        # Predict
        prediction = self.ml_model.predict(img, verbose=0)[0, 0]

        # Model outputs P(NotMelanoma), so invert for P(Melanoma)
        melanoma_prob = 1 - prediction

        return {
            'probability': float(melanoma_prob),
            'prediction': 'Melanoma' if melanoma_prob >= 0.5 else 'NotMelanoma',
            'confidence': float(max(melanoma_prob, 1 - melanoma_prob))
        }

    def _classify_with_derm_foundation(self, image: np.ndarray) -> Dict[str, Any]:
        """Classify using Derm Foundation embeddings."""
        import tensorflow as tf

        # Convert to PIL and resize
        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_rgb)
        pil_img = pil_img.resize((448, 448), Image.Resampling.LANCZOS)

        # Convert to PNG bytes
        buf = BytesIO()
        pil_img.save(buf, 'PNG')
        image_bytes = buf.getvalue()

        # Create TFRecord input
        input_tensor = tf.train.Example(
            features=tf.train.Features(
                feature={
                    'image/encoded': tf.train.Feature(
                        bytes_list=tf.train.BytesList(value=[image_bytes])
                    )
                }
            )
        ).SerializeToString()

        # Get embedding
        output = self.derm_infer(inputs=tf.constant([input_tensor]))
        embedding = output['embedding'].numpy().flatten()

        # Note: You would need to train a classifier on top of embeddings
        # For now, return placeholder
        logger.warning("Derm Foundation classifier not trained. Using placeholder.")
        return {
            'probability': 0.5,
            'prediction': 'Unknown',
            'confidence': 0.0,
            'embedding_extracted': True
        }

    def _combine_assessments(
        self,
        ml_result: Dict[str, Any],
        abcde_score: ABCDEScore
    ) -> Dict[str, Any]:
        """
        Combine ML and ABCDE assessments into final risk evaluation.

        Uses a weighted combination that prioritizes:
        1. High ML probability
        2. Multiple ABCDE warning signs
        3. Concordance between methods
        """
        # ML weight depends on confidence
        ml_weight = 0.5 * ml_result['confidence']
        abcde_weight = 0.5

        # Calculate combined risk score
        ml_risk = ml_result['probability']
        abcde_risk = abcde_score.total_score

        # If both methods agree on high risk, increase score
        if ml_risk > 0.6 and abcde_risk > 0.5:
            concordance_bonus = 0.1
        elif ml_risk < 0.4 and abcde_risk < 0.3:
            concordance_bonus = -0.05
        else:
            concordance_bonus = 0

        combined_score = (
            ml_risk * ml_weight +
            abcde_risk * abcde_weight +
            concordance_bonus
        )
        combined_score = min(max(combined_score, 0), 1)

        # Determine risk level
        if combined_score >= 0.65 or ml_risk >= 0.8:
            risk_level = "very_high"
            urgent = True
        elif combined_score >= 0.45 or ml_risk >= 0.6:
            risk_level = "high"
            urgent = False
        elif combined_score >= 0.3:
            risk_level = "moderate"
            urgent = False
        else:
            risk_level = "low"
            urgent = False

        # Generate combined recommendations
        recommendations = list(abcde_score.recommendations or [])

        if ml_risk >= 0.7:
            recommendations.insert(0,
                f"ML ALERT: The AI model detected a {ml_risk:.0%} probability of melanoma. "
                "This warrants professional evaluation."
            )
        elif ml_risk >= 0.5:
            recommendations.insert(0,
                f"The AI model indicates elevated melanoma risk ({ml_risk:.0%}). "
                "Consider consulting a dermatologist."
            )

        if risk_level == "very_high":
            recommendations.insert(0,
                "PRIORITY: Multiple risk factors detected. Please seek dermatological "
                "evaluation within 1-2 weeks."
            )

        return {
            'risk_level': risk_level,
            'risk_score': combined_score,
            'recommendations': recommendations,
            'urgent_referral': urgent,
        }

    def to_api_response(self, result: ComprehensiveAnalysisResult) -> Dict[str, Any]:
        """Convert analysis result to API response format."""
        return {
            'success': True,
            'analysis': {
                'ml_classification': {
                    'melanoma_probability': result.melanoma_probability,
                    'prediction': result.ml_prediction,
                    'confidence': result.ml_confidence,
                },
                'abcde_analysis': {
                    'asymmetry': {
                        'score': result.abcde_score.asymmetry_score,
                        'risk': result.abcde_score.asymmetry_risk,
                    },
                    'border': {
                        'score': result.abcde_score.border_score,
                        'risk': result.abcde_score.border_risk,
                    },
                    'color': {
                        'score': result.abcde_score.color_score,
                        'risk': result.abcde_score.color_risk,
                    },
                    'diameter_mm': result.abcde_score.diameter_mm,
                    'diameter_risk': result.abcde_score.diameter_risk,
                },
                'combined_assessment': {
                    'risk_level': result.combined_risk_level,
                    'risk_score': result.combined_risk_score,
                    'urgent_referral': result.urgent_referral,
                },
                'recommendations': result.recommendations,
            },
            'metadata': {
                'timestamp': result.analysis_timestamp,
                'version': result.analysis_version,
            }
        }


# Convenience function for API integration
def analyze_mole_image(
    image_base64: str,
    model_path: Optional[str] = None
) -> Dict[str, Any]:
    """
    Analyze a mole image and return API-ready response.

    Args:
        image_base64: Base64 encoded image
        model_path: Optional path to ML model

    Returns:
        API response dictionary
    """
    service = MoleAnalysisService(ml_model_path=model_path)
    result = service.analyze_mole(image_base64)
    return service.to_api_response(result)
