import time
import logging
from typing import List, Optional, Tuple, Dict, Any
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import io

# Dynamic imports with fallbacks
try:
    import cv2

    CV2_AVAILABLE = True
    logger = logging.getLogger(__name__)
    logger.info("OpenCV loaded successfully")
except ImportError as e:
    CV2_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.error(f"OpenCV not available: {e}")

try:
    import numpy as np

    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    logger.error("NumPy not available")

try:
    from sklearn.metrics.pairwise import cosine_similarity

    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logger.error("scikit-learn not available")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)


class ModelType(str, Enum):
    """Supported model types"""
    BUFFALO_L = 'buffalo_l'
    BUFFALO_S = 'buffalo_s'


class ServiceStatus(str, Enum):
    """Service status enumeration"""
    INITIALIZING = "initializing"
    READY = "ready"
    ERROR = "error"
    DEGRADED = "degraded"


@dataclass
class ServiceConfig:
    """Configuration for FaceService"""
    model_name: ModelType = ModelType.BUFFALO_S
    detection_threshold: float = 0.35
    detection_size: Tuple[int, int] = (640, 640)
    recognition_threshold: float = 0.6
    min_face_size: int = 80
    max_image_dimension: int = 2000
    providers: List[str] = field(default_factory=lambda: ['CPUExecutionProvider'])
    model_cache_dir: str = "/tmp/.insightface/models"
    enable_quality_check: bool = True
    max_retries: int = 3
    retry_delay: float = 2.0


@dataclass
class FaceDetectionResult:
    """Structured result for face detection"""
    success: bool
    faces_detected: int
    embeddings: List[List[float]]
    bounding_boxes: List[Dict[str, Any]]
    face_qualities: List[Dict[str, Any]]
    best_face_index: int = -1
    processing_time_ms: float = 0.0
    error_message: Optional[str] = None
    warnings: List[str] = field(default_factory=list)


@dataclass
class FaceComparisonResult:
    """Structured result for face comparison"""
    similarity_score: float
    is_match: bool
    threshold: float
    processing_time_ms: float = 0.0


class FaceServiceError(Exception):
    """Custom exception for FaceService errors"""
    pass


class FaceService:
    """
    Production-ready face recognition service
    """

    def __init__(self, config: Optional[ServiceConfig] = None):
        """
        Initialize FaceService with lazy loading of heavy dependencies.
        """
        self.config = config or ServiceConfig()
        self.app = None
        self.status = ServiceStatus.INITIALIZING
        self._insightface_available = False

        # Check basic dependencies
        self._check_dependencies()

        # Start initialization
        self._initialize()

    def _check_dependencies(self) -> bool:
        """Check if all required dependencies are available"""
        if not CV2_AVAILABLE:
            logger.error("OpenCV is not available. Please install opencv-python-headless")
            self.status = ServiceStatus.ERROR
            return False

        if not NUMPY_AVAILABLE:
            logger.error("NumPy is not available")
            self.status = ServiceStatus.ERROR
            return False

        if not SKLEARN_AVAILABLE:
            logger.error("scikit-learn is not available")
            self.status = ServiceStatus.ERROR
            return False

        return True

    def _initialize(self):
        """Initialize the face analysis service"""
        try:
            logger.info(f"🚀 Initializing FaceService with model: {self.config.model_name}")

            # Try to import insightface
            try:
                from insightface.app import FaceAnalysis
                self._insightface_available = True
                logger.info("InsightFace imported successfully")
            except ImportError as e:
                logger.error(f"Failed to import InsightFace: {e}")
                self.status = ServiceStatus.ERROR
                return

            # Ensure model directory exists
            model_dir = Path(self.config.model_cache_dir)
            model_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"Model directory: {model_dir}")

            # Initialize with retries
            for attempt in range(self.config.max_retries):
                try:
                    logger.info(f"Attempt {attempt + 1}/{self.config.max_retries}")

                    self.app = FaceAnalysis(
                        name=self.config.model_name.value,
                        providers=self.config.providers,
                        root=str(model_dir)
                    )

                    self.app.prepare(
                        ctx_id=-1,  # CPU
                        det_thresh=self.config.detection_threshold,
                        det_size=self.config.detection_size
                    )

                    # Simple test
                    test_img = np.ones((100, 100, 3), dtype=np.uint8) * 128
                    _ = self.app.get(test_img)

                    self.status = ServiceStatus.READY
                    logger.info("✅ FaceService initialized successfully")
                    return

                except Exception as e:
                    logger.error(f"Attempt {attempt + 1} failed: {e}")

                    if attempt < self.config.max_retries - 1:
                        time.sleep(self.config.retry_delay)
                        # Try smaller model on retry
                        if self.config.model_name == ModelType.BUFFALO_L:
                            self.config.model_name = ModelType.BUFFALO_S
                    else:
                        logger.error("All initialization attempts failed")
                        self.status = ServiceStatus.ERROR

        except Exception as e:
            logger.error(f"Critical initialization error: {e}")
            self.status = ServiceStatus.ERROR

    def is_ready(self) -> bool:
        """Check if service is ready to process requests"""
        return self.status == ServiceStatus.READY and self.app is not None

    def load_image(self, image_bytes: bytes) -> Optional[np.ndarray]:
        """
        Load and preprocess image from bytes.

        Args:
            image_bytes: Image data as bytes

        Returns:
            Preprocessed RGB image as numpy array
        """
        try:
            if not image_bytes or len(image_bytes) < 100:
                logger.error("Image data too small")
                return None

            # Convert bytes to numpy array
            nparr = np.frombuffer(image_bytes, np.uint8)

            # Decode image
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if img is None:
                # Try PIL as fallback
                try:
                    from PIL import Image
                    pil_image = Image.open(io.BytesIO(image_bytes))
                    if pil_image.mode != 'RGB':
                        pil_image = pil_image.convert('RGB')
                    img = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
                    logger.debug("Used PIL fallback for image loading")
                except Exception as pil_error:
                    logger.error(f"PIL fallback failed: {pil_error}")
                    return None

            # Convert to RGB
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # Resize if too large
            h, w = img_rgb.shape[:2]
            if max(h, w) > self.config.max_image_dimension:
                scale = self.config.max_image_dimension / max(h, w)
                new_w = int(w * scale)
                new_h = int(h * scale)
                img_rgb = cv2.resize(img_rgb, (new_w, new_h))
                logger.debug(f"Resized image from {w}x{h} to {new_w}x{new_h}")

            return img_rgb

        except Exception as e:
            logger.error(f"Error loading image: {e}")
            return None

    def _analyze_face_quality(self, face, image_rgb: np.ndarray) -> Dict[str, Any]:
        """Analyze quality of a single face"""
        try:
            if not hasattr(face, 'bbox'):
                return {'valid': False, 'reason': 'No bounding box'}

            x1, y1, x2, y2 = [int(coord) for coord in face.bbox]

            # Ensure coordinates are within bounds
            h, w = image_rgb.shape[:2]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)

            if x2 <= x1 or y2 <= y1:
                return {'valid': False, 'reason': 'Invalid bounding box'}

            # Extract face region
            face_region = image_rgb[y1:y2, x1:x2]
            if face_region.size == 0:
                return {'valid': False, 'reason': 'Empty face region'}

            # Calculate quality metrics
            gray_face = cv2.cvtColor(face_region, cv2.COLOR_RGB2GRAY)

            # Size check
            face_height, face_width = face_region.shape[:2]
            if face_width < self.config.min_face_size or face_height < self.config.min_face_size:
                return {'valid': False, 'reason': f'Face too small ({face_width}x{face_height})'}

            # Brightness check
            brightness = np.mean(gray_face)
            if brightness < 40 or brightness > 220:
                return {'valid': False, 'reason': f'Poor brightness ({brightness:.1f})'}

            # Blur check
            blur = cv2.Laplacian(gray_face, cv2.CV_64F).var()
            if blur < 50:
                return {'valid': False, 'reason': f'Face too blurry (score: {blur:.1f})'}

            # Aspect ratio check
            aspect_ratio = face_width / face_height
            if aspect_ratio < 0.7 or aspect_ratio > 1.3:
                return {'valid': False, 'reason': f'Poor aspect ratio ({aspect_ratio:.2f})'}

            return {
                'valid': True,
                'size': f"{face_width}x{face_height}",
                'brightness': float(brightness),
                'blur_score': float(blur),
                'aspect_ratio': float(aspect_ratio)
            }

        except Exception as e:
            return {'valid': False, 'reason': f'Quality check error: {str(e)}'}

    def process_image(self, image_bytes: bytes, require_single_face: bool = True) -> FaceDetectionResult:
        """
        Process an image and detect faces.

        Args:
            image_bytes: Image data as bytes
            require_single_face: If True, will return error if multiple faces detected

        Returns:
            FaceDetectionResult with structured results
        """
        start_time = time.time()
        result = FaceDetectionResult(
            success=False,
            faces_detected=0,
            embeddings=[],
            bounding_boxes=[],
            face_qualities=[],
            best_face_index=-1
        )

        try:
            # Check service readiness
            if not self.is_ready():
                result.error_message = "FaceService is not ready"
                return result

            # Load image
            image_rgb = self.load_image(image_bytes)
            if image_rgb is None:
                result.error_message = "Failed to load image"
                return result

            # Detect faces
            faces = self.app.get(image_rgb)
            result.faces_detected = len(faces)

            if not faces:
                result.error_message = (
                    "No faces detected. Please ensure:\n"
                    "1. Face is clearly visible\n"
                    "2. Good lighting conditions\n"
                    "3. Front-facing photo\n"
                    "4. No obstructions"
                )
                return result

            # Check face count requirement
            if require_single_face and len(faces) != 1:
                result.error_message = f"Expected exactly 1 face, found {len(faces)}"
                result.warnings.append(f"Found {len(faces)} faces")
                # Continue processing to provide feedback

            # Process each face
            best_face_index = -1
            best_quality_score = -1

            for i, face in enumerate(faces):
                # Get embedding
                if hasattr(face, 'embedding'):
                    result.embeddings.append(face.embedding.tolist())

                # Get bounding box
                if hasattr(face, 'bbox'):
                    x1, y1, x2, y2 = [int(coord) for coord in face.bbox]
                    bbox_info = {
                        'index': i,
                        'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2,
                        'width': x2 - x1,
                        'height': y2 - y1
                    }
                    if hasattr(face, 'det_score'):
                        bbox_info['confidence'] = float(face.det_score)
                    result.bounding_boxes.append(bbox_info)

                # Analyze quality
                if self.config.enable_quality_check:
                    quality = self._analyze_face_quality(face, image_rgb)
                    result.face_qualities.append(quality)

                    # Find best face based on quality
                    if quality.get('valid', False):
                        blur_score = quality.get('blur_score', 0)
                        if blur_score > best_quality_score:
                            best_quality_score = blur_score
                            best_face_index = i
                else:
                    result.face_qualities.append({'valid': True})
                    best_face_index = 0

            result.success = True
            result.best_face_index = best_face_index
            result.processing_time_ms = (time.time() - start_time) * 1000

            logger.info(f"Processed image: {len(faces)} face(s) in {result.processing_time_ms:.1f}ms")

            return result

        except Exception as e:
            logger.error(f"Image processing error: {e}")
            result.error_message = str(e)
            return result

    def compare_faces(self, embedding1: List[float],
                      embedding2: List[float],
                      threshold: Optional[float] = None) -> FaceComparisonResult:
        """
        Compare two face embeddings.

        Args:
            embedding1: First face embedding
            embedding2: Second face embedding
            threshold: Similarity threshold. If None, uses config threshold

        Returns:
            FaceComparisonResult with similarity score and match status
        """
        start_time = time.time()

        try:
            if threshold is None:
                threshold = self.config.recognition_threshold

            # Convert to numpy arrays
            emb1 = np.array(embedding1, dtype=np.float32).reshape(1, -1)
            emb2 = np.array(embedding2, dtype=np.float32).reshape(1, -1)

            # Calculate similarity
            similarity = cosine_similarity(emb1, emb2)[0][0]
            is_match = similarity > threshold

            processing_time = (time.time() - start_time) * 1000

            return FaceComparisonResult(
                similarity_score=float(similarity),
                is_match=is_match,
                threshold=threshold,
                processing_time_ms=processing_time
            )

        except Exception as e:
            logger.error(f"Face comparison error: {e}")
            raise FaceServiceError(f"Face comparison failed: {str(e)}")


# Singleton instance
_face_service_instance = None


def get_face_service() -> FaceService:
    """
    Get or create singleton FaceService instance.

    Returns:
        FaceService instance
    """
    global _face_service_instance

    if _face_service_instance is None:
        try:
            config = ServiceConfig(
                model_name=ModelType.BUFFALO_S,
                detection_threshold=0.35,
                detection_size=(640, 640),
                recognition_threshold=0.6,
                model_cache_dir="/tmp/.insightface/models"
            )
            _face_service_instance = FaceService(config)
        except Exception as e:
            logger.error(f"Failed to create FaceService: {e}")
            _face_service_instance = None

    return _face_service_instance