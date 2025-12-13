import warnings
warnings.filterwarnings('ignore')
import time
import cv2
import numpy as np
from insightface.app import FaceAnalysis
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Optional, Tuple, Dict, Any
import io
import os
import traceback
import warnings

warnings.filterwarnings('ignore')


class FaceService:
    def __init__(self, model_name='buffalo_l', det_thresh=0.35, det_size=(640, 640)):
        """
        Initialize FaceService with robust error handling and automatic model download

        Args:
            model_name: Model to use ('buffalo_l', 'buffalo_s', 'antelopev2')
            det_thresh: Detection threshold (0.35 is good balance)
            det_size: Detection size for optimal performance
        """
        self.model_name = model_name
        self.det_thresh = det_thresh
        self.det_size = det_size
        self.app = None
        self.initialized = False
        self.model_loaded = False

        print(f"🚀 Initializing FaceService with model: {model_name}")
        self._initialize()

    def _ensure_model_directory(self):
        """Create model directory if it doesn't exist"""
        model_dir = os.path.expanduser('~/.insightface/models')
        os.makedirs(model_dir, exist_ok=True)
        print(f"📁 Model directory: {model_dir}")
        return model_dir

    def _download_model_if_needed(self):
        """Download model if not available locally"""
        try:
            from insightface.model_zoo import get_model
            print(f"📦 Checking for model: {self.model_name}")

            # Try to get the model (will download if not exists)
            model = get_model(self.model_name, download=True)
            print(f"✅ Model {self.model_name} is available")
            return True
        except Exception as e:
            print(f"⚠️ Model check failed: {e}")
            return False

    def _initialize(self):
        """Initialize face analysis with multiple fallback strategies"""
        max_retries = 2

        for attempt in range(max_retries):
            try:
                print(f"\n🔄 Initialization attempt {attempt + 1}/{max_retries}")

                # Ensure model directory exists
                self._ensure_model_directory()

                # Try to download model
                if not self._download_model_if_needed():
                    print("⚠️ Model download may have failed, trying to initialize anyway")

                # Initialize FaceAnalysis
                print(f"🔧 Creating FaceAnalysis instance...")
                self.app = FaceAnalysis(
                    name=self.model_name,
                    providers=['CPUExecutionProvider'],
                    root='~/.insightface/models'
                )

                # Prepare the model
                print(f"🔧 Preparing model with det_thresh={self.det_thresh}, det_size={self.det_size}")
                self.app.prepare(
                    ctx_id=-1,  # CPU
                    det_thresh=self.det_thresh,
                    det_size=self.det_size
                )

                # Test with a simple image
                print("🧪 Testing model with synthetic image...")
                test_img = np.random.randint(100, 200, (300, 300, 3), dtype=np.uint8)
                test_faces = self.app.get(test_img)
                print(f"✅ Model test passed. Detected {len(test_faces)} faces in test")

                self.initialized = True
                self.model_loaded = True
                print(f"🎉 FaceService initialized successfully!")
                return

            except Exception as e:
                print(f"❌ Attempt {attempt + 1} failed: {str(e)}")
                if attempt < max_retries - 1:
                    print("🔄 Retrying with different parameters...")
                    # Try smaller model and size on retry
                    if self.model_name == 'buffalo_l':
                        self.model_name = 'buffalo_s'
                        self.det_size = (320, 320)
                    time.sleep(2)
                else:
                    print(f"💥 All initialization attempts failed")
                    traceback.print_exc()

    def is_initialized(self) -> bool:
        """Check if service is properly initialized"""
        return self.initialized and self.app is not None

    def load_image_from_bytes(self, image_bytes: bytes) -> Optional[np.ndarray]:
        """Load and preprocess image from bytes"""
        try:
            if not image_bytes or len(image_bytes) < 100:
                print("❌ Image data too small")
                return None

            # Method 1: OpenCV
            nparr = np.frombuffer(image_bytes, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            if img is None:
                print("⚠️ OpenCV failed, trying PIL fallback...")
                try:
                    from PIL import Image
                    pil_image = Image.open(io.BytesIO(image_bytes))
                    # Convert to RGB
                    if pil_image.mode != 'RGB':
                        pil_image = pil_image.convert('RGB')
                    img = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
                    print("✅ PIL fallback succeeded")
                except Exception as pil_error:
                    print(f"❌ PIL fallback failed: {pil_error}")
                    return None

            # Convert BGR to RGB (insightface expects RGB)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # Validate image
            if img_rgb is None or len(img_rgb.shape) != 3:
                print("❌ Invalid image format")
                return None

            # Resize if too large for performance
            height, width = img_rgb.shape[:2]
            max_dimension = 2000

            if max(height, width) > max_dimension:
                scale = max_dimension / max(height, width)
                new_width = int(width * scale)
                new_height = int(height * scale)
                img_rgb = cv2.resize(img_rgb, (new_width, new_height))
                print(f"📐 Resized from {width}x{height} to {new_width}x{new_height}")

            # Enhance contrast for better detection
            lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            l = clahe.apply(l)
            img_rgb = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2RGB)

            return img_rgb

        except Exception as e:
            print(f"❌ Error loading image: {e}")
            return None

    def detect_faces(self, image_rgb: np.ndarray, image_name: str = "image") -> List:
        """Detect faces in image"""
        if not self.is_initialized():
            print("❌ FaceService not initialized")
            return []

        try:
            print(f"\n🔍 Detecting faces in {image_name}...")

            # Check image quality
            gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
            brightness = np.mean(gray)
            contrast = np.std(gray)

            print(f"📊 Image stats - Size: {image_rgb.shape}, Brightness: {brightness:.1f}, Contrast: {contrast:.1f}")

            # Adjust detection based on image quality
            current_thresh = self.det_thresh
            if brightness < 50 or contrast < 30:
                current_thresh = max(0.2, current_thresh - 0.1)  # Be more lenient
                print(f"⚙️ Adjusting threshold to {current_thresh} due to poor image quality")

            # Detect faces
            start_time = time.time()

            # Try with current threshold
            faces = self.app.get(image_rgb)

            # If no faces found, try with lower threshold
            if len(faces) == 0 and current_thresh > 0.2:
                print("🔄 No faces found, trying with lower threshold...")
                # Temporarily adjust threshold
                self.app.det_thresh = 0.2
                faces = self.app.get(image_rgb)
                self.app.det_thresh = current_thresh  # Restore original

            detection_time = time.time() - start_time

            print(f"✅ Detection completed in {detection_time:.3f}s")
            print(f"👥 Faces found: {len(faces)}")

            # Log confidence scores
            for i, face in enumerate(faces):
                if hasattr(face, 'det_score'):
                    bbox = [int(x) for x in face.bbox]
                    print(f"  Face {i + 1}: bbox={bbox}, confidence={face.det_score:.3f}")

            return faces

        except Exception as e:
            print(f"❌ Error in face detection: {e}")
            traceback.print_exc()
            return []

    def get_face_embeddings(self, faces: List) -> List[np.ndarray]:
        """Extract embeddings from detected faces"""
        embeddings = []
        for i, face in enumerate(faces):
            if hasattr(face, 'embedding'):
                embeddings.append(face.embedding)
            else:
                print(f"⚠️ Face {i + 1} has no embedding")
        return embeddings

    def get_face_bounding_boxes(self, faces: List) -> List[Dict[str, Any]]:
        """Get bounding boxes with metadata"""
        boxes = []
        for i, face in enumerate(faces):
            if hasattr(face, 'bbox'):
                x1, y1, x2, y2 = [int(coord) for coord in face.bbox]
                box_info = {
                    'index': i,
                    'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2,
                    'width': x2 - x1,
                    'height': y2 - y1,
                    'area': (x2 - x1) * (y2 - y1)
                }

                if hasattr(face, 'det_score'):
                    box_info['confidence'] = float(face.det_score)

                if hasattr(face, 'kps'):
                    box_info['landmarks'] = face.kps.tolist()

                boxes.append(box_info)
        return boxes

    def analyze_face_quality(self, face, image_rgb: np.ndarray) -> Dict[str, Any]:
        """Analyze face quality for registration"""
        try:
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

            # 1. Size check
            face_height, face_width = face_region.shape[:2]
            min_face_size = 80
            if face_width < min_face_size or face_height < min_face_size:
                return {'valid': False, 'reason': f'Face too small ({face_width}x{face_height})'}

            # 2. Brightness check
            brightness = np.mean(gray_face)
            if brightness < 40 or brightness > 220:
                return {'valid': False, 'reason': f'Poor brightness ({brightness:.1f})'}

            # 3. Blur check
            blur = cv2.Laplacian(gray_face, cv2.CV_64F).var()
            if blur < 50:
                return {'valid': False, 'reason': f'Face too blurry (score: {blur:.1f})'}

            # 4. Aspect ratio check (should be roughly square)
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

    def process_single_image(self, image_bytes: bytes, require_single_face: bool = True) -> Dict[str, Any]:
        """Process image and return face data"""
        result = {
            "success": False,
            "faces_detected": 0,
            "embeddings": [],
            "bounding_boxes": [],
            "face_qualities": [],
            "best_face_index": -1,
            "error": None,
            "warnings": []
        }

        try:
            # Check initialization
            if not self.is_initialized():
                result["error"] = "Face detection service not initialized"
                return result

            # Load image
            image_rgb = self.load_image_from_bytes(image_bytes)
            if image_rgb is None:
                result["error"] = "Failed to load image. Please check image format (JPEG/PNG)"
                return result

            # Detect faces
            faces = self.detect_faces(image_rgb)

            if not faces:
                result[
                    "error"] = "No faces detected. Please ensure: 1) Face is clearly visible, 2) Good lighting, 3) Front-facing photo"
                return result

            result["faces_detected"] = len(faces)

            # Check if single face is required
            if require_single_face and len(faces) != 1:
                result[
                    "error"] = f"Expected exactly 1 face, found {len(faces)}. Please upload a photo with only one person."
                # Still process to show what was found
                result["warnings"].append(f"Found {len(faces)} faces")

            # Extract embeddings
            embeddings = self.get_face_embeddings(faces)
            if not embeddings:
                result["error"] = "Failed to extract face features"
                return result

            # Get bounding boxes
            bounding_boxes = self.get_face_bounding_boxes(faces)

            # Analyze face quality
            best_face_index = -1
            best_quality_score = -1

            for i, (face, embedding) in enumerate(zip(faces, embeddings)):
                quality = self.analyze_face_quality(face, image_rgb)
                result["face_qualities"].append(quality)

                # Calculate quality score
                if quality.get('valid', False):
                    # Simple scoring based on blur and size
                    score = quality.get('blur_score', 0) / 100
                    if quality.get('blur_score', 0) > 100:
                        score = 1.0

                    if score > best_quality_score:
                        best_quality_score = score
                        best_face_index = i
                else:
                    result["warnings"].append(f"Face {i + 1}: {quality.get('reason', 'Quality check failed')}")

            # Convert embeddings to lists for JSON
            embeddings_list = [embedding.tolist() for embedding in embeddings]

            # Update result
            result.update({
                "success": True,
                "faces_detected": len(faces),
                "embeddings": embeddings_list,
                "bounding_boxes": bounding_boxes,
                "best_face_index": best_face_index
            })

            # Add success message
            if len(faces) == 1:
                print(f"✅ Successfully processed image with 1 face")
            else:
                print(f"✅ Processed image with {len(faces)} faces")

        except Exception as e:
            result["error"] = f"Processing error: {str(e)}"
            print(f"❌ Error in process_single_image: {e}")
            traceback.print_exc()

        return result

    def compare_faces(self, embedding1, embedding2, threshold: float = 0.6) -> Tuple[float, bool]:
        """Compare two face embeddings"""
        try:
            if embedding1 is None or embedding2 is None:
                return 0.0, False

            # Ensure embeddings are numpy arrays
            emb1 = np.array(embedding1).reshape(1, -1)
            emb2 = np.array(embedding2).reshape(1, -1)

            # Check dimensions
            if emb1.shape[1] != emb2.shape[1]:
                print(f"⚠️ Embedding dimension mismatch: {emb1.shape[1]} vs {emb2.shape[1]}")
                return 0.0, False

            # Calculate similarity
            similarity = cosine_similarity(emb1, emb2)[0][0]
            is_match = similarity > threshold

            print(f"🔗 Similarity: {similarity:.3f}, Match: {is_match} (threshold: {threshold})")
            return similarity, is_match

        except Exception as e:
            print(f"❌ Error comparing faces: {e}")
            return 0.0, False

    def find_best_match(self, query_embedding: List[float], user_embeddings: List[List[float]],
                        threshold: float = 0.6) -> Tuple[float, bool]:
        """
        Find best match between query embedding and multiple user embeddings

        Args:
            query_embedding: Embedding from attendance image
            user_embeddings: List of embeddings stored for a user
            threshold: Minimum similarity threshold

        Returns:
            Tuple of (best_similarity, is_match)
        """
        best_similarity = 0.0

        for user_emb in user_embeddings:
            similarity, _ = self.compare_faces(query_embedding, user_emb, threshold=0.0)
            if similarity > best_similarity:
                best_similarity = similarity

        return best_similarity, best_similarity > threshold

    def batch_compare_faces(self, query_embeddings: List[List[float]],
                            user_embeddings_dict: Dict[str, List[List[float]]],
                            threshold: float = 0.6) -> Dict[str, List[Tuple[int, float]]]:
        """
        Compare multiple query embeddings with multiple users

        Returns:
            Dict with user_id -> list of (face_index, similarity)
        """
        matches = {}

        for user_id, user_embeddings in user_embeddings_dict.items():
            user_matches = []

            for face_idx, query_emb in enumerate(query_embeddings):
                similarity, is_match = self.find_best_match(query_emb, user_embeddings, threshold)
                if is_match:
                    user_matches.append((face_idx, similarity))

            if user_matches:
                matches[user_id] = user_matches

        return matches

    def draw_faces_on_image(self, image_bytes: bytes, output_path: str = "debug_faces.jpg"):
        """Draw detected faces on image for debugging"""
        try:
            image_rgb = self.load_image_from_bytes(image_bytes)
            if image_rgb is None:
                return False

            # Create copy for drawing
            output_img = image_rgb.copy()

            # Detect faces
            faces = self.detect_faces(image_rgb, "debug_image")

            # Draw bounding boxes
            for i, face in enumerate(faces):
                if hasattr(face, 'bbox'):
                    x1, y1, x2, y2 = [int(coord) for coord in face.bbox]

                    # Draw rectangle
                    color = (0, 255, 0)  # Green
                    thickness = 2
                    cv2.rectangle(output_img, (x1, y1), (x2, y2), color, thickness)

                    # Add face number
                    cv2.putText(output_img, f"Face {i + 1}", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

            # Save image
            output_img_bgr = cv2.cvtColor(output_img, cv2.COLOR_RGB2BGR)
            cv2.imwrite(output_path, output_img_bgr)
            print(f"📸 Debug image saved: {output_path}")
            return True

        except Exception as e:
            print(f"❌ Failed to draw faces: {e}")
            return False


# Create global instance
print("\n" + "=" * 50)
print("INITIALIZING FACE SERVICE")
print("=" * 50)

try:
    # Initialize with slightly lower threshold for better detection
    face_service = FaceService(
        model_name='buffalo_s',  # Smaller, faster model
        det_thresh=0.35,  # Balanced threshold
        det_size=(640, 640)
    )

    if not face_service.is_initialized():
        print("\n⚠️ WARNING: FaceService initialization had issues")
        print("Troubleshooting steps:")
        print("1. Check internet connection (for model download)")
        print("2. Run: pip install --upgrade insightface")
        print("3. Delete folder: ~/.insightface/models and restart")
    else:
        print("\n✅ FaceService ready for use!")

except Exception as e:
    print(f"\n💥 CRITICAL: Failed to initialize FaceService: {e}")
    print("The application may not work properly.")
    traceback.print_exc()


# Test function
def test_face_service():
    """Test the face service"""
    print("\n🧪 Running FaceService test...")

    if not face_service.is_initialized():
        print("❌ Service not initialized, test skipped")
        return False

    # Create a simple test image
    test_img = np.ones((500, 500, 3), dtype=np.uint8) * 200

    # Draw a simple face pattern
    cv2.circle(test_img, (250, 250), 100, (150, 150, 150), -1)  # Head
    cv2.circle(test_img, (200, 220), 20, (50, 50, 50), -1)  # Left eye
    cv2.circle(test_img, (300, 220), 20, (50, 50, 50), -1)  # Right eye
    cv2.ellipse(test_img, (250, 280), (50, 25), 0, 0, 180, (50, 50, 50), 10)  # Mouth

    # Convert to bytes
    success, buffer = cv2.imencode('.jpg', test_img)
    if success:
        test_bytes = buffer.tobytes()

        # Test processing
        result = face_service.process_single_image(test_bytes)
        print(f"Test result: Success={result['success']}, Faces={result['faces_detected']}")

        # Draw debug image
        face_service.draw_faces_on_image(test_bytes, "test_output.jpg")

        return result['success']

    return False


if __name__ == "__main__":
    # Run test if executed directly
    test_result = test_face_service()
    print(f"\n🎯 FaceService test {'PASSED' if test_result else 'FAILED'}")

# Create global instance
print("\n" + "=" * 50)
print("INITIALIZING FACE SERVICE")
print("=" * 50)

try:
    face_service = FaceService(
        model_name='buffalo_s',
        det_thresh=0.35,
        det_size=(640, 640)
    )

    if not face_service.is_initialized():
        print("\n⚠️ WARNING: FaceService initialization had issues")
    else:
        print("\n✅ FaceService ready for use!")

except Exception as e:
    print(f"\n💥 CRITICAL: Failed to initialize FaceService: {e}")
    traceback.print_exc()