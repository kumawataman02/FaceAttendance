import os
import traceback
from typing import List, Optional, Tuple, Dict, Any

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from insightface.app import FaceAnalysis


# =========================
# Lazy OpenCV loader
# =========================
def get_cv2():
    import cv2
    return cv2


# =========================
# Face Service
# =========================
class FaceService:
    def __init__(
        self,
        model_name: str = "buffalo_s",
        det_thresh: float = 0.35,
        det_size: Tuple[int, int] = (640, 640),
    ):
        self.model_name = model_name
        self.det_thresh = det_thresh
        self.det_size = det_size

        self.app: Optional[FaceAnalysis] = None
        self.initialized: bool = False

    # -------------------------
    # Lazy initialization
    # -------------------------
    def initialize(self) -> bool:
        if self.initialized:
            return True

        try:
            model_dir = os.path.expanduser("~/.insightface/models")
            os.makedirs(model_dir, exist_ok=True)

            self.app = FaceAnalysis(
                name=self.model_name,
                providers=["CPUExecutionProvider"],
                root=model_dir,
            )

            self.app.prepare(
                ctx_id=-1,
                det_thresh=self.det_thresh,
                det_size=self.det_size,
            )

            # lightweight sanity check
            test_img = np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8)
            self.app.get(test_img)

            self.initialized = True
            return True

        except Exception:
            traceback.print_exc()
            self.initialized = False
            return False

    # -------------------------
    # Helpers
    # -------------------------
    def _ensure_ready(self) -> bool:
        if not self.initialized:
            return self.initialize()
        return True

    # -------------------------
    # Image loader
    # -------------------------
    def load_image_from_bytes(self, image_bytes: bytes) -> Optional[np.ndarray]:
        try:
            if not image_bytes or len(image_bytes) < 100:
                return None

            cv2 = get_cv2()
            nparr = np.frombuffer(image_bytes, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            if img is None:
                return None

            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            return img_rgb

        except Exception:
            traceback.print_exc()
            return None

    # -------------------------
    # Face detection
    # -------------------------
    def detect_faces(self, image_rgb: np.ndarray) -> List:
        if not self._ensure_ready():
            return []

        try:
            return self.app.get(image_rgb)
        except Exception:
            traceback.print_exc()
            return []

    # -------------------------
    # Embeddings
    # -------------------------
    def get_embeddings(self, faces: List) -> List[np.ndarray]:
        embeddings = []
        for face in faces:
            if hasattr(face, "embedding"):
                embeddings.append(face.embedding)
        return embeddings

    # -------------------------
    # Bounding boxes
    # -------------------------
    def get_bounding_boxes(self, faces: List) -> List[Dict[str, Any]]:
        boxes = []
        for i, face in enumerate(faces):
            if hasattr(face, "bbox"):
                x1, y1, x2, y2 = map(int, face.bbox)
                boxes.append(
                    {
                        "index": i,
                        "x1": x1,
                        "y1": y1,
                        "x2": x2,
                        "y2": y2,
                        "width": x2 - x1,
                        "height": y2 - y1,
                    }
                )
        return boxes

    # -------------------------
    # Main processing
    # -------------------------
    def process_single_image(
        self, image_bytes: bytes, require_single_face: bool = True
    ) -> Dict[str, Any]:

        result = {
            "success": False,
            "faces_detected": 0,
            "embeddings": [],
            "bounding_boxes": [],
            "error": None,
        }

        try:
            image_rgb = self.load_image_from_bytes(image_bytes)
            if image_rgb is None:
                result["error"] = "Invalid image"
                return result

            faces = self.detect_faces(image_rgb)
            if not faces:
                result["error"] = "No faces detected"
                return result

            if require_single_face and len(faces) != 1:
                result["error"] = f"Expected 1 face, found {len(faces)}"
                return result

            embeddings = self.get_embeddings(faces)
            if not embeddings:
                result["error"] = "Failed to extract embeddings"
                return result

            result.update(
                {
                    "success": True,
                    "faces_detected": len(faces),
                    "embeddings": [e.tolist() for e in embeddings],
                    "bounding_boxes": self.get_bounding_boxes(faces),
                }
            )

            return result

        except Exception as e:
            traceback.print_exc()
            result["error"] = str(e)
            return result

    # -------------------------
    # Face comparison
    # -------------------------
    def compare_faces(
        self, emb1: List[float], emb2: List[float], threshold: float = 0.6
    ) -> Tuple[float, bool]:

        try:
            v1 = np.array(emb1).reshape(1, -1)
            v2 = np.array(emb2).reshape(1, -1)

            similarity = cosine_similarity(v1, v2)[0][0]
            return similarity, similarity >= threshold

        except Exception:
            traceback.print_exc()
            return 0.0, False


# =========================
# Lazy Singleton (IMPORTANT)
# =========================
_face_service: Optional[FaceService] = None


def get_face_service() -> FaceService:
    global _face_service
    if _face_service is None:
        _face_service = FaceService()
    return _face_service
