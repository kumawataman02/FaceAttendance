import os
from fastapi import UploadFile, HTTPException, status
from typing import Dict, Any, Optional, List, Tuple
from app.services.face_recognition import get_face_service
import logging

logger = logging.getLogger(__name__)


class FaceEnrollmentService:
    def __init__(self, upload_dir: str = "uploads/faces"):
        self.upload_dir = upload_dir
        os.makedirs(upload_dir, exist_ok=True)

        # Initialize face service (will be lazy-loaded)
        self.face_service = None

    def _ensure_service_ready(self):
        """Ensure face service is ready to use"""
        if self.face_service is None:
            self.face_service = get_face_service()

        if self.face_service is None or not self.face_service.is_ready():
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Face recognition service is not available or still initializing"
            )

    def process_face_photo(self, photo_bytes: bytes) -> Dict[str, Any]:
        """Process face photo and extract embedding"""
        self._ensure_service_ready()

        try:
            # Process the image
            result = self.face_service.process_image(photo_bytes, require_single_face=True)

            if not result.success:
                error_msg = result.error_message or "Face detection failed"
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=error_msg
                )

            # Check face quality if available
            if result.face_qualities:
                quality = result.face_qualities[0]
                if not quality.get('valid', False):
                    reason = quality.get('reason', 'Poor quality image')
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail=f"Image quality issue: {reason}"
                    )

            # Return the first face embedding
            if result.embeddings:
                embedding = result.embeddings[0]
                bounding_box = result.bounding_boxes[0] if result.bounding_boxes else {}

                return {
                    "embedding": embedding,
                    "bounding_box": bounding_box,
                    "quality_score": result.face_qualities[0].get('blur_score', 0) if result.face_qualities else 0,
                    "face_detected": True
                }

            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Failed to extract face features"
            )

        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error processing face photo: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Face processing error: {str(e)}"
            )

    def validate_two_photos(self, embedding1: List[float], embedding2: List[float],
                            threshold: float = 0.6) -> Dict[str, Any]:
        """Validate that two photos are of the same person"""
        self._ensure_service_ready()

        try:
            result = self.face_service.compare_faces(embedding1, embedding2, threshold)

            return {
                "same_person": result.is_match,
                "similarity_score": float(result.similarity_score),
                "threshold": threshold,
                "validation_passed": result.is_match
            }

        except Exception as e:
            logger.error(f"Error comparing faces: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Face comparison error: {str(e)}"
            )


# Create global instance
face_enrollment_service = FaceEnrollmentService()