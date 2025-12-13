import os
from fastapi import UploadFile, HTTPException, status
from typing import Dict, Any, Optional, List, Tuple


# Create a wrapper for your FaceService class
class FaceEnrollmentService:
    def __init__(self, upload_dir: str = "uploads/faces"):
        self.upload_dir = upload_dir
        os.makedirs(upload_dir, exist_ok=True)

        # Import and initialize your FaceService
        from app.services.face_recognition import face_service
        self.face_service = face_service



    def process_face_photo(self, photo_bytes: bytes) -> Dict[str, Any]:
        """Process face photo and extract embedding using your FaceService"""
        if not self.face_service or not self.face_service.is_initialized():
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Face recognition service is not available"
            )

        # Process the image using your FaceService
        result = self.face_service.process_single_image(photo_bytes, require_single_face=True)

        if not result.get("success"):
            error_msg = result.get("error", "Face detection failed")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=error_msg
            )

        # Check face quality
        if result.get("face_qualities"):
            quality = result["face_qualities"][0]
            if not quality.get("valid", False):
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Poor quality image: {quality.get('reason', 'Please upload a clearer photo')}"
                )

        # Return the first face embedding
        if result.get("embeddings"):
            embedding = result["embeddings"][0]
            bounding_box = result["bounding_boxes"][0] if result.get("bounding_boxes") else {}

            return {
                "embedding": embedding,
                "bounding_box": bounding_box,
                "quality_score": result.get("face_qualities", [{}])[0].get("blur_score", 0),
                "face_detected": True
            }

        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Failed to extract face features"
        )

    def validate_two_photos(self, embedding1: List[float], embedding2: List[float],
                            threshold: float = 0.6) -> Dict[str, Any]:
        """Validate that two photos are of the same person"""
        similarity, is_match = self.face_service.compare_faces(embedding1, embedding2)

        return {
            "same_person": is_match,
            "similarity_score": float(similarity),
            "threshold": threshold,
            "validation_passed": is_match
        }


# Create global instance
face_enrollment_service = FaceEnrollmentService()