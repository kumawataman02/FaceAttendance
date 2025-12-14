# app/routers/attendance.py
import json
import logging
from datetime import datetime
from fastapi import APIRouter, Depends, HTTPException, status, UploadFile, File, Query
from sqlalchemy.ext.asyncio import AsyncSession
import traceback

from app.crud.attendance import get_all_enrolled_students, get_today_attendance_records, calculate_total_duration_today
from app.dependencies import get_db
from app.models.attendance import AttendanceRecord
from app.schemas.attendance import AttendanceResponse
from app.services.face_recognition import get_face_service, ServiceStatus

# Setup logging
logger = logging.getLogger(__name__)
router = APIRouter(prefix="/attendance", tags=["attendance"])


@router.get("/health")
async def health_check():
    """Check the health status of the face recognition service"""
    try:
        face_service = get_face_service()

        if face_service is None:
            return {
                "status": "error",
                "message": "Face service not initialized",
                "details": "The face recognition service failed to initialize",
                "timestamp": datetime.now().isoformat()
            }

        service_status = face_service.get_status()

        return {
            "status": service_status["status"],
            "message": f"Face recognition service is {service_status['status']}",
            "details": service_status,
            "timestamp": datetime.now().isoformat()
        }

    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {
            "status": "error",
            "message": "Health check failed",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }


@router.get("/diagnostic")
async def diagnostic_check():
    """Diagnostic endpoint to check system health"""
    import sys
    import pkg_resources
    import os

    diagnostic_info = {
        "python_version": sys.version,
        "system_info": {
            "platform": sys.platform,
            "processor": os.uname().machine if hasattr(os, 'uname') else "unknown"
        },
        "installed_packages": {},
        "import_tests": {}
    }

    # Test package imports
    packages_to_test = [
        "cv2", "numpy", "sklearn", "insightface", "onnxruntime",
        "PIL", "sqlalchemy", "fastapi"
    ]

    for package in packages_to_test:
        try:
            __import__(package.replace('-', '_'))
            diagnostic_info["import_tests"][package] = "SUCCESS"

            # Try to get version
            try:
                version = pkg_resources.get_distribution(package).version
                diagnostic_info["installed_packages"][package] = version
            except:
                diagnostic_info["installed_packages"][package] = "unknown"

        except ImportError as e:
            diagnostic_info["import_tests"][package] = f"FAILED: {str(e)}"

    # Check face service
    face_service = get_face_service()
    if face_service:
        diagnostic_info["face_service"] = face_service.get_status()
    else:
        diagnostic_info["face_service"] = {"status": "NOT_INITIALIZED"}

    return diagnostic_info


@router.post("/face/attendance", response_model=AttendanceResponse)
async def face_attendance(
        image: UploadFile = File(..., description="Face image for attendance"),
        branch_name: str = Query(..., description="Branch name"),
        db: AsyncSession = Depends(get_db)
):
    """
    Face-based attendance recording for a student

    - Upload face image
    - System matches face with enrolled students in the branch
    - Records attendance entry with current timestamp
    - Multiple entries per day are allowed and accumulated
    """
    logger.info(f"Face attendance request received for branch: {branch_name}")

    try:
        # Get face service instance
        face_service = get_face_service()

        # Check if face service is initialized
        if face_service is None:
            logger.error("Face service instance is None - initialization failed")
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail={
                    "message": "Face recognition service failed to initialize",
                    "action": "Please check server logs and ensure all dependencies are installed",
                    "error_type": "service_initialization_failed"
                }
            )

        # Check service status
        service_status = face_service.status
        logger.info(f"Face service status: {service_status}")

        if service_status == ServiceStatus.ERROR:
            error_msg = face_service.initialization_error or "Unknown error during initialization"
            logger.error(f"Face service in ERROR state: {error_msg}")
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail={
                    "message": "Face recognition service is not available",
                    "error": error_msg,
                    "action": "Please check that insightface and onnxruntime are properly installed",
                    "error_type": "service_error"
                }
            )

        if service_status == ServiceStatus.INITIALIZING:
            logger.warning("Face service still initializing")
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail={
                    "message": "Face recognition service is initializing",
                    "action": "Please wait a moment and try again",
                    "estimated_wait": "10-30 seconds",
                    "error_type": "service_initializing"
                }
            )

        if not face_service.is_ready():
            logger.error(f"Face service not ready. Status: {service_status}")
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail={
                    "message": f"Face recognition service is not ready (status: {service_status})",
                    "action": "Contact system administrator",
                    "error_type": "service_not_ready"
                }
            )

        logger.info(f"\n{'=' * 60}")
        logger.info(f"📸 FACE ATTENDANCE - Started")
        logger.info(f"{'=' * 60}")

        # Validate image
        allowed_types = {'image/jpeg', 'image/jpg', 'image/png', 'image/webp'}
        if image.content_type not in allowed_types:
            logger.warning(f"Unsupported file type: {image.content_type}")
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file type '{image.content_type}'. Use JPEG, PNG, or WebP"
            )

        # Read image
        image_bytes = await image.read()
        logger.info(f"Image read: {len(image_bytes)} bytes")

        if len(image_bytes) < 1024:
            logger.warning("Image file too small")
            raise HTTPException(
                status_code=400,
                detail="Image file too small or corrupted"
            )

        if len(image_bytes) > 10 * 1024 * 1024:  # 10MB limit
            logger.warning("Image file too large")
            raise HTTPException(
                status_code=400,
                detail="Image file too large (max 10MB)"
            )

        # Process image with face detection
        logger.info("Processing image with face detection")
        result = face_service.process_image(
            image_bytes,
            require_single_face=True
        )

        # Log face detection result
        logger.info(f"Face detection result - success: {result.success}, faces: {result.faces_detected}")

        if not result.success:
            error_msg = result.error_message or "Face detection failed"
            logger.error(f"Face detection failed: {error_msg}")
            raise HTTPException(
                status_code=400,
                detail=error_msg
            )

        if result.faces_detected == 0:
            logger.warning("No face detected in image")
            raise HTTPException(
                status_code=400,
                detail="No face detected in the image. Please ensure face is clearly visible."
            )

        if result.faces_detected > 1:
            logger.warning(f"Multiple faces detected: {result.faces_detected}")
            raise HTTPException(
                status_code=400,
                detail="Multiple faces detected. Please upload an image with only one person."
            )

        # Check face quality if available
        if hasattr(result, 'face_qualities') and result.face_qualities:
            quality = result.face_qualities[0]
            if not quality.get('valid', False):
                reason = quality.get('reason', 'Poor face quality')
                logger.warning(f"Poor face quality: {reason}")
                raise HTTPException(
                    status_code=400,
                    detail=f"Image quality issue: {reason}"
                )

        # Get face embedding
        if not result.embeddings or len(result.embeddings) == 0:
            logger.error("No face embedding extracted")
            raise HTTPException(
                status_code=400,
                detail="Failed to extract face features"
            )

        face_embedding = result.embeddings[0]
        logger.info(f"Face embedding extracted: {len(face_embedding)} dimensions")

        # Get all enrolled students in the branch
        logger.info(f"Fetching enrolled students for branch: {branch_name}")
        students = await get_all_enrolled_students(db, branch_name)

        if not students:
            logger.warning(f"No enrolled students found in branch: {branch_name}")
            raise HTTPException(
                status_code=404,
                detail="No enrolled students found in this branch"
            )

        logger.info(f"Found {len(students)} enrolled students")

        # Prepare student embeddings dictionary
        student_embeddings_dict = {}
        student_info_dict = {}
        valid_students = 0

        for student in students:
            student_id = student.dlb_sty_id
            embeddings = []

            # Get first embedding
            if student.face_embedding1:
                try:
                    emb1 = json.loads(student.face_embedding1)
                    if isinstance(emb1, list) and len(emb1) > 0:
                        embeddings.append(emb1)
                except (json.JSONDecodeError, TypeError) as e:
                    logger.warning(f"Failed to parse embedding1 for student {student_id}: {e}")

            # Get second embedding
            if student.face_embedding2:
                try:
                    emb2 = json.loads(student.face_embedding2)
                    if isinstance(emb2, list) and len(emb2) > 0:
                        embeddings.append(emb2)
                except (json.JSONDecodeError, TypeError) as e:
                    logger.warning(f"Failed to parse embedding2 for student {student_id}: {e}")

            if embeddings:
                student_embeddings_dict[student_id] = embeddings
                student_info_dict[student_id] = {
                    "name": student.dlb_sty_name,
                    "roll_number": student.dlb_roll_number,
                    "branch": student.dlb_offline_name
                }
                valid_students += 1

        if valid_students == 0:
            logger.error(f"No valid face embeddings found for any student in branch {branch_name}")
            raise HTTPException(
                status_code=400,
                detail="No students with valid face embeddings found in this branch"
            )

        logger.info(f"Processed {valid_students} students with valid embeddings")

        # Find best match
        best_student_id = None
        best_similarity = 0.0
        threshold = 0.6  # Recognition threshold

        logger.info("Starting face matching process...")
        for student_id, embeddings in student_embeddings_dict.items():
            max_similarity = 0.0
            for student_emb in embeddings:
                try:
                    # Compare faces
                    comparison_result = face_service.compare_faces(
                        face_embedding,
                        student_emb,
                        threshold=0.0
                    )
                    similarity = comparison_result.similarity_score
                    max_similarity = max(max_similarity, similarity)
                except Exception as e:
                    logger.warning(f"Failed to compare faces for student {student_id}: {e}")
                    continue

            if max_similarity > best_similarity:
                best_similarity = max_similarity
                best_student_id = student_id

        logger.info(f"Best match - Student ID: {best_student_id}, Similarity: {best_similarity:.3f}")

        # Check if match meets threshold
        if not best_student_id or best_similarity < threshold:
            logger.warning(f"No match found above threshold. Best similarity: {best_similarity:.3f}")
            raise HTTPException(
                status_code=400,
                detail=f"No matching student found. Best similarity: {best_similarity:.3f} (threshold: {threshold})"
            )

        student_info = student_info_dict[best_student_id]
        current_time = datetime.now()
        today = current_time.date()

        logger.info(f"Matched student: {student_info['name']} (Roll: {student_info['roll_number']})")

        # Get today's attendance records for this student
        today_attendance = await get_today_attendance_records(db, best_student_id, today)
        total_entries_today = len(today_attendance) + 1  # Including current entry

        # Calculate total duration for today (pairing consecutive entries)
        total_duration = calculate_total_duration_today(today_attendance)

        # Create new attendance record
        new_attendance = AttendanceRecord(
            student_id=best_student_id,
            student_name=student_info["name"],
            branch_name=branch_name,
            attendance_date=today,
            attendance_time=current_time,
            created_at=current_time
        )

        db.add(new_attendance)
        await db.commit()
        await db.refresh(new_attendance)

        logger.info(f"Attendance recorded successfully for student {best_student_id} at {current_time}")

        # Prepare response
        response = AttendanceResponse(
            success=True,
            message="Attendance recorded successfully via face recognition",
            action="attendance_recorded",
            student_id=best_student_id,
            student_name=student_info["name"],
            roll_number=student_info["roll_number"],
            similarity_score=float(best_similarity),
            attendance_time=current_time,
            total_duration_today=total_duration,
            total_entries_today=total_entries_today
        )

        # Log successful attendance
        logger.info(f"""
           ATTENDANCE RECORDED:
           Student: {student_info['name']} (ID: {best_student_id})
           Roll: {student_info['roll_number']}
           Branch: {branch_name}
           Similarity: {best_similarity:.3f}
           Attendance Time: {current_time.strftime('%Y-%m-%d %H:%M:%S')}
           Today's Entries: {total_entries_today}
           Total Duration Today: {total_duration:.2f} hours
        """)

        return response

    except HTTPException as he:
        # Re-raise HTTP exceptions as-is
        logger.error(f"HTTP Exception in face attendance: {he.detail}")
        raise he

    except Exception as e:
        logger.error(f"Unexpected error in face attendance: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "message": "Internal server error",
                "error": str(e),
                "action": "Please try again or contact support",
                "error_type": "unexpected_error"
            }
        )