import logging
import asyncio
from typing import Optional
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, status, Query, UploadFile, File, BackgroundTasks
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.exc import SQLAlchemyError

from app.dependencies import get_db
from app.crud.student import (
    update_student_profile,
    get_student_by_id,
    enroll_student_face,
    get_students
)
from app.schemas.student_schema import PaginatedStudentResponse, StudentFilter, StudentResponse, StudentUpdate, \
    FaceEnrollmentResponse
from app.services.face_service import face_enrollment_service


# Setup logger
logger = logging.getLogger(__name__)

str_router = APIRouter(prefix="/students", tags=["students"])


@str_router.get("/branch/{branch_name}", response_model=PaginatedStudentResponse)
async def get_students_by_branch(
        branch_name: str,
        page: int = Query(1, ge=1, description="Page number"),
        limit: int = Query(10, ge=1, le=100, description="Items per page"),
        status: Optional[int] = Query(None, description="Filter by student status"),
        search: Optional[str] = Query(None, description="Search by name, mobile, email, or roll number"),
        db: AsyncSession = Depends(get_db)
):
    """
    Get students belonging to a specific branch (dlb_offline_name).

    - **branch_name**: Name of the branch to fetch students for
    - **page**: Page number (default: 1)
    - **limit**: Items per page (default: 10, max: 100)
    - **status**: Filter by student status
    - **search**: Search in name, mobile, email, or roll number
    """
    try:
        logger.info(
            f"Fetching students for branch: {branch_name}, "
            f"page: {page}, limit: {limit}, status: {status}, search: {search}"
        )

        # Calculate skip
        skip = (page - 1) * limit

        # Create filter object
        filters = StudentFilter(
            status=status,
            search=search
        )

        # Get students and total count
        students_result, total = await get_students(
            db,
            branch_name=branch_name,
            skip=skip,
            limit=limit,
            filters=filters
        )

        logger.debug(f"Retrieved {len(students_result)} students from database")

        # Convert SQLAlchemy models to Pydantic models
        student_responses = []
        for student in students_result:
            try:
                # Convert SQLAlchemy model to dict, then to Pydantic model
                student_dict = student.__dict__
                # Remove SQLAlchemy internal attribute
                student_dict.pop('_sa_instance_state', None)
                # Convert datetime objects to strings if needed
                for key, value in student_dict.items():
                    if isinstance(value, datetime):
                        student_dict[key] = value.isoformat()

                student_response = StudentResponse(**student_dict)
                student_responses.append(student_response)
            except Exception as e:
                logger.error(
                    f"Error converting student {student.id if hasattr(student, 'id') else 'unknown'}: {str(e)}")
                continue

        # Calculate total pages
        total_pages = (total + limit - 1) // limit

        logger.info(f"Successfully returned {len(student_responses)} students, total: {total}, pages: {total_pages}")

        return PaginatedStudentResponse(
            total=total,
            page=page,
            limit=limit,
            total_pages=total_pages,
            students=student_responses
        )

    except SQLAlchemyError as e:
        logger.error(f"Database error fetching students for branch {branch_name}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Database service temporarily unavailable"
        )
    except Exception as e:
        logger.error(f"Unexpected error fetching students for branch {branch_name}: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred"
        )


@str_router.put("/{student_id}", response_model=StudentResponse)
async def update_student_profile_endpoint(
        student_id: int,
        student_update: StudentUpdate,
        db: AsyncSession = Depends(get_db)
):
    """
    Update student profile information.

    - **student_id**: ID of student to update
    - **student_update**: Fields to update
    """
    try:
        logger.info(f"Updating student profile for student_id: {student_id}")

        # Check if student exists
        student = await get_student_by_id(db, student_id)
        if not student:
            logger.warning(f"Student not found with id: {student_id}")
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Student not found"
            )

        # Update student profile
        updated_student = await update_student_profile(db, student_id, student_update)
        if not updated_student:
            logger.error(f"Failed to update student profile for student_id: {student_id}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to update student profile"
            )

        # Convert to response model
        student_dict = updated_student.__dict__
        student_dict.pop('_sa_instance_state', None)

        logger.info(f"Successfully updated student profile for student_id: {student_id}")
        return StudentResponse(**student_dict)

    except HTTPException:
        raise
    except SQLAlchemyError as e:
        logger.error(f"Database error updating student {student_id}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Database service temporarily unavailable"
        )
    except Exception as e:
        logger.error(f"Unexpected error updating student {student_id}: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred"
        )


@str_router.post("/{student_id}/enroll-face", response_model=FaceEnrollmentResponse)
async def enroll_student_face_endpoint(
        student_id: int,
        photo1: UploadFile = File(...),
        photo2: UploadFile = File(...),
        db: AsyncSession = Depends(get_db)
):
    """
    Enroll student face with two photos for face recognition.

    - **photo1**: First face photo (front-facing)
    - **photo2**: Second face photo (different angle)
    """
    try:
        logger.info(f"Starting face enrollment for student_id: {student_id}")

        # Validate file types
        if not photo1.content_type.startswith('image/') or not photo2.content_type.startswith('image/'):
            logger.warning(f"Invalid file type for student {student_id}: {photo1.content_type}, {photo2.content_type}")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Only image files are allowed"
            )

        # Check file size (max 5MB each)
        max_size = 5 * 1024 * 1024  # Use config value
        await photo1.seek(0)
        await photo2.seek(0)

        photo1_bytes = await photo1.read()
        photo2_bytes = await photo2.read()

        if len(photo1_bytes) > max_size or len(photo2_bytes) > max_size:
            logger.warning(f"File size too large for student {student_id}: {len(photo1_bytes)}, {len(photo2_bytes)}")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Image size should be less than {5}MB"
            )

        # Check if student exists
        student = await get_student_by_id(db, student_id)
        if not student:
            logger.warning(f"Student not found for face enrollment: {student_id}")
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Student not found"
            )

        # Process first photo
        logger.debug(f"Processing first photo for student {student_id}")
        face_data1 = face_enrollment_service.process_face_photo(photo1_bytes)

        # Process second photo
        logger.debug(f"Processing second photo for student {student_id}")
        face_data2 = face_enrollment_service.process_face_photo(photo2_bytes)

        # Validate that both photos are of the same person
        logger.debug(f"Validating photos for student {student_id}")
        validation_result = face_enrollment_service.validate_two_photos(
            face_data1["embedding"],
            face_data2["embedding"]
        )

        if not validation_result["validation_passed"]:
            logger.warning(
                f"Face validation failed for student {student_id}: "
                f"similarity: {validation_result['similarity_score']:.3f}"
            )
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Photos don't appear to be of the same person. Similarity: {validation_result['similarity_score']:.3f} (threshold: 0.6)"
            )

        # Save embeddings to database
        logger.info(f"Saving face embeddings for student {student_id}")
        enrolled_student = await enroll_student_face(
            db,
            student_id,
            face_data1["embedding"],
            face_data2["embedding"]
        )

        if not enrolled_student:
            logger.error(f"Failed to save face data for student {student_id}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to save face data"
            )

        logger.info(f"Successfully enrolled face for student {student_id}")
        return FaceEnrollmentResponse(
            success=True,
            message="Face enrollment successful",
            student_id=student_id,
            face_enrolled_date=enrolled_student.face_enrolled_date
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Face enrollment failed for student {student_id}: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Face enrollment failed: {str(e)}"
        )


@str_router.get("/service-status")
async def get_face_service_status():
    """Check face recognition service status"""
    try:
        from app.services.face_recognition import get_face_service

        logger.debug("Checking face service status")
        service = get_face_service()

        if service is None:
            logger.warning("Face service not initialized")
            return {
                "status": "not_initialized",
                "ready": False,
                "message": "Face service not initialized",
                "timestamp": datetime.utcnow().isoformat()
            }

        is_ready = service.is_ready()
        status_info = {
            "status": service.status.value,
            "ready": is_ready,
            "message": "Face service is running" if is_ready else "Face service is not ready",
            "timestamp": datetime.utcnow().isoformat()
        }

        logger.debug(f"Face service status: {status_info}")
        return status_info

    except Exception as e:
        logger.error(f"Error checking face service status: {str(e)}", exc_info=True)
        return {
            "status": "error",
            "ready": False,
            "message": f"Error checking service status: {str(e)}",
            "timestamp": datetime.utcnow().isoformat()
        }


@str_router.get("/health")
async def health_check():
    """Health check endpoint for monitoring"""
    return {
        "status": "healthy",
        "service": "students-api",
        "timestamp": datetime.utcnow().isoformat(),
        "version": "1.0.0"
    }