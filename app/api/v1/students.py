# app/api/v1/students.py
from fastapi import APIRouter, Depends, HTTPException, status, Query, UploadFile, File, BackgroundTasks
from sqlalchemy.ext.asyncio import AsyncSession
from typing import Optional
import asyncio

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
    # Calculate skip
    skip = (page - 1) * limit

    # Create filter object
    filters = StudentFilter(
        status=status,
        search=search
    )

    # Get students and total count
    students, total = await get_students(
        db,
        branch_name=branch_name,
        skip=skip,
        limit=limit,
        filters=filters
    )

    # Calculate total pages
    total_pages = (total + limit - 1) // limit

    return PaginatedStudentResponse(
        total=total,
        page=page,
        limit=limit,
        total_pages=total_pages,
        students=students
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
    # Check if student exists
    student = await get_student_by_id(db, student_id)
    if not student:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Student not found"
        )

    # Update student profile
    updated_student = await update_student_profile(db, student_id, student_update)
    if not updated_student:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to update student profile"
        )

    return updated_student


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
        # Validate file types
        if not photo1.content_type.startswith('image/') or not photo2.content_type.startswith('image/'):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Only image files are allowed"
            )

        # Check file size (max 5MB each)
        max_size = 5 * 1024 * 1024  # 5MB
        await photo1.seek(0)
        await photo2.seek(0)

        photo1_bytes = await photo1.read()
        photo2_bytes = await photo2.read()

        if len(photo1_bytes) > max_size or len(photo2_bytes) > max_size:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Image size should be less than 5MB"
            )

        # Check if student exists
        student = await get_student_by_id(db, student_id)
        if not student:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Student not found"
            )

        # Process first photo
        face_data1 = face_enrollment_service.process_face_photo(photo1_bytes)

        # Process second photo
        face_data2 = face_enrollment_service.process_face_photo(photo2_bytes)

        # Validate that both photos are of the same person
        validation_result = face_enrollment_service.validate_two_photos(
            face_data1["embedding"],
            face_data2["embedding"]
        )

        if not validation_result["validation_passed"]:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Photos don't appear to be of the same person. Similarity: {validation_result['similarity_score']:.3f} (threshold: 0.6)"
            )

        # Save embeddings to database
        enrolled_student = await enroll_student_face(
            db,
            student_id,
            face_data1["embedding"],
            face_data2["embedding"]
        )

        if not enrolled_student:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to save face data"
            )

        return FaceEnrollmentResponse(
            success=True,
            message="Face enrollment successful",
            student_id=student_id,
            face_enrolled_date=enrolled_student.face_enrolled_date
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Face enrollment failed: {str(e)}"
        )


@str_router.get("/service-status")
async def get_face_service_status():
    """Check face recognition service status"""
    try:
        from app.services.face_recognition import get_face_service
        service = get_face_service()

        if service is None:
            return {
                "status": "not_initialized",
                "ready": False,
                "message": "Face service not initialized"
            }

        return {
            "status": service.status.value,
            "ready": service.is_ready(),
            "message": "Face service is running" if service.is_ready() else "Face service is not ready"
        }
    except Exception as e:
        return {
            "status": "error",
            "ready": False,
            "message": f"Error checking service status: {str(e)}"
        }