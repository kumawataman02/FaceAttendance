import json
from fastapi import APIRouter, Depends, HTTPException, status, UploadFile, File, Query
from sqlalchemy.ext.asyncio import AsyncSession
from datetime import datetime
import traceback

from app.crud.attendance import get_all_enrolled_students, get_today_attendance_records, calculate_total_duration_today
from app.dependencies import get_db
from app.models.attendance import AttendanceRecord
from app.schemas.attendance import AttendanceResponse

from app.services.face_recognition import face_service

router = APIRouter(prefix="/attendance", tags=["attendance"])


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
    try:
        print(f"\n{'=' * 60}")
        print(f"📸 FACE ATTENDANCE - Started")
        print(f"{'=' * 60}")

        # Validate image
        allowed_types = {'image/jpeg', 'image/jpg', 'image/png', 'image/webp'}
        if image.content_type not in allowed_types:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file type '{image.content_type}'. Use JPEG, PNG, or WebP"
            )

        # Read image
        image_bytes = await image.read()
        if len(image_bytes) < 1024:
            raise HTTPException(
                status_code=400,
                detail="Image file too small or corrupted"
            )

        # Process image with face detection
        result = face_service.process_single_image(
            image_bytes,
            require_single_face=True
        )

        if not result["success"]:
            error_msg = result.get("error", "Face detection failed")
            raise HTTPException(
                status_code=400,
                detail=error_msg
            )

        if result["faces_detected"] == 0:
            raise HTTPException(
                status_code=400,
                detail="No face detected in the image"
            )

        if result["faces_detected"] > 1:
            raise HTTPException(
                status_code=400,
                detail="Multiple faces detected. Please upload an image with only one person."
            )

        # Get face embedding
        face_embedding = result["embeddings"][0]

        # Get all enrolled students in the branch
        students = await get_all_enrolled_students(db, branch_name)

        if not students:
            raise HTTPException(
                status_code=404,
                detail="No enrolled students found in this branch"
            )

        # Prepare student embeddings dictionary
        student_embeddings_dict = {}
        student_info_dict = {}

        for student in students:
            student_id = student.dlb_sty_id
            embeddings = []

            # Get first embedding
            if student.face_embedding1:
                try:
                    emb1 = json.loads(student.face_embedding1)
                    embeddings.append(emb1)
                except:
                    pass

            # Get second embedding
            if student.face_embedding2:
                try:
                    emb2 = json.loads(student.face_embedding2)
                    embeddings.append(emb2)
                except:
                    pass

            if embeddings:
                student_embeddings_dict[student_id] = embeddings
                student_info_dict[student_id] = {
                    "name": student.dlb_sty_name,
                    "roll_number": student.dlb_roll_number,
                    "branch": student.dlb_offline_name
                }

        # Find best match
        best_student_id = None
        best_similarity = 0.0

        for student_id, embeddings in student_embeddings_dict.items():
            max_similarity = 0.0
            for student_emb in embeddings:
                similarity, _ = face_service.compare_faces(face_embedding, student_emb, threshold=0.0)
                max_similarity = max(max_similarity, similarity)

            if max_similarity > best_similarity:
                best_similarity = max_similarity
                best_student_id = student_id

        # Check if match meets threshold
        if not best_student_id or best_similarity < 0.6:
            raise HTTPException(
                status_code=400,
                detail=f"No matching student found. Best similarity: {best_similarity:.3f} (threshold: 0.6)"
            )

        student_info = student_info_dict[best_student_id]
        current_time = datetime.now()
        today = current_time.date()

        # Get today's attendance records for this student
        today_attendance = await get_today_attendance_records(db, best_student_id, today)

        # Calculate total duration for today (pairing consecutive entries)
        total_duration = calculate_total_duration_today(today_attendance)

        # Create new attendance record
        new_attendance = AttendanceRecord(
            student_id=best_student_id,
            student_name=student_info["name"],  # Added student_name
            branch_name=branch_name,
            attendance_date=today,
            attendance_time=current_time,
            created_at=current_time
        )

        db.add(new_attendance)
        await db.commit()
        await db.refresh(new_attendance)

        # Get updated list including new entry
        updated_attendance = await get_today_attendance_records(db, best_student_id, today)
        total_entries_today = len(updated_attendance)

        print(f"\n✅ ATTENDANCE RECORDED:")
        print(f"   Student: {student_info['name']} (ID: {best_student_id})")
        print(f"   Roll: {student_info['roll_number']}")
        print(f"   Similarity: {best_similarity:.3f}")
        print(f"   Attendance Time: {current_time.strftime('%H:%M:%S')}")
        print(f"   Today's Entries: {total_entries_today}")
        print(f"   Total Duration Today: {total_duration:.2f} hours")
        print(f"{'=' * 60}")

        return AttendanceResponse(
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

    except HTTPException:
        raise
    except Exception as e:
        print(f"💥 Face attendance error: {e}")
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail=f"Failed to process face attendance: {str(e)}"
        )


