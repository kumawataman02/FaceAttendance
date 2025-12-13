from datetime import datetime
from typing import Optional, Tuple, List

import json
from sqlalchemy import or_, func
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select

from app.models.student import WifiOffline
from app.schemas.student_schema import StudentFilter, StudentUpdate


async def get_students(db:AsyncSession,branch_name:str,skip:int=0,limit:int=10,filters:Optional[StudentFilter]=None) -> Tuple[List[WifiOffline], int]:

    """
       Get students by branch name (dlb_offline_name)
       """

    #Build base Query

    query = select(WifiOffline).where(WifiOffline.dlb_offline_name==branch_name)

    if filters:
        if filters.status is not None:
            query = query.where(WifiOffline.dlb_sty_status == filters.status)

        if filters.batch:
            query = query.where(WifiOffline.dlb_sty_batch == filters.batch)
        if filters.search:
            search_pattern = f"%{filters.search}%"
            query = query.where(
                or_(
                    WifiOffline.dlb_sty_name.like(search_pattern),
                    WifiOffline.dlb_sty_mobile.like(search_pattern),
                    WifiOffline.dlb_sty_email.like(search_pattern),
                    WifiOffline.dlb_roll_number.like(search_pattern)
                )
            )

    count_query = select(func.count()).select_from(query.subquery())
    total_result = await db.execute(count_query)
    total = total_result.scalar()

    # Get paginated results
    query = query.order_by(WifiOffline.dlb_sty_created_date.desc()).offset(skip).limit(limit)
    result = await db.execute(query)
    students = result.scalars().all()

    return students, total


async def update_student_profile(
        db: AsyncSession,
        student_id: int,
        student_update: StudentUpdate
) -> Optional[WifiOffline]:
    """
    Update student profile information
    """
    result = await db.execute(
        select(WifiOffline).where(WifiOffline.dlb_sty_id == student_id)
    )
    db_student = result.scalar_one_or_none()

    if not db_student:
        return None

    # Update only provided fields
    update_data = student_update.dict(exclude_unset=True)
    for field, value in update_data.items():
        setattr(db_student, field, value)

    await db.commit()
    await db.refresh(db_student)
    return db_student


async def get_student_by_id(db: AsyncSession, student_id: int) -> Optional[WifiOffline]:
    result = await db.execute(
        select(WifiOffline).where(WifiOffline.dlb_sty_id == student_id)
    )
    return result.scalar_one_or_none()


async def enroll_student_face(
        db: AsyncSession,
        student_id: int,
        face_embedding1: List[float],
        face_embedding2: List[float],

) -> Optional[WifiOffline]:
    """
    Enroll student face with two photos and embeddings
    """
    result = await db.execute(
        select(WifiOffline).where(WifiOffline.dlb_sty_id == student_id)
    )
    db_student = result.scalar_one_or_none()

    if not db_student:
        return None

    # Update face data
    db_student.face_embedding1 = json.dumps(face_embedding1)
    db_student.face_embedding2 = json.dumps(face_embedding2)
    db_student.face_enrolled_date = datetime.now()

    await db.commit()
    await db.refresh(db_student)
    return db_student






