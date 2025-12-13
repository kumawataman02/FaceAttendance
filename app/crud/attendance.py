from sqlalchemy.future import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import func, and_, or_
from typing import List, Optional, Dict
from datetime import datetime, date, timedelta
import json

from app.models.attendance import AttendanceRecord
from app.models.student import WifiOffline


async def get_all_enrolled_students(
        db: AsyncSession,
        branch_name: str
) -> List[WifiOffline]:
    """Get all enrolled students in a specific branch"""
    query = select(WifiOffline).where(
        and_(
            WifiOffline.dlb_offline_name == branch_name,
            or_(
                WifiOffline.face_embedding1.isnot(None),
                WifiOffline.face_embedding2.isnot(None)
            )
        )
    )
    result = await db.execute(query)
    return result.scalars().all()


async def get_today_attendance_records(
        db: AsyncSession,
        student_id: int,
        today: date
) -> List[AttendanceRecord]:
    """Get all attendance records for a student today, ordered by time"""
    query = select(AttendanceRecord).where(
        and_(
            AttendanceRecord.student_id == student_id,
            AttendanceRecord.attendance_date == today
        )
    ).order_by(AttendanceRecord.attendance_time)

    result = await db.execute(query)
    return result.scalars().all()


def calculate_total_duration_today(attendance_records: List[AttendanceRecord]) -> float:
    """
    Calculate total duration by pairing consecutive attendance entries.
    Each pair (punch in -> punch out) adds to total duration.
    If odd number of entries, the last one is considered punch in (not paired).
    """
    total_seconds = 0

    # Sort by time (just in case)
    sorted_records = sorted(attendance_records, key=lambda x: x.attendance_time)

    # Process in pairs
    for i in range(0, len(sorted_records) - 1, 2):
        entry_in = sorted_records[i]
        entry_out = sorted_records[i + 1]

        duration = (entry_out.attendance_time - entry_in.attendance_time).total_seconds()
        total_seconds += duration

    return total_seconds / 3600.0