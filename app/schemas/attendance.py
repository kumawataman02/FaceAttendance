from pydantic import BaseModel
from datetime import datetime, date


class AttendanceResponse(BaseModel):
    success: bool
    message: str
    action: str  # "attendance_recorded"
    student_id: int
    student_name: str
    roll_number: str
    similarity_score: float
    attendance_time: datetime
    total_duration_today: float = 0.0  # Total working hours for today
    total_entries_today: int = 0  # Number of attendance entries today