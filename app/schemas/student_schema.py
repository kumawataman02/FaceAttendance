from datetime import datetime, date
from typing import Optional, List

from pydantic import BaseModel, EmailStr, Field


class StudentBase(BaseModel):
    dlb_sty_name: Optional[str] = None
    dlb_sty_email: Optional[EmailStr] = None
    dlb_sty_mobile: Optional[str] = None
    dlb_sty_mother: Optional[str] = None
    dlb_sty_father: Optional[str] = None
    dlb_parents_number: Optional[str] = None
    dlb_sty_address: Optional[str] = None
    dlb_zipcode: Optional[str] = None
    dlb_offline_name: Optional[str] = None
    dlb_center_name: Optional[str] = None
    dlb_sty_batch: Optional[str] = None
    dlb_sty_exam: Optional[str] = None
    dlb_total_fees: Optional[str] = None
    dlb_final_fees: Optional[str] = None
    dlb_remark: Optional[str] = None
    dlb_sty_image: Optional[str] = None
    dlb_sty_status: int = 0
    dlb_sty_activedate: date
    dlb_order_id: int = 0
    sendFeedback: int = 0
    feedbackLoop: int = 0
    dlb_call_status: int
    dlb_call_mark: Optional[str] = None
    dlb_call_activate_date: Optional[datetime] = None
    send_feedback_date: Optional[date] = None
    dlb_sty_session: Optional[str] = None
    dlb_roll_number: Optional[str] = None
    dlb_event: int = 0

class StudentUpdate(BaseModel):
    """Schema for updating student profile"""
    dlb_sty_name: Optional[str] = Field(None, min_length=1, max_length=255)
    dlb_sty_email: Optional[EmailStr] = None
    dlb_sty_mobile: Optional[str] = Field(None, min_length=10, max_length=15)
    dlb_sty_mother: Optional[str] = None
    dlb_sty_father: Optional[str] = None
    dlb_parents_number: Optional[str] = None
    dlb_sty_address: Optional[str] = None
    dlb_zipcode: Optional[str] = None
    dlb_sty_batch: Optional[str] = None
    dlb_sty_exam: Optional[str] = None
    dlb_total_fees: Optional[str] = None
    dlb_final_fees: Optional[str] = None
    dlb_remark: Optional[str] = None
    dlb_sty_image: Optional[str] = None
    dlb_sty_status: Optional[int] = None
    dlb_sty_activedate: Optional[date] = None
    dlb_sty_session: Optional[str] = None
    dlb_roll_number: Optional[str] = None

class StudentResponse(BaseModel):
    dlb_sty_id: int
    dlb_a_id: int
    dlb_sty_name: Optional[str]
    dlb_sty_email: Optional[str]
    dlb_sty_mobile: Optional[str]
    dlb_roll_number: Optional[str]
    dlb_sty_batch: Optional[str]
    dlb_sty_status: int
    dlb_sty_created_date: datetime
    dlb_offline_name: Optional[str]
    dlb_center_name: Optional[str]
    face_photo1: Optional[str] = None
    face_photo2: Optional[str] = None
    face_enrolled_date: Optional[datetime] = None


    class Config:
        from_attributes = True


class PaginatedStudentResponse(BaseModel):
    total: int
    page: int
    limit: int
    total_pages: int
    students: List[StudentResponse]

class StudentFilter(BaseModel):
    status: Optional[int] = None
    batch: Optional[str] = None
    search: Optional[str] = None

class FaceEnrollmentResponse(BaseModel):
    success: bool
    message: str
    student_id: int
    face_enrolled_date: Optional[datetime]

