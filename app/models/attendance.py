from sqlalchemy import Column, Integer, String, DateTime, Date, Float, ForeignKey
from sqlalchemy.sql import func
from app.database import Base


class AttendanceRecord(Base):
    __tablename__ = "attendance_records"

    attendance_id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    student_id = Column(Integer, ForeignKey("wifi_offline.dlb_sty_id"), nullable=False)
    student_name = Column(String, nullable=False)
    branch_name = Column(String(255), nullable=False)
    attendance_date = Column(Date, nullable=False)
    attendance_time = Column(DateTime, nullable=False)
    created_at = Column(DateTime, nullable=False, server_default=func.now())
