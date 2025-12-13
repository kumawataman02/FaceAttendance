from sqlalchemy import Column, Integer, String, Date, DateTime, func, Text

from app.database import Base


class WifiOffline(Base):
    __tablename__ = "wifi_offline"

    dlb_sty_id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    dlb_a_id = Column(Integer, nullable=False, default=0)
    dlb_sty_name = Column(String(255), nullable=True)
    dlb_sty_email = Column(String(255), nullable=True)
    dlb_sty_mobile = Column(String(255), nullable=True)
    dlb_sty_mother = Column(String(255), nullable=True)
    dlb_sty_father = Column(String(255), nullable=True)
    dlb_parents_number = Column(String(255), nullable=True)
    dlb_sty_address = Column(String(255), nullable=True)
    dlb_zipcode = Column(String(255), nullable=True)
    dlb_offline_name = Column(String(255), nullable=True)
    dlb_center_name = Column(String(255), nullable=True)
    dlb_sty_batch = Column(String(256), nullable=True)
    dlb_sty_exam = Column(String(256), nullable=True)
    dlb_total_fees = Column(String(256), nullable=True)
    dlb_final_fees = Column(String(256), nullable=True)
    dlb_remark = Column(String(256), nullable=True)
    dlb_sty_image = Column(String(256), nullable=True)
    dlb_sty_status = Column(Integer, nullable=False, default=0)
    dlb_sty_activedate = Column(Date, nullable=False)
    dlb_sty_created_date = Column(DateTime, nullable=False, server_default=func.now())
    dlb_order_id = Column(Integer, nullable=False, default=0)
    sendFeedback = Column(Integer, nullable=False, default=0)
    feedbackLoop = Column(Integer, nullable=False, default=0)
    dlb_call_status = Column(Integer, nullable=False)
    dlb_call_mark = Column(String(255), nullable=True)
    dlb_call_activate_date = Column(DateTime, nullable=True)
    send_feedback_date = Column(Date, nullable=True)
    dlb_sty_session = Column(String(255), nullable=True)
    dlb_roll_number = Column(String(255), nullable=True)
    dlb_event = Column(Integer, nullable=False)

    # Face recognition fields
    face_photo1 = Column(String(500), nullable=True)
    face_photo2 = Column(String(500), nullable=True)
    face_embedding1 = Column(Text, nullable=True)
    face_embedding2 = Column(Text, nullable=True)
    face_enrolled_date = Column(DateTime, nullable=True)
