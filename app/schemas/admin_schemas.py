# schemas.py
from datetime import date, datetime
from decimal import Decimal

from pydantic import BaseModel, EmailStr, Field
from typing import Optional

class AdminBase(BaseModel):
    dlb_a_name: str = Field(..., min_length=1, max_length=255)
    dlb_a_phone: str = Field(..., min_length=10, max_length=15)
    dlb_a_email: EmailStr
    dlb_a_password: str = Field(..., min_length=6)
    dlb_a_city: str = Field(..., max_length=100)
    dlb_a_state: str = Field(..., max_length=20)
    dlb_a_country: str = Field(..., max_length=100)
    dlb_a_address: str
    dlb_a_zip_code: str = Field(..., max_length=15)
    dlb_admin_id: int
    dlb_super_admin: int = Field(0, ge=0, le=1)
    dlb_a_type: int = Field(0, ge=0)
    dlb_emp_code: int


class AdminCreate(AdminBase):
    dlb_a_ivr_number: Optional[str] = None
    dlb_a_image: str = ""
    dlb_a_image_thumb: str = ""
    dlb_a_status: int = 0
    dlb_ds_id: str = ""
    dlb_a_subject: Optional[str] = None
    dlb_a_promo: Optional[str] = None
    dlb_profile_link: Optional[str] = None
    dlb_tele_url: Optional[str] = None
    dlb_pro_url: Optional[str] = None
    dlb_pro_text: Optional[str] = None
    dlb_code_name: Optional[str] = None
    dlb_pkg_id: Optional[str] = None
    dlb_salary: Optional[float] = None
    dlb_branch_name: Optional[str] = None
    dlb_center_name: Optional[str] = None
    dlb_cls_type: int = 0
    dlb_aadhar_pdf: Optional[str] = None
    dlb_other_pdf: Optional[str] = None
    dlb_field: Optional[str] = None
    dlb_ac_number: Optional[str] = None
    dlb_ifsc_code: Optional[str] = None
    dlb_emp_type: Optional[str] = None
    dlb_counsellor_ids: Optional[str] = None
    dlb_auto_lead: int = 0
    dlb_emp: int = 0
    dlb_joining_date: Optional[date] = None
    dlb_release_date: Optional[date] = None
    dlb_gender: Optional[int] = None
    dlb_dob_date: Optional[date] = None
    dlb_bank_name: Optional[str] = None
    dlb_designation: Optional[str] = None
    dlb_uan_number: Optional[str] = None
    dlb_bank_holder_name: Optional[str] = None
    dlb_esi_number: Optional[str] = None
    dlb_pancard_number: Optional[str] = None
    dlb_basic_salary: Optional[Decimal] = None
    dlb_hra: Optional[Decimal] = None
    dlb_transport_allowance: Optional[Decimal] = None
    dlb_other_allowance: Optional[Decimal] = None
    dlb_epf: int = 0
    dlb_esi: int = 0
    dlb_biometric_code: Optional[str] = None

class AdminLogin(BaseModel):
    dlb_a_email: EmailStr
    dlb_a_password: str


class AdminResponse(BaseModel):
    dlb_a_id: int
    dlb_a_name: str
    dlb_a_email: str
    dlb_a_phone: str
    dlb_admin_id: int
    dlb_branch_name: Optional[str]
    dlb_a_created_date: datetime
    dlb_a_status: int

    class Config:
        from_attributes = True

class LoginResponse(BaseModel):
    message: str
    admin: AdminResponse
