from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from typing import List
from app.crud.admin import create_admin, authenticate_admin, get_admin_by_email
from app.dependencies import get_db
from app.schemas.admin_schemas import AdminResponse, AdminCreate, LoginResponse, AdminLogin

admin_routes = APIRouter(prefix="/admin", tags=["admin"])


@admin_routes.post("/register",response_model=AdminResponse,status_code=status.HTTP_201_CREATED)
async def register_admin(admin:AdminCreate,db:AsyncSession = Depends(get_db)):
    # Check if admin already exists
    db_admin = await get_admin_by_email(db, email=admin.dlb_a_email)
    if db_admin:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Admin with this email already exists"
        )

    # Create new admin
    new_admin = await create_admin(db=db, admin=admin)
    return new_admin


@admin_routes.post("/login", response_model=LoginResponse)
async def login_admin(login_data: AdminLogin, db: AsyncSession = Depends(get_db)):
    # Authenticate admin
    admin = await authenticate_admin(db, login_data.dlb_a_email, login_data.dlb_a_password)

    if not admin:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password"
        )

    # Check if admin is active
    if admin.dlb_a_status != 0:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin account is inactive"
        )

    # Prepare response data
    admin_response = AdminResponse(
        dlb_a_id=admin.dlb_a_id,
        dlb_a_name=admin.dlb_a_name,
        dlb_a_email=admin.dlb_a_email,
        dlb_a_phone=admin.dlb_a_phone,
        dlb_admin_id=admin.dlb_admin_id,
        dlb_branch_name=admin.dlb_branch_name,
        dlb_a_created_date=admin.dlb_a_created_date,
        dlb_a_status=admin.dlb_a_status
    )

    return LoginResponse(
        message="Login successful",
        admin=admin_response
    )







