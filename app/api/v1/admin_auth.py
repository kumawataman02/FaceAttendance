import logging
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.exc import SQLAlchemyError, IntegrityError

from app.crud.admin import create_admin, authenticate_admin, get_admin_by_email
from app.dependencies import get_db
from app.schemas.admin_schemas import AdminResponse, AdminCreate, LoginResponse, AdminLogin
from app.models.admin import WifiAdmin

# Setup logger
logger = logging.getLogger(__name__)

admin_routes = APIRouter(prefix="/admin", tags=["admin"])


def convert_to_admin_response(admin: WifiAdmin) -> AdminResponse:
    """Convert SQLAlchemy model to Pydantic response model."""
    return AdminResponse(
        dlb_a_id=admin.dlb_a_id,
        dlb_a_name=admin.dlb_a_name,
        dlb_a_email=admin.dlb_a_email,
        dlb_a_phone=admin.dlb_a_phone,
        dlb_admin_id=admin.dlb_admin_id,
        dlb_branch_name=admin.dlb_branch_name,
        dlb_a_created_date=admin.dlb_a_created_date,
        dlb_a_status=admin.dlb_a_status
    )


@admin_routes.post("/register", response_model=AdminResponse, status_code=status.HTTP_201_CREATED)
async def register_admin(
        admin: AdminCreate,
        db: AsyncSession = Depends(get_db)
):
    """
    Register a new admin.

    - **admin**: Admin registration data
    """
    try:
        logger.info(f"Starting admin registration for email: {admin.dlb_a_email}")

        # Check if admin already exists
        db_admin = await get_admin_by_email(db, email=admin.dlb_a_email)
        if db_admin:
            logger.warning(f"Admin registration failed - email already exists: {admin.dlb_a_email}")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Admin with this email already exists"
            )

        # Create new admin
        logger.debug(f"Creating admin with data: {admin.dict(exclude={'dlb_a_password'})}")
        new_admin = await create_admin(db=db, admin=admin)

        # Convert to response model
        admin_response = convert_to_admin_response(new_admin)

        logger.info(f"Admin registered successfully: {admin.dlb_a_email} (ID: {new_admin.dlb_a_id})")
        return admin_response

    except IntegrityError as e:
        logger.error(f"Database integrity error during admin registration: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Database constraint violation. Check your input data."
        )
    except SQLAlchemyError as e:
        logger.error(f"Database error during admin registration: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Database service temporarily unavailable"
        )
    except HTTPException:
        # Re-raise HTTP exceptions so they're handled properly
        raise
    except Exception as e:
        logger.error(f"Unexpected error during admin registration: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred during registration"
        )


@admin_routes.post("/login", response_model=LoginResponse)
async def login_admin(
        login_data: AdminLogin,
        db: AsyncSession = Depends(get_db)
):
    """
    Admin login endpoint.

    - **login_data**: Admin login credentials
    """
    try:
        logger.info(f"Login attempt for email: {login_data.dlb_a_email}")

        # Authenticate admin
        admin = await authenticate_admin(db, login_data.dlb_a_email, login_data.dlb_a_password)

        if not admin:
            logger.warning(f"Failed login attempt for email: {login_data.dlb_a_email}")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid email or password"
            )

        # Check if admin is active (assuming 0 = active)
        if admin.dlb_a_status != 0:
            logger.warning(f"Inactive admin login attempt: {login_data.dlb_a_email}")
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Admin account is inactive"
            )

        # Prepare response data
        admin_response = convert_to_admin_response(admin)

        logger.info(f"Successful login for admin: {login_data.dlb_a_email} (ID: {admin.dlb_a_id})")

        return LoginResponse(
            message="Login successful",
            admin=admin_response,
            timestamp=datetime.utcnow().isoformat()
        )

    except SQLAlchemyError as e:
        logger.error(f"Database error during login: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Database service temporarily unavailable"
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error during login: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred during login"
        )


@admin_routes.get("/health")
async def health_check():
    """Health check endpoint for admin service."""
    return {
        "status": "healthy",
        "service": "admin-api",
        "timestamp": datetime.utcnow().isoformat(),
        "version": "1.0.0"
    }


@admin_routes.get("/{admin_id}", response_model=AdminResponse)
async def get_admin(
        admin_id: int,
        db: AsyncSession = Depends(get_db)
):
    """
    Get admin details by ID.

    - **admin_id**: Admin ID
    """
    try:
        logger.info(f"Fetching admin details for ID: {admin_id}")

        from app.crud.admin import get_admin_by_id

        admin = await get_admin_by_id(db, admin_id)

        if not admin:
            logger.warning(f"Admin not found with ID: {admin_id}")
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Admin not found"
            )

        admin_response = convert_to_admin_response(admin)

        logger.debug(f"Successfully retrieved admin: {admin_id}")
        return admin_response

    except SQLAlchemyError as e:
        logger.error(f"Database error fetching admin {admin_id}: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Database service temporarily unavailable"
        )
    except Exception as e:
        logger.error(f"Unexpected error fetching admin {admin_id}: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred"
        )