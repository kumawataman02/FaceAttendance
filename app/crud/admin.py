# app/crud/admin.py
import logging
from typing import Optional

from sqlalchemy.future import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.exc import SQLAlchemyError

from app.models.admin import WifiAdmin
from app.core.security import hash_password, verify_password
from app.schemas.admin_schemas import AdminCreate

# Setup logger
logger = logging.getLogger(__name__)


async def get_admin_by_email(db: AsyncSession, email: str) -> Optional[WifiAdmin]:
    """
    Get admin by email.

    Args:
        db: Database session
        email: Admin email

    Returns:
        WifiAdmin instance or None
    """
    try:
        logger.debug(f"Querying admin by email: {email}")
        result = await db.execute(select(WifiAdmin).where(WifiAdmin.dlb_a_email == email))
        admin = result.scalar_one_or_none()

        if admin:
            logger.debug(f"Found admin with email: {email}")
        else:
            logger.debug(f"No admin found with email: {email}")

        return admin

    except SQLAlchemyError as e:
        logger.error(f"Database error querying admin by email {email}: {str(e)}", exc_info=True)
        raise


async def get_admin_by_id(db: AsyncSession, admin_id: int) -> Optional[WifiAdmin]:
    """
    Get admin by ID.

    Args:
        db: Database session
        admin_id: Admin ID

    Returns:
        WifiAdmin instance or None
    """
    try:
        logger.debug(f"Querying admin by ID: {admin_id}")
        result = await db.execute(
            select(WifiAdmin).where(WifiAdmin.dlb_a_id == admin_id)
        )
        admin = result.scalar_one_or_none()

        if admin:
            logger.debug(f"Found admin with ID: {admin_id}")
        else:
            logger.debug(f"No admin found with ID: {admin_id}")

        return admin

    except SQLAlchemyError as e:
        logger.error(f"Database error querying admin by ID {admin_id}: {str(e)}", exc_info=True)
        raise


async def get_admin_by_phone_number(db: AsyncSession, phone_number: str) -> Optional[WifiAdmin]:
    """
    Get admin by phone number.

    Args:
        db: Database session
        phone_number: Admin phone number

    Returns:
        WifiAdmin instance or None
    """
    try:
        logger.debug(f"Querying admin by phone number: {phone_number}")
        result = await db.execute(
            select(WifiAdmin).where(WifiAdmin.dlb_a_phone_number == phone_number)
        )
        admin = result.scalar_one_or_none()

        if admin:
            logger.debug(f"Found admin with phone number: {phone_number}")
        else:
            logger.debug(f"No admin found with phone number: {phone_number}")

        return admin

    except SQLAlchemyError as e:
        logger.error(f"Database error querying admin by phone {phone_number}: {str(e)}", exc_info=True)
        raise


async def create_admin(db: AsyncSession, admin: AdminCreate) -> WifiAdmin:
    """
    Create a new admin.

    Args:
        db: Database session
        admin: Admin creation data

    Returns:
        Created WifiAdmin instance

    Raises:
        SQLAlchemyError: If database operation fails
    """
    try:
        logger.info(f"Creating new admin with email: {admin.dlb_a_email}")

        # Hash the password
        hashed_password = hash_password(admin.dlb_a_password)

        # Convert Pydantic model to dict and update password
        admin_data = admin.dict(exclude={'dlb_a_password'})
        admin_data['dlb_a_password'] = hashed_password

        # Create SQLAlchemy model instance
        db_admin = WifiAdmin(**admin_data)

        # Add to database
        db.add(db_admin)
        await db.commit()
        await db.refresh(db_admin)

        logger.info(f"Admin created successfully: {admin.dlb_a_email} (ID: {db_admin.dlb_a_id})")
        return db_admin

    except SQLAlchemyError as e:
        logger.error(f"Database error creating admin {admin.dlb_a_email}: {str(e)}", exc_info=True)
        await db.rollback()
        raise
    except Exception as e:
        logger.error(f"Unexpected error creating admin {admin.dlb_a_email}: {str(e)}", exc_info=True)
        await db.rollback()
        raise


async def authenticate_admin(db: AsyncSession, email: str, password: str) -> Optional[WifiAdmin]:
    """
    Authenticate admin credentials.

    Args:
        db: Database session
        email: Admin email
        password: Plain text password

    Returns:
        WifiAdmin instance if authentication successful, else None
    """
    try:
        logger.debug(f"Authenticating admin: {email}")

        # Get admin by email
        admin = await get_admin_by_email(db, email)

        if not admin:
            logger.debug(f"Authentication failed: Admin not found for email: {email}")
            return None

        # Verify password
        if not verify_password(password, admin.dlb_a_password):
            logger.debug(f"Authentication failed: Invalid password for email: {email}")
            return None

        logger.debug(f"Authentication successful for admin: {email}")
        return admin

    except SQLAlchemyError as e:
        logger.error(f"Database error during authentication for {email}: {str(e)}", exc_info=True)
        raise
    except Exception as e:
        logger.error(f"Unexpected error during authentication for {email}: {str(e)}", exc_info=True)
        raise


async def update_admin_status(
        db: AsyncSession,
        admin_id: int,
        status: int
) -> Optional[WifiAdmin]:
    """
    Update admin status.

    Args:
        db: Database session
        admin_id: Admin ID
        status: New status value

    Returns:
        Updated WifiAdmin instance or None
    """
    try:
        logger.info(f"Updating admin status for ID: {admin_id} to {status}")

        admin = await get_admin_by_id(db, admin_id)
        if not admin:
            logger.warning(f"Cannot update status: Admin not found with ID: {admin_id}")
            return None

        admin.dlb_a_status = status
        await db.commit()
        await db.refresh(admin)

        logger.info(f"Admin status updated for ID: {admin_id}")
        return admin

    except SQLAlchemyError as e:
        logger.error(f"Database error updating admin status for ID {admin_id}: {str(e)}", exc_info=True)
        await db.rollback()
        raise
    except Exception as e:
        logger.error(f"Unexpected error updating admin status for ID {admin_id}: {str(e)}", exc_info=True)
        await db.rollback()
        raise