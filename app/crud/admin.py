from sqlalchemy.future import select
from sqlalchemy.ext.asyncio import AsyncSession
from app.models.admin import WifiAdmin
from app.core.security import hash_password, verify_password
from typing import Optional

from app.schemas.admin_schemas import AdminCreate


async def get_admin_by_email(db:AsyncSession, email:str) -> Optional[WifiAdmin]:
    result = await db.execute(select(WifiAdmin).where(WifiAdmin.dlb_a_email == email))
    return result.scalar_one_or_none()


async def get_admin_by_id(db:AsyncSession, admin_id:int) -> Optional[WifiAdmin]:
    result = await db.execute(
        select(WifiAdmin).where(WifiAdmin.dlb_a_id == admin_id)
    )
    return result.scalar_one_or_none()



async def get_admin_by_phone_number(db:AsyncSession, phone_number:str) -> Optional[WifiAdmin]:
    result = await db.execute(
        select(WifiAdmin).where(WifiAdmin.dlb_a_phone_number == phone_number)
    )

    return result.scalar_one_or_none()


async def create_admin(db: AsyncSession, admin: AdminCreate) -> WifiAdmin:
    hashed_password = hash_password(admin.dlb_a_password)

    # Convert Pydantic model to dict and update password
    admin_data = admin.dict(exclude={'dlb_a_password'})
    admin_data['dlb_a_password'] = hashed_password

    db_admin = WifiAdmin(**admin_data)

    db.add(db_admin)
    await db.commit()
    await db.refresh(db_admin)
    return db_admin

async def authenticate_admin(db: AsyncSession, email: str, password: str) -> Optional[WifiAdmin]:
    admin = await get_admin_by_email(db, email)
    if not admin:
        return None
    if not verify_password(password, admin.dlb_a_password):
        return None
    return admin
