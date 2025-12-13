from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from app.api.v1.admin_auth import admin_routes
from app.api.v1.students import str_router
from app.api.v1.attendance import router
from app.database import database


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("🔄 Starting application...")
    await database.connect()
    await database.create_tables()
    print("✅ Application startup complete")
    yield
    print("🔄 Shutting down application...")
    await database.disconnect()
    print("✅ Application shutdown complete")


app = FastAPI(
    title="Admin Management API",
    description="API for admin registration, students & attendance",
    version="1.0.0",
    lifespan=lifespan
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


app.include_router(admin_routes)
app.include_router(str_router)
app.include_router(router)


@app.get("/")
async def root():
    return {"message": "Welcome to Admin Management API"}

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "database_connected": database.is_connected
    }
