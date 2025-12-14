import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import logging
import sys

from app.api.v1.admin_auth import admin_routes
from app.api.v1.students import str_router
from app.api.v1.attendance import router
from app.database import database

# Configure logging with UTF-8 encoding
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('app.log', encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Starting application...")
    try:
        await database.connect()
        await database.create_tables()
        print("✅ Database connected and tables created")

        # Initialize face service in background
        from app.services.face_recognition import get_face_service
        print(" Initializing face recognition service...")
        face_service = get_face_service()

        # Check service status after a few seconds
        import asyncio
        await asyncio.sleep(3)

        if face_service and face_service.is_ready():
            print("✅ Face recognition service initialized successfully")
            status_info = face_service.get_status()
            logger.info(f"Face service status: {status_info}")
        else:
            print("⚠️ Face recognition service may not be fully initialized")
            if face_service:
                status_info = face_service.get_status()
                logger.warning(f"Face service status: {status_info}")
                if status_info.get('error'):
                    logger.error(f"Face service error: {status_info.get('error')}")

    except Exception as e:
        logger.error(f"Error during startup: {e}")
        print(f"❌ Startup error: {e}")

    print("✅ Application startup complete")
    yield

    print(" Shutting down application...")
    try:
        await database.disconnect()
        print("✅ Database disconnected")
    except Exception as e:
        logger.error(f"Error during shutdown: {e}")
    print(" Application shutdown complete")


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

# Include routers
app.include_router(admin_routes)
app.include_router(str_router)
app.include_router(router)


@app.get("/")
async def root():
    return {
        "message": "Welcome to Admin Management API",
        "version": "1.0.0",
        "endpoints": {
            "students": "/students",
            "attendance": "/attendance",
            "admin": "/admin",
            "health": "/health",
            "face_service_status": "/attendance/health"
        }
    }


@app.get("/health")
async def health_check():
    try:
        db_status = await database.check_connection()

        # Check face service
        from app.services.face_recognition import get_face_service
        face_service = get_face_service()

        face_status = "unknown"
        if face_service:
            status_info = face_service.get_status()
            face_status = status_info.get("status", "unknown")

        return {
            "status": "healthy" if db_status and face_status == "ready" else "degraded",
            "database": "connected" if db_status else "disconnected",
            "face_service": face_status
        }
    except Exception as e:
        logger.error(f"Health check error: {e}")
        return {
            "status": "error",
            "database": "error",
            "face_service": "error"
        }


# Error handler
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    logger.error(f"Unhandled exception: {exc}")
    return {
        "success": False,
        "error": "Internal server error",
        "detail": str(exc) if str(exc) else "Unknown error"
    }


if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)