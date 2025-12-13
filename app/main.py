from app.api.v1.admin_auth import admin_routes
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from app.api.v1.attendance import router
from app.api.v1.students import str_router
from app.database import database

# Define lifespan function correctly
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("🔄 Starting application...")
    await database.connect()
    await database.create_tables()
    print("✅ Application startup complete")
    yield
    # Shutdown
    print("🔄 Shutting down application...")
    await database.disconnect()
    print("✅ Application shutdown complete")

app = FastAPI(
    title="Admin Management API",
    description="API for admin registration and login",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(admin_routes)

@app.get("/")
async def read_root():
    return {"message": "Welcome to Admin Management API"}

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "database_connected": database.is_connected
    }

app.include_router(admin_routes)
app.include_router(str_router)
app.include_router(router)





# # Attendance endpoint
# @app.post("/attendance")
# async def mark_attendance(
#         image: UploadFile = File(...),
#         db: AsyncSession = Depends(get_db)
# ):
#     """Mark attendance by detecting face in image"""
#     start_time = time.time()
#
#     try:
#         print(f"\n{'=' * 60}")
#         print(f"MARKING ATTENDANCE")
#         print(f"{'=' * 60}")
#
#         # Validate image
#         allowed_types = {'image/jpeg', 'image/jpg', 'image/png', 'image/webp'}
#         if image.content_type not in allowed_types:
#             raise HTTPException(
#                 status_code=400,
#                 detail=f"Unsupported file type '{image.content_type}'. Use JPEG, PNG, or WebP"
#             )
#
#         # Read image
#         image_bytes = await image.read()
#         if len(image_bytes) < 1024:
#             raise HTTPException(status_code=400, detail="Image file too small or corrupted")
#
#         # Process image
#         result = face_service.process_single_image(image_bytes, require_single_face=False)
#
#         if not result["success"]:
#             raise HTTPException(status_code=400, detail=result.get("error", "Face detection failed"))
#
#         total_faces = result["faces_detected"]
#         if total_faces == 0:
#             print("no fase")
#
#         # Get all active users
#         users = await database.get_all_active_users()
#         if not users:
#             print("no user")
#
#         # Prepare user data
#         user_embeddings_dict = {}
#         user_info_dict = {}
#
#         for user in users:
#             user_id = str(user.id)
#             embeddings = user.embeddings or []
#             if embeddings:
#                 user_embeddings_dict[user_id] = embeddings
#                 user_info_dict[user_id] = {
#                     "name": user.name,
#                     "phone": user.phone,
#                     "email": user.email
#                 }
#
#         # Compare faces
#         query_embeddings = result["embeddings"]
#         matched_users = []
#         attendance_recorded = 0
#         matched_faces = set()
#
#         for user_id, user_embeddings in user_embeddings_dict.items():
#             user_matches = []
#
#             for face_idx, query_emb in enumerate(query_embeddings):
#                 if face_idx in matched_faces:
#                     continue
#
#                 # Find best similarity
#                 best_similarity = 0.0
#                 for user_emb in user_embeddings:
#                     similarity, _ = face_service.compare_faces(query_emb, user_emb, threshold=0.0)
#                     if similarity > best_similarity:
#                         best_similarity = similarity
#
#                 if best_similarity >= 0.6:
#                     user_matches.append({
#                         "face_index": face_idx,
#                         "similarity": float(best_similarity),
#                         "bounding_box": result["bounding_boxes"][face_idx] if face_idx < len(
#                             result["bounding_boxes"]) else None
#                     })
#                     matched_faces.add(face_idx)
#
#             if user_matches:
#                 # Record attendance
#                 now = datetime.utcnow()
#                 today_start = datetime(now.year, now.month, now.day)
#
#                 # Check if attendance already recorded today
#                 check_query = select(Attendance).where(
#                     Attendance.user_id == int(user_id),
#                     Attendance.timestamp >= today_start
#                 )
#                 result_check = await db.execute(check_query)
#                 existing_attendance = result_check.scalar_one_or_none()
#
#                 if not existing_attendance:
#                     # Create attendance record
#                     attendance_record = Attendance(
#                         user_id=int(user_id),
#                         user_name=user_info_dict[user_id]["name"],
#                         user_phone=user_info_dict[user_id]["phone"],
#                         timestamp=now,
#                         confidence=float(max(match["similarity"] for match in user_matches)),
#                         image_hash=calculate_image_hash(image_bytes),
#                         faces_matched=len(user_matches),
#                         matched_faces=user_matches
#                     )
#
#                     db.add(attendance_record)
#
#                     # Update user record
#                     user = await database.get_user_by_id(int(user_id))
#                     if user:
#                         user.last_attendance = now
#                         user.total_attendance = user.total_attendance + 1
#                         db.add(user)
#
#                     await db.commit()
#                     attendance_recorded += 1
#
#                 # Add to matched users
#                 matched_users.append({
#                     "user_id": user_id,
#                     "name": user_info_dict[user_id]["name"],
#                     "phone": user_info_dict[user_id]["phone"],
#                     "matches": user_matches,
#                     "attendance_recorded": not existing_attendance,
#                     "already_marked_today": bool(existing_attendance)
#                 })
#
#         unknown_faces = total_faces - len(matched_faces)
#         processing_time = time.time() - start_time
#
#         # Prepare response message
#         message_parts = []
#         if attendance_recorded > 0:
#             message_parts.append(f"Attendance recorded for {attendance_recorded} user(s)")
#         if matched_users and not attendance_recorded:
#             message_parts.append(f"{len(matched_users)} user(s) already marked attendance today")
#         if unknown_faces > 0:
#             message_parts.append(f"{unknown_faces} unknown face(s) detected")
#
#         message = ". ".join(message_parts) if message_parts else "Attendance processed"
#
#         return {"success": True, "message": message}
#
#     except HTTPException:
#         raise
#     except Exception as e:
#         print(f"💥 Attendance error: {e}")
#         traceback.print_exc()
#         raise HTTPException(status_code=500, detail=str(e))





if __name__ == "__main__":
    import uvicorn

    print("\n🚀 Starting Face Attendance System (MySQL Version)...")
    print("🌐 API Documentation: http://localhost:8000/docs")
    print("📊 Health Check: http://localhost:8000/health")
    print("\n📝 Registration endpoint: POST http://localhost:8000/register")
    print("✏️ User management: GET/PUT/DELETE http://localhost:8000/users/{id}")
    print("🎯 Test endpoint: POST http://localhost:8000/test-detection\n")

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )