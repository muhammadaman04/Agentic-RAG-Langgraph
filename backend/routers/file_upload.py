from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
import os
import aiofiles
from typing import List
import uuid

router = APIRouter(prefix="/api", tags=["file-upload"])

# Ensure uploads directory exists
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@router.post("/upload-documents")
async def upload_documents(files: List[UploadFile] = File(...)):
    """
    Upload multiple documents for processing
    """
    if not files:
        raise HTTPException(status_code=400, detail="No files provided")
    
    uploaded_files = []
    
    for file in files:
        # Validate file type (allow common document formats)
        allowed_extensions = {'.pdf', '.txt', '.doc', '.docx', '.md'}
        file_extension = os.path.splitext(file.filename)[1].lower()
        
        if file_extension not in allowed_extensions:
            raise HTTPException(
                status_code=400, 
                detail=f"File type {file_extension} not allowed. Allowed types: {allowed_extensions}"
            )
        
        # Generate unique filename
        unique_filename = f"{uuid.uuid4()}_{file.filename}"
        file_path = os.path.join(UPLOAD_DIR, unique_filename)
        
        try:
            # Save file
            async with aiofiles.open(file_path, 'wb') as f:
                content = await file.read()
                await f.write(content)
            
            uploaded_files.append({
                "original_name": file.filename,
                "saved_name": unique_filename,
                "size": len(content),
                "type": file.content_type
            })
            
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error saving file {file.filename}: {str(e)}")
    
    return JSONResponse({
        "message": f"Successfully uploaded {len(uploaded_files)} documents",
        "files": uploaded_files,
        "session_id": str(uuid.uuid4())  # Generate session ID for chatbot
    })

@router.get("/health")
async def health_check():
    """
    Health check endpoint
    """
    return {"status": "healthy", "message": "File upload service is running"} 