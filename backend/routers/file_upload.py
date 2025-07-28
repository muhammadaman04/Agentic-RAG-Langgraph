from fastapi import APIRouter, UploadFile, File
from services.file_service import save_uploaded_file, read_file_content

router = APIRouter(prefix="/upload", tags=["File Upload"])

@router.post("/")
def upload_file(file: UploadFile = File(...)):
    filename = save_uploaded_file(file)
    content = read_file_content(filename)
    return {"filename": filename, "message": "File uploaded successfully!"}
