import os
import aiofiles
from typing import List, Dict, Any
from fastapi import UploadFile, HTTPException

class FileService:
    def __init__(self, upload_dir: str = "uploads"):
        self.upload_dir = upload_dir
        self.allowed_extensions = {'.pdf', '.txt', '.doc', '.docx', '.md'}
        os.makedirs(upload_dir, exist_ok=True)
    
    def validate_file(self, file: UploadFile) -> bool:
        """
        Validate if the uploaded file is allowed
        """
        file_extension = os.path.splitext(file.filename)[1].lower()
        return file_extension in self.allowed_extensions
    
    def get_file_info(self, file: UploadFile, content: bytes) -> Dict[str, Any]:
        """
        Get file information
        """
        return {
            "original_name": file.filename,
            "size": len(content),
            "type": file.content_type,
            "extension": os.path.splitext(file.filename)[1].lower()
        }
    
    async def save_file(self, file: UploadFile, unique_filename: str) -> str:
        """
        Save uploaded file to disk
        """
        file_path = os.path.join(self.upload_dir, unique_filename)
        
        async with aiofiles.open(file_path, 'wb') as f:
            content = await file.read()
            await f.write(content)
        
        return file_path
    
    def get_uploaded_files(self) -> List[str]:
        """
        Get list of uploaded files
        """
        if not os.path.exists(self.upload_dir):
            return []
        
        return [f for f in os.listdir(self.upload_dir) if os.path.isfile(os.path.join(self.upload_dir, f))] 