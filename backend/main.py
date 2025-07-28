from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routers import file_upload

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(file_upload.router)

@app.get("/")
def read_root():
    return {"message": "FastAPI backend is working!"}
