from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routers import file_upload

app = FastAPI(
    title="IntelliRAG API",
    description="Intelligent RAG System API",
    version="1.0.0"
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],  # React dev servers
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(file_upload.router)

@app.get("/")
def read_root():
    return {
        "message": "IntelliRAG API is running!",
        "docs": "/docs",
        "version": "1.0.0"
    }

@app.get("/health")
def health_check():
    return {"status": "healthy", "service": "IntelliRAG API"}
