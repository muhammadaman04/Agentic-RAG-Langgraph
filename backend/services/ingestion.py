import os
import io
import uuid
from pathlib import Path
from typing import List, Dict, Any

import fitz  # PyMuPDF
from PIL import Image
import google.generativeai as genai
from pinecone import Pinecone
from langchain.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings

# Replace these with actual env vars or config values
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY", "your-pinecone-api-key")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "multimodal-rag")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "your-gemini-api-key")
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Initialize Gemini for OCR
genai.configure(api_key=GEMINI_API_KEY)
gemini_model = genai.GenerativeModel("gemini-2.0-flash-exp")

# Initialize embeddings
embedding_model = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL,
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True}
)

def get_loader(file_path: str):
    ext = Path(file_path).suffix.lower()
    if ext == ".pdf":
        return PyPDFLoader(file_path)
    if ext in [".docx", ".doc"]:
        return Docx2txtLoader(file_path)
    if ext == ".txt":
        return TextLoader(file_path, encoding="utf-8")
    raise ValueError(f"Unsupported type: {ext}")

def extract_images_from_pdf(pdf_path: str) -> Dict[int, List[bytes]]:
    page_images = {}
    doc = fitz.open(pdf_path)
    for p in range(len(doc)):
        imgs = doc.load_page(p).get_images()
        data = []
        for img in imgs:
            try:
                pix = fitz.Pixmap(doc, img[0])
                if pix.n - pix.alpha < 4:
                    data.append(pix.tobytes("png"))
            except Exception:
                continue
        if data:
            page_images[p] = data
    doc.close()
    return page_images

def perform_ocr(image_bytes: bytes) -> str:
    try:
        img = Image.open(io.BytesIO(image_bytes))
        prompt = """
        Extract all readable text from the image. Preserve structure (tables, labels, captions).
        Return only the extracted text, no commentary.
        """
        resp = gemini_model.generate_content([prompt, img])
        return resp.text.strip() if resp.text else ""
    except Exception:
        return ""

def ingest_documents(file_paths: List[str], enable_ocr: bool = True) -> Dict[str, Any]:
    # Setup Pinecone
    pc = Pinecone(api_key=PINECONE_API_KEY)
    if PINECONE_INDEX_NAME not in pc.list_indexes().names():
        pc.create_index(name=PINECONE_INDEX_NAME, dimension=384, metric="cosine",
                        spec=pc.ServerlessSpec(cloud="aws", region="us-east-1"))
    index = pc.Index(PINECONE_INDEX_NAME)

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

    all_chunk_vectors = []

    docs_loaded = 0
    chunks_created = 0
    ocr_pages = 0

    for path in file_paths:
        loader = get_loader(path)
        docs = loader.load()
        docs_loaded += len(docs)

        for doc in docs:
            doc.metadata.update({"source_file": path, "file_type": Path(path).suffix.lower()})
            # OCR enhancement if PDF
            if enable_ocr and doc.metadata["file_type"] == ".pdf":
                page_num = doc.metadata.get("page", 0)
                imgs = extract_images_from_pdf(path).get(page_num, [])
                if imgs:
                    texts = [perform_ocr(img) for img in imgs if perform_ocr(img)]
                    if texts:
                        doc.page_content += "\n\n--- OCR Content ---\n\n" + "\n\n".join(texts)
                        doc.metadata["has_ocr"] = True
                        ocr_pages += 1
                else:
                    doc.metadata["has_ocr"] = False

            # Chunk
            chunks = splitter.split_documents([doc])
            for idx, chunk in enumerate(chunks):
                emb = embedding_model.embed_query(chunk.page_content)
                vector = {
                    "id": str(uuid.uuid4()),
                    "values": emb,
                    "metadata": {
                        "text": chunk.page_content,
                        "source_file": doc.metadata["source_file"],
                        "file_type": doc.metadata["file_type"],
                        "has_ocr": doc.metadata.get("has_ocr", False),
                        "chunk_index": idx
                    }
                }
                all_chunk_vectors.append(vector)
                chunks_created += 1

    # Upsert batches
    for i in range(0, len(all_chunk_vectors), 100):
        batch = all_chunk_vectors[i : i + 100]
        index.upsert(vectors=batch)

    return {
        "files_processed": len(file_paths),
        "docs_loaded": docs_loaded,
        "ocr_pages": ocr_pages,
        "chunks_created": chunks_created
    }
