

import os
import io
import base64
from typing import List, Dict, Any, Tuple, Union, Optional
import uuid
from pathlib import Path
import logging
import asyncio
from concurrent.futures import ThreadPoolExecutor

# Core libraries
import fitz  # PyMuPDF
from PIL import Image
import google.generativeai as genai
from pinecone import Pinecone
from langchain.embeddings import HuggingFaceEmbeddings

# LangChain components
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
from langchain.document_loaders.base import BaseLoader

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
SUPPORTED_EXTENSIONS = {'.pdf', '.docx', '.doc', '.txt'}

class IngestionError(Exception):
    """Custom exception for ingestion errors"""
    pass

def create_clients(pinecone_api_key: str, 
                  pinecone_index_name: str, 
                  gemini_api_key: str,
                  embedding_model_name: str = EMBEDDING_MODEL):
    """Create and return all necessary clients"""
    try:
        # Initialize Pinecone
        logger.info("Creating Pinecone client...")
        pc = Pinecone(api_key=pinecone_api_key)
        
        # Check if index exists, create if not
        if pinecone_index_name not in pc.list_indexes().names():
            logger.info(f"Creating Pinecone index: {pinecone_index_name}")
            pc.create_index(
                name=pinecone_index_name,
                dimension=384,  # Dimension for all-MiniLM-L6-v2
                metric='cosine',
                spec=pc.ServerlessSpec(
                    cloud='aws',
                    region='us-east-1'
                )
            )
        
        pinecone_index = pc.Index(pinecone_index_name)
        
        # Initialize Gemini
        logger.info("Creating Gemini client...")
        genai.configure(api_key=gemini_api_key)
        gemini_model = genai.GenerativeModel('gemini-2.0-flash-exp')
        
        # Initialize embedding model
        logger.info(f"Creating embedding model: {embedding_model_name}")
        embedding_model = HuggingFaceEmbeddings(
            model_name=embedding_model_name,
            model_kwargs={'device': 'cpu'},  # Change to 'cuda' if you have GPU
            encode_kwargs={'normalize_embeddings': True}
        )
        
        logger.info("All clients created successfully")
        return pinecone_index, gemini_model, embedding_model
        
    except Exception as e:
        logger.error(f"Error creating clients: {e}")
        raise IngestionError(f"Failed to create clients: {e}")

def get_document_loader(file_path: str) -> BaseLoader:
    """Get appropriate LangChain loader based on file extension"""
    file_extension = Path(file_path).suffix.lower()
    
    if file_extension == '.pdf':
        return PyPDFLoader(file_path)
    elif file_extension in ['.docx', '.doc']:
        return Docx2txtLoader(file_path)
    elif file_extension == '.txt':
        return TextLoader(file_path, encoding='utf-8')
    else:
        raise IngestionError(f"Unsupported file type: {file_extension}")

def load_documents_with_langchain(file_paths: List[str]) -> List[Document]:
    """Load documents using appropriate LangChain loaders"""
    if not file_paths:
        raise IngestionError("No file paths provided")
    
    all_docs = []
    failed_files = []
    
    for file_path in file_paths:
        logger.info(f"Loading {file_path} with LangChain loader...")
        
        try:
            if not os.path.exists(file_path):
                logger.error(f"File not found: {file_path}")
                failed_files.append(file_path)
                continue
                
            loader = get_document_loader(file_path)
            docs = loader.load()
            
            # Add file path to metadata
            for doc in docs:
                doc.metadata['source_file'] = file_path
                doc.metadata['file_type'] = Path(file_path).suffix.lower()
                doc.metadata['file_name'] = Path(file_path).name
            
            all_docs.extend(docs)
            logger.info(f"Loaded {len(docs)} pages/sections from {Path(file_path).name}")
            
        except Exception as e:
            logger.error(f"Error loading {file_path}: {e}")
            failed_files.append(file_path)
            continue
    
    if failed_files:
        logger.warning(f"Failed to load {len(failed_files)} files: {failed_files}")
    
    if not all_docs:
        raise IngestionError("No documents were loaded successfully")
    
    return all_docs

def extract_images_from_pdf(pdf_path: str) -> Dict[int, List[bytes]]:
    """Extract images from PDF pages for OCR processing"""
    try:
        doc = fitz.open(pdf_path)
        page_images = {}
        
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            image_list = page.get_images()
            images_data = []
            
            if image_list:
                for img_index, img in enumerate(image_list):
                    try:
                        xref = img[0]
                        pix = fitz.Pixmap(doc, xref)
                        
                        if pix.n - pix.alpha < 4:  # GRAY or RGB
                            img_data = pix.tobytes("png")
                            images_data.append(img_data)
                        pix = None
                    except Exception as e:
                        logger.error(f"Error extracting image {img_index} from page {page_num + 1}: {e}")
            
            if images_data:
                page_images[page_num] = images_data
        
        doc.close()
        return page_images
        
    except Exception as e:
        logger.error(f"Error extracting images from PDF {pdf_path}: {e}")
        return {}

def perform_ocr_with_gemini(image_data: bytes, gemini_model) -> str:
    """Perform OCR on image using Gemini Flash 2.0"""
    try:
        # Convert image to PIL Image
        image = Image.open(io.BytesIO(image_data))
        
        # Prepare the prompt
        prompt = """
        Extract all text from this image. Please provide:
        1. All readable text in the image
        2. Maintain the original structure and formatting as much as possible
        3. If there are tables, preserve the tabular structure
        4. Include any captions, labels, or annotations
        
        Return only the extracted text without any additional commentary.
        """
        
        # Generate content with image
        response = gemini_model.generate_content([prompt, image])
        
        return response.text.strip() if response.text else ""
        
    except Exception as e:
        logger.error(f"Error performing OCR: {e}")
        return ""

def enhance_documents_with_ocr(documents: List[Document], gemini_model) -> List[Document]:
    """Enhance documents with OCR text from images (for PDFs)"""
    enhanced_docs = []
    
    for doc in documents:
        enhanced_doc = Document(
            page_content=doc.page_content,
            metadata=doc.metadata.copy()
        )
        
        # Only process PDFs for OCR
        if doc.metadata.get('file_type', '').lower() == '.pdf':
            source_file = doc.metadata['source_file']
            page_num = doc.metadata.get('page', 0)
            
            logger.info(f"Checking for images in {Path(source_file).name}, page {page_num + 1}...")
            
            # Extract images from this specific page
            try:
                page_images = extract_images_from_pdf(source_file)
                
                if page_num in page_images:
                    logger.info(f"Found {len(page_images[page_num])} images on page {page_num + 1}")
                    
                    ocr_texts = []
                    for i, img_data in enumerate(page_images[page_num]):
                        logger.info(f"Performing OCR on image {i + 1}...")
                        ocr_text = perform_ocr_with_gemini(img_data, gemini_model)
                        if ocr_text:
                            ocr_texts.append(ocr_text)
                    
                    # Combine original text with OCR text
                    if ocr_texts:
                        enhanced_content = enhanced_doc.page_content
                        if enhanced_content.strip():
                            enhanced_content += "\n\n--- OCR Content ---\n\n"
                        else:
                            enhanced_content = "--- OCR Content ---\n\n"
                        
                        enhanced_content += "\n\n".join(ocr_texts)
                        enhanced_doc.page_content = enhanced_content
                        enhanced_doc.metadata['has_ocr'] = True
                        enhanced_doc.metadata['ocr_images_count'] = len(ocr_texts)
                        
                        logger.info(f"Enhanced page with {len(ocr_texts)} OCR texts")
            
            except Exception as e:
                logger.error(f"Error processing images for {source_file}: {e}")
                enhanced_doc.metadata['has_ocr'] = False
        else:
            enhanced_doc.metadata['has_ocr'] = False
        
        enhanced_docs.append(enhanced_doc)
    
    return enhanced_docs

def create_embeddings_and_chunks(documents: List[Document], embedding_model) -> List[Dict[str, Any]]:
    """Create text chunks and embeddings for vector store"""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    
    all_chunks = []
    
    logger.info(f"Processing {len(documents)} documents for chunking and embedding...")
    
    for doc_idx, doc in enumerate(documents):
        if not doc.page_content.strip():
            continue
        
        logger.info(f"Processing document {doc_idx + 1}/{len(documents)}: {doc.metadata.get('file_name', 'Unknown')}")
        
        # Split into chunks
        chunks = text_splitter.split_documents([doc])
        
        for i, chunk in enumerate(chunks):
            try:
                # Create embedding
                embedding = embedding_model.embed_query(chunk.page_content)
                
                # Prepare chunk data
                chunk_data = {
                    'id': str(uuid.uuid4()),
                    'text': chunk.page_content,
                    'embedding': embedding,
                    'metadata': {
                        **chunk.metadata,
                        'chunk_index': i,
                        'total_chunks_in_doc': len(chunks),
                        'document_index': doc_idx
                    }
                }
                
                all_chunks.append(chunk_data)
                
            except Exception as e:
                logger.error(f"Error creating embedding for chunk {i} in document {doc_idx}: {e}")
                continue
        
        logger.info(f"Created {len(chunks)} chunks from this document")
    
    return all_chunks

def store_in_pinecone(chunks: List[Dict[str, Any]], pinecone_index):
    """Store embeddings in Pinecone vector database"""
    if not chunks:
        raise IngestionError("No chunks to store")
    
    logger.info(f"Storing {len(chunks)} chunks in Pinecone...")
    
    # Prepare vectors for upsert
    vectors = []
    
    for chunk in chunks:
        # Truncate text if too long for metadata
        text_content = chunk['text'][:1000] if len(chunk['text']) > 1000 else chunk['text']
        
        vector = {
            'id': chunk['id'],
            'values': chunk['embedding'],
            'metadata': {
                'text': text_content,
                'source_file': chunk['metadata']['source_file'],
                'file_name': chunk['metadata'].get('file_name', ''),
                'file_type': chunk['metadata']['file_type'],
                'page': chunk['metadata'].get('page', 0),
                'chunk_index': chunk['metadata']['chunk_index'],
                'has_ocr': chunk['metadata'].get('has_ocr', False),
                'ocr_images_count': chunk['metadata'].get('ocr_images_count', 0),
                'document_index': chunk['metadata'].get('document_index', 0)
            }
        }
        vectors.append(vector)
    
    # Upsert in batches
    batch_size = 100
    failed_batches = 0
    
    for i in range(0, len(vectors), batch_size):
        batch = vectors[i:i + batch_size]
        try:
            pinecone_index.upsert(vectors=batch)
            logger.info(f"Upserted batch {i//batch_size + 1}/{(len(vectors) + batch_size - 1)//batch_size}")
        except Exception as e:
            logger.error(f"Error upserting batch {i//batch_size + 1}: {e}")
            failed_batches += 1
    
    if failed_batches > 0:
        logger.warning(f"Failed to upsert {failed_batches} batches")
    
    logger.info("Successfully stored embeddings in Pinecone!")

def validate_file_paths(file_paths: List[str]) -> List[str]:
    """Validate file paths and filter supported extensions"""
    valid_files = []
    
    for file_path in file_paths:
        file_path = str(file_path).strip()
        
        # Check if file exists
        if not os.path.exists(file_path):
            logger.warning(f"File not found: {file_path}")
            continue
        
        # Check if file extension is supported
        file_extension = Path(file_path).suffix.lower()
        if file_extension not in SUPPORTED_EXTENSIONS:
            logger.warning(f"Unsupported file type {file_extension}: {file_path}")
            continue
        
        valid_files.append(file_path)
    
    return valid_files

def ingest_documents(file_paths: List[str], 
                    enable_ocr: bool = True,
                    pinecone_api_key: str = None,
                    pinecone_index_name: str = "multimodal-rag",
                    gemini_api_key: str = None) -> Dict[str, Any]:
    """
    Main function to ingest multiple documents into the RAG system
    """
    try:
        logger.info("="*60)
        logger.info("MULTIMODAL RAG BATCH INGESTION STARTED")
        logger.info(f"Processing {len(file_paths)} documents")
        logger.info("="*60)
        
        # Get API keys from environment if not provided
        if not pinecone_api_key:
            pinecone_api_key = os.getenv("PINECONE_API_KEY")
        if not gemini_api_key:
            gemini_api_key = os.getenv("GEMINI_API_KEY")
        
        # Validate API keys
        if not pinecone_api_key or not gemini_api_key:
            raise IngestionError("Missing required API keys (PINECONE_API_KEY, GEMINI_API_KEY)")
        
        # Create clients for this ingestion session
        pinecone_index, gemini_model, embedding_model = create_clients(
            pinecone_api_key=pinecone_api_key,
            pinecone_index_name=pinecone_index_name,
            gemini_api_key=gemini_api_key
        )
        
        # Validate file paths
        valid_file_paths = validate_file_paths(file_paths)
        if not valid_file_paths:
            raise IngestionError("No valid file paths found")
        
        logger.info(f"Valid files to process: {len(valid_file_paths)}")
        
        # Load documents using LangChain loaders
        logger.info("Loading documents with LangChain loaders...")
        documents = load_documents_with_langchain(valid_file_paths)
        
        # Enhance with OCR if enabled
        if enable_ocr:
            logger.info(f"Enhancing {len(documents)} documents with OCR...")
            documents = enhance_documents_with_ocr(documents, gemini_model)
        else:
            logger.info("OCR processing disabled, using text-only content...")
            for doc in documents:
                doc.metadata['has_ocr'] = False
        
        # Create embeddings and chunks
        logger.info("Creating embeddings and chunks...")
        chunks = create_embeddings_and_chunks(documents, embedding_model)
        
        if not chunks:
            raise IngestionError("No chunks were created")
        
        # Store in Pinecone
        store_in_pinecone(chunks, pinecone_index)
        
        # Summary statistics
        pdf_docs = [d for d in documents if d.metadata.get('file_type', '').lower() == '.pdf']
        ocr_enhanced_docs = [d for d in documents if d.metadata.get('has_ocr', False)]
        
        result = {
            'success': True,
            'files_processed': len(valid_file_paths),
            'documents_loaded': len(documents),
            'pdf_documents': len(pdf_docs),
            'ocr_enhanced_documents': len(ocr_enhanced_docs),
            'chunks_created': len(chunks),
            'file_types': list(set([Path(fp).suffix.lower() for fp in valid_file_paths])),
            'failed_files': len(file_paths) - len(valid_file_paths)
        }
        
        logger.info("="*60)
        logger.info("BATCH INGESTION COMPLETED SUCCESSFULLY!")
        logger.info(f"Files processed: {result['files_processed']}")
        logger.info(f"Documents loaded: {result['documents_loaded']}")
        logger.info(f"PDF documents: {result['pdf_documents']}")
        logger.info(f"OCR enhanced documents: {result['ocr_enhanced_documents']}")
        logger.info(f"Total chunks created: {result['chunks_created']}")
        logger.info("="*60)
        
        return result
        
    except Exception as e:
        logger.error(f"Error during ingestion: {e}")
        return {
            'success': False,
            'error': str(e),
            'files_processed': 0,
            'documents_loaded': 0,
            'chunks_created': 0
        }

def ingest_single_document(file_path: str, 
                          enable_ocr: bool = True,
                          pinecone_api_key: str = None,
                          pinecone_index_name: str = "multimodal-rag",
                          gemini_api_key: str = None) -> Dict[str, Any]:
    """Convenience function for single document ingestion"""
    return ingest_documents(
        file_paths=[file_path], 
        enable_ocr=enable_ocr,
        pinecone_api_key=pinecone_api_key,
        pinecone_index_name=pinecone_index_name,
        gemini_api_key=gemini_api_key
    )

def process_document_folder(folder_path: str, 
                          file_extensions: List[str] = None,
                          enable_ocr: bool = True,
                          pinecone_api_key: str = None,
                          pinecone_index_name: str = "multimodal-rag",
                          gemini_api_key: str = None) -> Dict[str, Any]:
    """
    Process all documents in a folder automatically
    """
    if file_extensions is None:
        file_extensions = list(SUPPORTED_EXTENSIONS)
    
    folder = Path(folder_path)
    
    if not folder.exists():
        return {
            'success': False,
            'error': f"Folder {folder_path} does not exist",
            'files_processed': 0
        }
    
    files_to_process = []
    
    # Find all files with supported extensions
    for ext in file_extensions:
        # Use both case variations to catch files like .PDF, .Pdf, etc.
        files_to_process.extend(folder.glob(f"*{ext.lower()}"))
        files_to_process.extend(folder.glob(f"*{ext.upper()}"))
    
    # Remove duplicates and convert to strings
    file_paths = list(set([str(f) for f in files_to_process]))
    
    if not file_paths:
        return {
            'success': False,
            'error': f"No supported files found in {folder_path}",
            'files_processed': 0
        }
    
    logger.info(f"Found {len(file_paths)} files to process:")
    for fp in sorted(file_paths):
        logger.info(f"  - {Path(fp).name}")
    
    return ingest_documents(
        file_paths=file_paths, 
        enable_ocr=enable_ocr,
        pinecone_api_key=pinecone_api_key,
        pinecone_index_name=pinecone_index_name,
        gemini_api_key=gemini_api_key
    )

def check_ingestion_status(pinecone_api_key: str = None,
                          pinecone_index_name: str = "multimodal-rag") -> Dict[str, Any]:
    """Check the current status of the vector database"""
    try:
        # Get API key from environment if not provided
        if not pinecone_api_key:
            pinecone_api_key = os.getenv("PINECONE_API_KEY")
        
        if not pinecone_api_key:
            return {
                'success': False,
                'error': 'Pinecone API key not provided'
            }
        
        # Create Pinecone client
        pc = Pinecone(api_key=pinecone_api_key)
        
        if pinecone_index_name not in pc.list_indexes().names():
            return {
                'success': False,
                'error': f'Index {pinecone_index_name} does not exist'
            }
        
        index = pc.Index(pinecone_index_name)
        stats = index.describe_index_stats()
        
        return {
            'success': True,
            'total_vectors': stats['total_vector_count'],
            'dimension': stats['dimension'],
            'index_fullness': stats.get('index_fullness', 0),
            'namespaces': stats.get('namespaces', {})
        }
    except Exception as e:
        logger.error(f"Error checking ingestion status: {e}")
        return {
            'success': False,
            'error': str(e)
        }

def clear_vector_database(pinecone_api_key: str = None,
                         pinecone_index_name: str = "multimodal-rag") -> Dict[str, Any]:
    """Clear all vectors from the database (use with caution)"""
    try:
        # Get API key from environment if not provided
        if not pinecone_api_key:
            pinecone_api_key = os.getenv("PINECONE_API_KEY")
        
        if not pinecone_api_key:
            return {
                'success': False,
                'error': 'Pinecone API key not provided'
            }
        
        # Create Pinecone client
        pc = Pinecone(api_key=pinecone_api_key)
        
        if pinecone_index_name not in pc.list_indexes().names():
            return {
                'success': False,
                'error': f'Index {pinecone_index_name} does not exist'
            }
        
        index = pc.Index(pinecone_index_name)
        index.delete(delete_all=True)
        
        logger.info("Vector database cleared successfully")
        return {
            'success': True,
            'message': 'Vector database cleared successfully'
        }
    except Exception as e:
        logger.error(f"Error clearing vector database: {e}")
        return {
            'success': False,
            'error': str(e)
        }