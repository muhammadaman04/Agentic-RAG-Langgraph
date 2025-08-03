import os
import io
import base64
from typing import List, Dict, Any, Tuple, Union, Optional
import uuid
from pathlib import Path
import asyncio
from concurrent.futures import ThreadPoolExecutor
import hashlib

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

# Configuration
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
SUPPORTED_EXTENSIONS = {'.pdf', '.docx', '.doc', '.txt'}

class IngestionError(Exception):
    """Custom exception for ingestion errors"""
    pass

class MultimodalRAGClient:
    """
    Centralized client manager for Multimodal RAG system
    This should be initialized once and reused across requests
    """
    def __init__(self, pinecone_api_key: str, pinecone_index_name: str, 
                 gemini_api_key: str, embedding_model_name: str = EMBEDDING_MODEL):
        self.pinecone_api_key = pinecone_api_key
        self.pinecone_index_name = pinecone_index_name
        self.gemini_api_key = gemini_api_key
        self.embedding_model_name = embedding_model_name
        
        # Initialize clients
        self._initialize_clients()
    
    def _initialize_clients(self):
        """Initialize all necessary clients"""
        try:
            print("Initializing MultimodalRAGClient...")
            
            # Initialize Pinecone
            print("Creating Pinecone client...")
            self.pc = Pinecone(api_key=self.pinecone_api_key)
            
            # Check if index exists, create if not
            if self.pinecone_index_name not in self.pc.list_indexes().names():
                print(f"Creating Pinecone index: {self.pinecone_index_name}")
                self.pc.create_index(
                    name=self.pinecone_index_name,
                    dimension=384,  # Dimension for all-MiniLM-L6-v2
                    metric='cosine',
                    spec=self.pc.ServerlessSpec(
                        cloud='aws',
                        region='us-east-1'
                    )
                )
            
            self.pinecone_index = self.pc.Index(self.pinecone_index_name)
            
            # Initialize Gemini
            print("Creating Gemini client...")
            genai.configure(api_key=self.gemini_api_key)
            self.gemini_model = genai.GenerativeModel('gemini-2.0-flash-exp')
            
            # Initialize embedding model
            print(f"Creating embedding model: {self.embedding_model_name}")
            self.embedding_model = HuggingFaceEmbeddings(
                model_name=self.embedding_model_name,
                model_kwargs={'device': 'cpu'},  # Change to 'cuda' if you have GPU
                encode_kwargs={'normalize_embeddings': True}
            )
            
            print("MultimodalRAGClient initialized successfully")
            
        except Exception as e:
            print(f"Error initializing clients: {e}")
            raise IngestionError(f"Failed to initialize clients: {e}")
    
    def get_user_namespace(self, user_id: str) -> str:
        """Generate consistent namespace for user"""
        # Create a hash of user_id to ensure consistent namespace naming
        # and avoid issues with special characters
        user_hash = hashlib.md5(str(user_id).encode()).hexdigest()[:8]
        return f"user_{user_hash}"
    
    def list_user_namespaces(self) -> List[str]:
        """List all user namespaces in the index"""
        try:
            stats = self.pinecone_index.describe_index_stats()
            namespaces = list(stats.get('namespaces', {}).keys())
            # Filter only user namespaces
            user_namespaces = [ns for ns in namespaces if ns.startswith('user_')]
            return user_namespaces
        except Exception as e:
            print(f"Error listing namespaces: {e}")
            return []
    
    def get_namespace_stats(self, namespace: str) -> Dict[str, Any]:
        """Get statistics for a specific namespace"""
        try:
            stats = self.pinecone_index.describe_index_stats()
            namespace_stats = stats.get('namespaces', {}).get(namespace, {})
            return {
                'vector_count': namespace_stats.get('vector_count', 0),
                'namespace': namespace
            }
        except Exception as e:
            print(f"Error getting namespace stats: {e}")
            return {'vector_count': 0, 'namespace': namespace}
    
    def clear_user_namespace(self, user_id: str) -> Dict[str, Any]:
        """Clear all vectors for a specific user"""
        try:
            namespace = self.get_user_namespace(user_id)
            self.pinecone_index.delete(delete_all=True, namespace=namespace)
            print(f"Cleared namespace {namespace} for user {user_id}")
            return {
                'success': True,
                'message': f'Cleared all documents for user {user_id}',
                'namespace': namespace
            }
        except Exception as e:
            print(f"Error clearing user namespace: {e}")
            return {
                'success': False,
                'error': str(e)
            }

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
        print(f"Loading {file_path} with LangChain loader...")
        
        try:
            if not os.path.exists(file_path):
                print(f"File not found: {file_path}")
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
            print(f"Loaded {len(docs)} pages/sections from {Path(file_path).name}")
            
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            failed_files.append(file_path)
            continue
    
    if failed_files:
        print(f"Failed to load {len(failed_files)} files: {failed_files}")
    
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
                        print(f"Error extracting image {img_index} from page {page_num + 1}: {e}")
            
            if images_data:
                page_images[page_num] = images_data
        
        doc.close()
        return page_images
        
    except Exception as e:
        print(f"Error extracting images from PDF {pdf_path}: {e}")
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
        print(f"Error performing OCR: {e}")
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
            
            print(f"Checking for images in {Path(source_file).name}, page {page_num + 1}...")
            
            # Extract images from this specific page
            try:
                page_images = extract_images_from_pdf(source_file)
                
                if page_num in page_images:
                    print(f"Found {len(page_images[page_num])} images on page {page_num + 1}")
                    
                    ocr_texts = []
                    for i, img_data in enumerate(page_images[page_num]):
                        print(f"Performing OCR on image {i + 1}...")
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
                        
                        print(f"Enhanced page with {len(ocr_texts)} OCR texts")
            
            except Exception as e:
                print(f"Error processing images for {source_file}: {e}")
                enhanced_doc.metadata['has_ocr'] = False
        else:
            enhanced_doc.metadata['has_ocr'] = False
        
        enhanced_docs.append(enhanced_doc)
    
    return enhanced_docs

def create_embeddings_and_chunks(documents: List[Document], embedding_model, user_id: str) -> List[Dict[str, Any]]:
    """Create text chunks and embeddings for vector store"""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    
    all_chunks = []
    
    print(f"Processing {len(documents)} documents for chunking and embedding...")
    
    for doc_idx, doc in enumerate(documents):
        if not doc.page_content.strip():
            continue
        
        print(f"Processing document {doc_idx + 1}/{len(documents)}: {doc.metadata.get('file_name', 'Unknown')}")
        
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
                        'document_index': doc_idx,
                        'user_id': user_id,  # Add user_id to metadata
                        'upload_timestamp': str(pd.Timestamp.now()) if 'pd' in globals() else str(uuid.uuid4())
                    }
                }
                
                all_chunks.append(chunk_data)
                
            except Exception as e:
                print(f"Error creating embedding for chunk {i} in document {doc_idx}: {e}")
                continue
        
        print(f"Created {len(chunks)} chunks from this document")
    
    return all_chunks

def store_in_pinecone_with_namespace(chunks: List[Dict[str, Any]], rag_client: MultimodalRAGClient, 
                                   user_id: str):
    """Store embeddings in Pinecone vector database with user namespace"""
    if not chunks:
        raise IngestionError("No chunks to store")
    
    namespace = rag_client.get_user_namespace(user_id)
    print(f"Storing {len(chunks)} chunks in Pinecone namespace: {namespace}")
    
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
                'document_index': chunk['metadata'].get('document_index', 0),
                'user_id': user_id
            }
        }
        vectors.append(vector)
    
    # Upsert in batches with namespace
    batch_size = 100
    failed_batches = 0
    
    for i in range(0, len(vectors), batch_size):
        batch = vectors[i:i + batch_size]
        try:
            rag_client.pinecone_index.upsert(vectors=batch, namespace=namespace)
            print(f"Upserted batch {i//batch_size + 1}/{(len(vectors) + batch_size - 1)//batch_size} to namespace {namespace}")
        except Exception as e:
            print(f"Error upserting batch {i//batch_size + 1}: {e}")
            failed_batches += 1
    
    if failed_batches > 0:
        print(f"Failed to upsert {failed_batches} batches")
    
    print(f"Successfully stored embeddings in Pinecone namespace: {namespace}!")

def validate_file_paths(file_paths: List[str]) -> List[str]:
    """Validate file paths and filter supported extensions"""
    valid_files = []
    
    for file_path in file_paths:
        file_path = str(file_path).strip()
        
        # Check if file exists
        if not os.path.exists(file_path):
            print(f"File not found: {file_path}")
            continue
        
        # Check if file extension is supported
        file_extension = Path(file_path).suffix.lower()
        if file_extension not in SUPPORTED_EXTENSIONS:
            print(f"Unsupported file type {file_extension}: {file_path}")
            continue
        
        valid_files.append(file_path)
    
    return valid_files

def ingest_documents_for_user(file_paths: List[str], 
                             user_id: str,
                             rag_client: MultimodalRAGClient,
                             enable_ocr: bool = True) -> Dict[str, Any]:
    """
    Main function to ingest multiple documents for a specific user
    """
    try:
        print("="*60)
        print("MULTIMODAL RAG USER DOCUMENT INGESTION STARTED")
        print(f"User ID: {user_id}")
        print(f"Processing {len(file_paths)} documents")
        print(f"Namespace: {rag_client.get_user_namespace(user_id)}")
        print("="*60)
        
        # Validate file paths
        valid_file_paths = validate_file_paths(file_paths)
        if not valid_file_paths:
            raise IngestionError("No valid file paths found")
        
        print(f"Valid files to process: {len(valid_file_paths)}")
        
        # Load documents using LangChain loaders
        print("Loading documents with LangChain loaders...")
        documents = load_documents_with_langchain(valid_file_paths)
        
        # Enhance with OCR if enabled
        if enable_ocr:
            print(f"Enhancing {len(documents)} documents with OCR...")
            documents = enhance_documents_with_ocr(documents, rag_client.gemini_model)
        else:
            print("OCR processing disabled, using text-only content...")
            for doc in documents:
                doc.metadata['has_ocr'] = False
        
        # Create embeddings and chunks
        print("Creating embeddings and chunks...")
        chunks = create_embeddings_and_chunks(documents, rag_client.embedding_model, user_id)
        
        if not chunks:
            raise IngestionError("No chunks were created")
        
        # Store in Pinecone with user namespace
        store_in_pinecone_with_namespace(chunks, rag_client, user_id)
        
        # Summary statistics
        pdf_docs = [d for d in documents if d.metadata.get('file_type', '').lower() == '.pdf']
        ocr_enhanced_docs = [d for d in documents if d.metadata.get('has_ocr', False)]
        
        # Get final namespace stats
        namespace_stats = rag_client.get_namespace_stats(rag_client.get_user_namespace(user_id))
        
        result = {
            'success': True,
            'user_id': user_id,
            'namespace': rag_client.get_user_namespace(user_id),
            'files_processed': len(valid_file_paths),
            'documents_loaded': len(documents),
            'pdf_documents': len(pdf_docs),
            'ocr_enhanced_documents': len(ocr_enhanced_docs),
            'chunks_created': len(chunks),
            'total_vectors_in_namespace': namespace_stats['vector_count'],
            'file_types': list(set([Path(fp).suffix.lower() for fp in valid_file_paths])),
            'failed_files': len(file_paths) - len(valid_file_paths)
        }
        
        print("="*60)
        print("USER DOCUMENT INGESTION COMPLETED SUCCESSFULLY!")
        print(f"User ID: {result['user_id']}")
        print(f"Namespace: {result['namespace']}")
        print(f"Files processed: {result['files_processed']}")
        print(f"Documents loaded: {result['documents_loaded']}")
        print(f"PDF documents: {result['pdf_documents']}")
        print(f"OCR enhanced documents: {result['ocr_enhanced_documents']}")
        print(f"Chunks created: {result['chunks_created']}")
        print(f"Total vectors in user namespace: {result['total_vectors_in_namespace']}")
        print("="*60)
        
        return result
        
    except Exception as e:
        print(f"Error during ingestion: {e}")
        return {
            'success': False,
            'error': str(e),
            'user_id': user_id,
            'files_processed': 0,
            'documents_loaded': 0,
            'chunks_created': 0
        }

def process_uploaded_files_for_user(upload_folder: str, 
                                   user_id: str,
                                   rag_client: MultimodalRAGClient,
                                   file_extensions: List[str] = None,
                                   enable_ocr: bool = True,
                                   cleanup_after_processing: bool = True) -> Dict[str, Any]:
    """
    Process all uploaded files in a user's upload folder
    This is the main function you'd call from your API
    """
    if file_extensions is None:
        file_extensions = list(SUPPORTED_EXTENSIONS)
    
    upload_path = Path(upload_folder)
    
    if not upload_path.exists():
        return {
            'success': False,
            'error': f"Upload folder {upload_folder} does not exist",
            'user_id': user_id,
            'files_processed': 0
        }
    
    files_to_process = []
    
    # Find all files with supported extensions
    for ext in file_extensions:
        # Use both case variations to catch files like .PDF, .Pdf, etc.
        files_to_process.extend(upload_path.glob(f"*{ext.lower()}"))
        files_to_process.extend(upload_path.glob(f"*{ext.upper()}"))
    
    # Remove duplicates and convert to strings
    file_paths = list(set([str(f) for f in files_to_process]))
    
    if not file_paths:
        return {
            'success': False,
            'error': f"No supported files found in {upload_folder}",
            'user_id': user_id,
            'files_processed': 0
        }
    
    print(f"Found {len(file_paths)} files to process for user {user_id}:")
    for fp in sorted(file_paths):
        print(f"  - {Path(fp).name}")
    
    # Process the documents
    result = ingest_documents_for_user(
        file_paths=file_paths, 
        user_id=user_id,
        rag_client=rag_client,
        enable_ocr=enable_ocr
    )
    
    # Cleanup uploaded files if requested and processing was successful
    if cleanup_after_processing and result.get('success', False):
        try:
            for file_path in file_paths:
                os.remove(file_path)
                print(f"Cleaned up: {Path(file_path).name}")
            print("Upload folder cleaned up successfully")
        except Exception as e:
            print(f"Warning: Could not cleanup some files: {e}")
    
    return result

def get_user_documents_info(user_id: str, rag_client: MultimodalRAGClient) -> Dict[str, Any]:
    """Get information about user's documents"""
    try:
        namespace = rag_client.get_user_namespace(user_id)
        stats = rag_client.get_namespace_stats(namespace)
        
        return {
            'success': True,
            'user_id': user_id,
            'namespace': namespace,
            'total_documents': stats['vector_count'],
            'index_exists': stats['vector_count'] > 0
        }
    except Exception as e:
        return {
            'success': False,
            'error': str(e),
            'user_id': user_id
        }