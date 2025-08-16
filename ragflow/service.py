#!/usr/bin/env python3
"""
RagFlow Service Implementation
Actual integration with RagFlow for RAG capabilities
"""

import os
import sys
import logging
from typing import Dict, Any, List, Optional
from pathlib import Path

from flask import Flask, request, jsonify
from flask_cors import CORS
from flask_restful import Api, Resource

import numpy as np
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS, Chroma
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI
from langchain.callbacks import get_openai_callback
import faiss
from pydantic import BaseModel, Field
from prometheus_client import Counter, Histogram, generate_latest
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Metrics
request_count = Counter('ragflow_requests_total', 'Total requests', ['method', 'endpoint'])
request_duration = Histogram('ragflow_request_duration_seconds', 'Request duration')
index_size = Counter('ragflow_index_size', 'Number of indexed documents')

class RagFlowService:
    """Main RagFlow service implementation."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.embeddings = None
        self.vector_store = None
        self.text_splitter = None
        self.llm = None
        self.retriever = None
        self.initialize()
    
    def initialize(self):
        """Initialize RAG components."""
        logger.info("Initializing RagFlow service...")
        
        # Initialize embeddings
        self.embeddings = HuggingFaceEmbeddings(
            model_name=self.config.get('embedding_model', 'sentence-transformers/all-MiniLM-L6-v2'),
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        
        # Initialize text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.config.get('chunk_size', 1000),
            chunk_overlap=self.config.get('chunk_overlap', 100),
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        
        # Initialize vector store
        vector_db_type = self.config.get('vector_db_type', 'faiss')
        if vector_db_type == 'faiss':
            self.initialize_faiss()
        elif vector_db_type == 'chroma':
            self.initialize_chroma()
        else:
            raise ValueError(f"Unsupported vector DB type: {vector_db_type}")
        
        # Initialize LLM (optional, can use local models)
        if os.getenv('OPENAI_API_KEY'):
            self.llm = OpenAI(
                temperature=self.config.get('temperature', 0.7),
                max_tokens=self.config.get('max_tokens', 2048)
            )
        else:
            logger.warning("No OpenAI API key found, using mock LLM")
            self.llm = MockLLM()
        
        logger.info("RagFlow service initialized successfully")
    
    def initialize_faiss(self):
        """Initialize FAISS vector store."""
        index_path = self.config.get('index_path', '/data/faiss_index')
        
        if os.path.exists(index_path):
            # Load existing index
            self.vector_store = FAISS.load_local(
                index_path, 
                self.embeddings,
                allow_dangerous_deserialization=True
            )
            logger.info(f"Loaded existing FAISS index from {index_path}")
        else:
            # Create new index
            dimension = 384  # for all-MiniLM-L6-v2
            index = faiss.IndexFlatL2(dimension)
            
            # Create empty FAISS store
            dummy_text = ["Initial document"]
            self.vector_store = FAISS.from_texts(
                dummy_text,
                self.embeddings
            )
            logger.info("Created new FAISS index")
    
    def initialize_chroma(self):
        """Initialize Chroma vector store."""
        persist_directory = self.config.get('persist_directory', '/data/chroma_db')
        
        self.vector_store = Chroma(
            persist_directory=persist_directory,
            embedding_function=self.embeddings
        )
        logger.info(f"Initialized Chroma DB at {persist_directory}")
    
    def index_document(self, content: str, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Index a document for RAG retrieval."""
        start_time = time.time()
        
        try:
            # Split text into chunks
            chunks = self.text_splitter.split_text(content)
            logger.info(f"Split document into {len(chunks)} chunks")
            
            # Add metadata to each chunk
            metadatas = [metadata for _ in chunks]
            
            # Add to vector store
            ids = self.vector_store.add_texts(
                texts=chunks,
                metadatas=metadatas
            )
            
            # Persist if using FAISS
            if isinstance(self.vector_store, FAISS):
                index_path = self.config.get('index_path', '/data/faiss_index')
                self.vector_store.save_local(index_path)
            
            index_size.inc(len(chunks))
            
            duration = time.time() - start_time
            
            return {
                'status': 'success',
                'chunks_indexed': len(chunks),
                'document_ids': ids,
                'duration_seconds': duration,
                'metadata': metadata
            }
            
        except Exception as e:
            logger.error(f"Error indexing document: {e}")
            return {
                'status': 'error',
                'error': str(e)
            }
    
    def query(self, query_text: str, k: int = 5, filters: Optional[Dict] = None) -> Dict[str, Any]:
        """Query the vector store and generate response."""
        start_time = time.time()
        
        try:
            # Similarity search
            if filters:
                docs = self.vector_store.similarity_search_with_score(
                    query_text,
                    k=k,
                    filter=filters
                )
            else:
                docs = self.vector_store.similarity_search_with_score(
                    query_text,
                    k=k
                )
            
            # Extract sources
            sources = []
            for doc, score in docs:
                sources.append({
                    'content': doc.page_content,
                    'metadata': doc.metadata,
                    'similarity_score': float(score)
                })
            
            # Generate response using LLM
            if self.llm:
                context = "\n\n".join([doc[0].page_content for doc in docs])
                prompt = f"Based on the following context, answer the question.\n\nContext:\n{context}\n\nQuestion: {query_text}\n\nAnswer:"
                
                if hasattr(self.llm, '__call__'):
                    response = self.llm(prompt)
                else:
                    response = "Mock response for: " + query_text
            else:
                response = "No LLM configured"
            
            duration = time.time() - start_time
            
            return {
                'status': 'success',
                'query': query_text,
                'response': response,
                'sources': sources,
                'num_sources': len(sources),
                'duration_seconds': duration
            }
            
        except Exception as e:
            logger.error(f"Error querying: {e}")
            return {
                'status': 'error',
                'query': query_text,
                'error': str(e)
            }
    
    def extract_insights(self, content: str) -> Dict[str, Any]:
        """Extract insights from document content."""
        start_time = time.time()
        
        try:
            insights = {}
            
            # Generate summary
            if self.llm:
                summary_prompt = f"Summarize the following text in 3-5 sentences:\n\n{content[:3000]}"
                insights['summary'] = self.llm(summary_prompt) if hasattr(self.llm, '__call__') else "Mock summary"
                
                # Extract key points
                key_points_prompt = f"List the 5 most important points from this text:\n\n{content[:3000]}"
                insights['key_points'] = self.llm(key_points_prompt) if hasattr(self.llm, '__call__') else ["Point 1", "Point 2"]
                
                # Generate questions
                questions_prompt = f"Generate 3 study questions based on this text:\n\n{content[:3000]}"
                insights['questions'] = self.llm(questions_prompt) if hasattr(self.llm, '__call__') else ["Question 1?", "Question 2?"]
            else:
                insights = {
                    'summary': 'Summary generation requires LLM',
                    'key_points': [],
                    'questions': []
                }
            
            duration = time.time() - start_time
            
            return {
                'status': 'success',
                'insights': insights,
                'content_length': len(content),
                'duration_seconds': duration
            }
            
        except Exception as e:
            logger.error(f"Error extracting insights: {e}")
            return {
                'status': 'error',
                'error': str(e)
            }
    
    def get_stats(self) -> Dict[str, Any]:
        """Get service statistics."""
        stats = {
            'service': 'ragflow',
            'status': 'operational',
            'config': {
                'embedding_model': self.config.get('embedding_model'),
                'vector_db_type': self.config.get('vector_db_type'),
                'chunk_size': self.config.get('chunk_size')
            }
        }
        
        if self.vector_store:
            if isinstance(self.vector_store, FAISS):
                # Get FAISS index stats
                stats['index_size'] = self.vector_store.index.ntotal
            elif hasattr(self.vector_store, '_collection'):
                # Get Chroma stats
                stats['index_size'] = self.vector_store._collection.count()
        
        return stats

class MockLLM:
    """Mock LLM for testing without API key."""
    def __call__(self, prompt: str) -> str:
        return f"Mock response for prompt: {prompt[:50]}..."

# Flask Application
app = Flask(__name__)
CORS(app)
api = Api(app)

# Load configuration
config = {
    'embedding_model': os.getenv('EMBEDDING_MODEL', 'sentence-transformers/all-MiniLM-L6-v2'),
    'vector_db_type': os.getenv('VECTOR_DB_TYPE', 'faiss'),
    'chunk_size': int(os.getenv('CHUNK_SIZE', '1000')),
    'chunk_overlap': int(os.getenv('CHUNK_OVERLAP', '100')),
    'temperature': float(os.getenv('TEMPERATURE', '0.7')),
    'max_tokens': int(os.getenv('MAX_TOKENS', '2048')),
    'index_path': os.getenv('INDEX_PATH', '/data/faiss_index'),
    'persist_directory': os.getenv('PERSIST_DIRECTORY', '/data/chroma_db')
}

# Initialize service
service = RagFlowService(config)

# API Resources
class HealthCheck(Resource):
    def get(self):
        return {'status': 'healthy', 'service': 'ragflow'}

class IndexDocument(Resource):
    @request_duration.time()
    def post(self):
        request_count.labels(method='POST', endpoint='/index').inc()
        
        data = request.get_json()
        content = data.get('content', '')
        metadata = data.get('metadata', {})
        
        result = service.index_document(content, metadata)
        return result

class Query(Resource):
    @request_duration.time()
    def post(self):
        request_count.labels(method='POST', endpoint='/query').inc()
        
        data = request.get_json()
        query = data.get('query', '')
        k = data.get('k', 5)
        filters = data.get('filters', None)
        
        result = service.query(query, k, filters)
        return result

class ExtractInsights(Resource):
    @request_duration.time()
    def post(self):
        request_count.labels(method='POST', endpoint='/insights').inc()
        
        data = request.get_json()
        content = data.get('content', '')
        
        result = service.extract_insights(content)
        return result

class Stats(Resource):
    def get(self):
        return service.get_stats()

class Metrics(Resource):
    def get(self):
        return generate_latest().decode('utf-8'), 200, {'Content-Type': 'text/plain'}

# Register API endpoints
api.add_resource(HealthCheck, '/health')
api.add_resource(IndexDocument, '/index')
api.add_resource(Query, '/query')
api.add_resource(ExtractInsights, '/insights')
api.add_resource(Stats, '/stats')
api.add_resource(Metrics, '/metrics')

# Error handlers
@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    port = int(os.getenv('SERVICE_PORT', '8010'))
    debug = os.getenv('DEBUG', 'false').lower() == 'true'
    
    logger.info(f"Starting RagFlow service on port {port}")
    app.run(host='0.0.0.0', port=port, debug=debug)