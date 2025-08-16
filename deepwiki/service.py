#!/usr/bin/env python3
"""
DeepWiki Service Implementation
AI-powered wiki system with knowledge organization
"""

import os
import sys
import logging
from typing import Dict, Any, List, Optional
from pathlib import Path
import json

from flask import Flask, request, jsonify
from flask_cors import CORS
from flask_restful import Api, Resource
from flask_socketio import SocketIO, emit

from wiki_engine import WikiEngine
from langchain.llms import OpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from prometheus_client import Counter, Histogram, generate_latest
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Metrics
request_count = Counter('deepwiki_requests_total', 'Total requests', ['method', 'endpoint'])
request_duration = Histogram('deepwiki_request_duration_seconds', 'Request duration')
article_count = Counter('deepwiki_articles_total', 'Total articles created')

class DeepWikiService:
    """Main DeepWiki service implementation."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.wiki_engine = WikiEngine(config)
        self.ai_assistant = None
        self.initialize_ai()
    
    def initialize_ai(self):
        """Initialize AI assistant capabilities."""
        logger.info("Initializing DeepWiki AI assistant...")
        
        # Initialize LLM if API key available
        if os.getenv('OPENAI_API_KEY'):
            self.llm = OpenAI(
                temperature=self.config.get('temperature', 0.7),
                max_tokens=self.config.get('max_tokens', 2048)
            )
            
            # Create conversation memory
            self.memory = ConversationBufferMemory(
                memory_key="chat_history",
                return_messages=True
            )
            
            logger.info("AI assistant initialized with OpenAI")
        else:
            logger.warning("No OpenAI API key found, AI features limited")
            self.llm = None
            self.memory = None
    
    def create_article(self, title: str, content: str, 
                      metadata: Optional[Dict] = None) -> Dict[str, Any]:
        """Create a new wiki article with AI enhancement."""
        start_time = time.time()
        
        try:
            # Enhance content with AI if available
            if self.llm and self.config.get('ai_enhancement', True):
                enhanced_content = self.enhance_content_with_ai(content)
            else:
                enhanced_content = content
            
            # Extract metadata using AI
            if self.llm:
                extracted_metadata = self.extract_metadata_with_ai(enhanced_content)
                if metadata:
                    metadata.update(extracted_metadata)
                else:
                    metadata = extracted_metadata
            
            # Create article using wiki engine
            result = self.wiki_engine.create_article(title, enhanced_content, metadata)
            
            if result['status'] == 'success':
                article_count.inc()
            
            duration = time.time() - start_time
            result['duration_seconds'] = duration
            
            return result
            
        except Exception as e:
            logger.error(f"Error creating article: {e}")
            return {
                'status': 'error',
                'error': str(e)
            }
    
    def update_article(self, article_id: str, updates: Dict[str, Any]) -> Dict[str, Any]:
        """Update an existing wiki article."""
        start_time = time.time()
        
        try:
            # Enhance updated content if provided
            if 'content' in updates and self.llm:
                updates['content'] = self.enhance_content_with_ai(updates['content'])
            
            result = self.wiki_engine.update_article(article_id, updates)
            
            duration = time.time() - start_time
            result['duration_seconds'] = duration
            
            return result
            
        except Exception as e:
            logger.error(f"Error updating article: {e}")
            return {
                'status': 'error',
                'error': str(e)
            }
    
    def search_wiki(self, query: str, limit: int = 10) -> Dict[str, Any]:
        """Search wiki with AI-enhanced results."""
        start_time = time.time()
        
        try:
            # Basic search
            results = self.wiki_engine.search_articles(query, limit)
            
            # Enhance with AI if available
            if self.llm and results:
                # Generate AI summary of results
                context = "\n\n".join([
                    f"{r['title']}: {r.get('snippet', '')}" 
                    for r in results[:3]
                ])
                
                summary_prompt = f"Based on these search results about '{query}':\n\n{context}\n\nProvide a brief summary:"
                ai_summary = self.llm(summary_prompt) if hasattr(self.llm, '__call__') else None
            else:
                ai_summary = None
            
            duration = time.time() - start_time
            
            return {
                'status': 'success',
                'query': query,
                'results': results,
                'result_count': len(results),
                'ai_summary': ai_summary,
                'duration_seconds': duration
            }
            
        except Exception as e:
            logger.error(f"Error searching wiki: {e}")
            return {
                'status': 'error',
                'query': query,
                'error': str(e)
            }
    
    def ask_assistant(self, question: str, context_articles: Optional[List[str]] = None) -> Dict[str, Any]:
        """Ask AI assistant a question about wiki content."""
        start_time = time.time()
        
        try:
            if not self.llm:
                return {
                    'status': 'error',
                    'error': 'AI assistant not available'
                }
            
            # Gather context from wiki
            context = ""
            if context_articles:
                for article_id in context_articles:
                    article = self.wiki_engine.get_article(article_id)
                    if article:
                        context += f"\n{article['title']}:\n{article['markdown_content'][:1000]}\n"
            else:
                # Search for relevant articles
                search_results = self.wiki_engine.search_articles(question, limit=3)
                for result in search_results:
                    article = self.wiki_engine.get_article(result['article_id'])
                    if article:
                        context += f"\n{article['title']}:\n{article['markdown_content'][:1000]}\n"
            
            # Generate answer
            if context:
                prompt = f"Based on the following wiki content:\n\n{context}\n\nAnswer this question: {question}"
            else:
                prompt = f"Answer this question based on general knowledge: {question}"
            
            answer = self.llm(prompt) if hasattr(self.llm, '__call__') else "AI response unavailable"
            
            # Find related articles
            related = self.wiki_engine.search_articles(question, limit=5)
            
            duration = time.time() - start_time
            
            return {
                'status': 'success',
                'question': question,
                'answer': answer,
                'related_articles': related,
                'context_used': bool(context),
                'duration_seconds': duration
            }
            
        except Exception as e:
            logger.error(f"Error with AI assistant: {e}")
            return {
                'status': 'error',
                'question': question,
                'error': str(e)
            }
    
    def get_link_graph(self, article_id: Optional[str] = None, depth: int = 2) -> Dict[str, Any]:
        """Get the wiki link graph for visualization."""
        start_time = time.time()
        
        try:
            graph_data = self.wiki_engine.get_link_graph(article_id, depth)
            
            duration = time.time() - start_time
            graph_data['duration_seconds'] = duration
            
            return graph_data
            
        except Exception as e:
            logger.error(f"Error getting link graph: {e}")
            return {
                'status': 'error',
                'error': str(e)
            }
    
    def find_similar(self, article_id: str, limit: int = 5) -> Dict[str, Any]:
        """Find similar articles."""
        start_time = time.time()
        
        try:
            similar = self.wiki_engine.find_similar_articles(article_id, limit)
            
            duration = time.time() - start_time
            
            return {
                'status': 'success',
                'article_id': article_id,
                'similar_articles': similar,
                'duration_seconds': duration
            }
            
        except Exception as e:
            logger.error(f"Error finding similar articles: {e}")
            return {
                'status': 'error',
                'error': str(e)
            }
    
    def enhance_content_with_ai(self, content: str) -> str:
        """Use AI to enhance wiki content."""
        if not self.llm:
            return content
        
        try:
            # Add section suggestions
            prompt = f"Enhance this wiki content by adding helpful sections or improving formatting (keep the same information, just improve presentation):\n\n{content[:2000]}"
            enhanced = self.llm(prompt) if hasattr(self.llm, '__call__') else content
            
            return enhanced if enhanced else content
            
        except Exception as e:
            logger.error(f"Error enhancing content: {e}")
            return content
    
    def extract_metadata_with_ai(self, content: str) -> Dict[str, Any]:
        """Extract metadata from content using AI."""
        if not self.llm:
            return {}
        
        try:
            prompt = f"Extract key metadata from this content. Return as JSON with fields: tags (list), categories (list), key_concepts (list):\n\n{content[:1500]}"
            
            response = self.llm(prompt) if hasattr(self.llm, '__call__') else "{}"
            
            # Try to parse JSON response
            try:
                metadata = json.loads(response)
            except:
                metadata = {
                    'tags': [],
                    'categories': [],
                    'key_concepts': []
                }
            
            return metadata
            
        except Exception as e:
            logger.error(f"Error extracting metadata: {e}")
            return {}
    
    def export_wiki(self, format: str = 'json') -> Any:
        """Export wiki data."""
        return self.wiki_engine.export_wiki(format)
    
    def import_wiki(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Import wiki data."""
        try:
            self.wiki_engine.import_wiki(data)
            return {'status': 'success', 'message': 'Wiki imported successfully'}
        except Exception as e:
            return {'status': 'error', 'error': str(e)}

# Flask Application
app = Flask(__name__)
CORS(app)
api = Api(app)
socketio = SocketIO(app, cors_allowed_origins="*")

# Load configuration
config = {
    'data_path': os.getenv('WIKI_DATA_PATH', '/data/wiki'),
    'database_url': os.getenv('DATABASE_URL', 'sqlite:///data/wiki/wiki.db'),
    'temperature': float(os.getenv('TEMPERATURE', '0.7')),
    'max_tokens': int(os.getenv('MAX_TOKENS', '2048')),
    'ai_enhancement': os.getenv('AI_ENHANCEMENT', 'true').lower() == 'true'
}

# Initialize service
service = DeepWikiService(config)

# API Resources
class HealthCheck(Resource):
    def get(self):
        return {'status': 'healthy', 'service': 'deepwiki'}

class CreateArticle(Resource):
    @request_duration.time()
    def post(self):
        request_count.labels(method='POST', endpoint='/article').inc()
        
        data = request.get_json()
        title = data.get('title', '')
        content = data.get('content', '')
        metadata = data.get('metadata', {})
        
        result = service.create_article(title, content, metadata)
        return result

class UpdateArticle(Resource):
    @request_duration.time()
    def put(self, article_id):
        request_count.labels(method='PUT', endpoint='/article').inc()
        
        data = request.get_json()
        result = service.update_article(article_id, data)
        return result

class GetArticle(Resource):
    def get(self, article_id):
        article = service.wiki_engine.get_article(article_id)
        if article:
            return article
        return {'error': 'Article not found'}, 404

class SearchWiki(Resource):
    @request_duration.time()
    def post(self):
        request_count.labels(method='POST', endpoint='/search').inc()
        
        data = request.get_json()
        query = data.get('query', '')
        limit = data.get('limit', 10)
        
        result = service.search_wiki(query, limit)
        return result

class AskAssistant(Resource):
    @request_duration.time()
    def post(self):
        request_count.labels(method='POST', endpoint='/ask').inc()
        
        data = request.get_json()
        question = data.get('question', '')
        context_articles = data.get('context_articles', None)
        
        result = service.ask_assistant(question, context_articles)
        return result

class LinkGraph(Resource):
    def get(self):
        article_id = request.args.get('article_id', None)
        depth = int(request.args.get('depth', 2))
        
        result = service.get_link_graph(article_id, depth)
        return result

class FindSimilar(Resource):
    def get(self, article_id):
        limit = int(request.args.get('limit', 5))
        result = service.find_similar(article_id, limit)
        return result

class ExportWiki(Resource):
    def get(self):
        format = request.args.get('format', 'json')
        data = service.export_wiki(format)
        return data

class ImportWiki(Resource):
    def post(self):
        data = request.get_json()
        result = service.import_wiki(data)
        return result

class Metrics(Resource):
    def get(self):
        return generate_latest().decode('utf-8'), 200, {'Content-Type': 'text/plain'}

# Register API endpoints
api.add_resource(HealthCheck, '/health')
api.add_resource(CreateArticle, '/article')
api.add_resource(UpdateArticle, '/article/<string:article_id>')
api.add_resource(GetArticle, '/article/<string:article_id>')
api.add_resource(SearchWiki, '/search')
api.add_resource(AskAssistant, '/ask')
api.add_resource(LinkGraph, '/graph')
api.add_resource(FindSimilar, '/similar/<string:article_id>')
api.add_resource(ExportWiki, '/export')
api.add_resource(ImportWiki, '/import')
api.add_resource(Metrics, '/metrics')

# WebSocket events for real-time collaboration
@socketio.on('connect')
def handle_connect():
    logger.info(f"Client connected: {request.sid}")
    emit('connected', {'message': 'Connected to DeepWiki'})

@socketio.on('disconnect')
def handle_disconnect():
    logger.info(f"Client disconnected: {request.sid}")

@socketio.on('article_update')
def handle_article_update(data):
    # Broadcast article updates to all connected clients
    emit('article_updated', data, broadcast=True, include_self=False)

# Error handlers
@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    port = int(os.getenv('SERVICE_PORT', '8011'))
    debug = os.getenv('DEBUG', 'false').lower() == 'true'
    
    logger.info(f"Starting DeepWiki service on port {port}")
    socketio.run(app, host='0.0.0.0', port=port, debug=debug, allow_unsafe_werkzeug=True)