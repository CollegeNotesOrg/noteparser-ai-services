#!/usr/bin/env python3
"""
LangExtract NLP Service
Natural Language Processing service for entity extraction, relationship detection,
and semantic analysis of academic documents
"""

import os
import sys
import logging
import re
from typing import Dict, Any, List, Optional, Tuple, Set
from pathlib import Path
import asyncio
import json

from flask import Flask, request, jsonify
from flask_cors import CORS
from flask_restful import Api, Resource
import spacy
from spacy import displacy
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.chunk import ne_chunk
from nltk.tag import pos_tag
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import textstat
from textblob import TextBlob
from collections import Counter, defaultdict
import networkx as nx
from pydantic import BaseModel, Field
from prometheus_client import Counter, Histogram, generate_latest
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('taggers/averaged_perceptron_tagger')
except LookupError:
    nltk.download('averaged_perceptron_tagger')

try:
    nltk.data.find('chunkers/maxent_ne_chunker')
except LookupError:
    nltk.download('maxent_ne_chunker')

try:
    nltk.data.find('corpora/words')
except LookupError:
    nltk.download('words')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

# Metrics
request_count = Counter('langextract_requests_total', 'Total requests', ['method', 'endpoint'])
request_duration = Histogram('langextract_request_duration_seconds', 'Request duration')
text_length_processed = Counter('langextract_text_length_processed', 'Total text length processed')
entities_extracted = Counter('langextract_entities_extracted_total', 'Total entities extracted', ['type'])


class Entity(BaseModel):
    """Extracted entity structure."""
    text: str
    label: str
    start: int
    end: int
    confidence: float
    description: Optional[str] = None


class Relationship(BaseModel):
    """Relationship between entities."""
    source: str
    target: str
    relation_type: str
    confidence: float
    context: str


class KeyPhrase(BaseModel):
    """Key phrase or term."""
    phrase: str
    frequency: int
    importance: float
    context: List[str] = []


class ReadabilityMetrics(BaseModel):
    """Document readability analysis."""
    flesch_reading_ease: float
    flesch_kincaid_grade: float
    gunning_fog: float
    coleman_liau: float
    automated_readability: float
    avg_sentence_length: float
    avg_word_length: float


class SemanticAnalysis(BaseModel):
    """Semantic analysis results."""
    sentiment: Dict[str, float]
    key_concepts: List[str]
    topic_categories: List[str]
    semantic_density: float
    coherence_score: float


class ExtractionResult(BaseModel):
    """Complete extraction result."""
    status: str
    text_stats: Dict[str, Any]
    entities: List[Entity]
    relationships: List[Relationship]
    key_phrases: List[KeyPhrase]
    readability: ReadabilityMetrics
    semantic_analysis: SemanticAnalysis
    processing_time: float
    language: str


class LangExtractService:
    """Advanced NLP service for academic document analysis."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.nlp_model = None
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        
        # Academic domain patterns
        self.academic_patterns = self._load_academic_patterns()
        
        # Initialize NLP models
        self._initialize_models()
    
    def _initialize_models(self):
        """Initialize NLP models."""
        try:
            # Load spaCy model (try different models in order of preference)
            models_to_try = ['en_core_web_sm', 'en_core_web_md', 'en_core_web_lg']
            
            for model_name in models_to_try:
                try:
                    self.nlp_model = spacy.load(model_name)
                    logger.info(f"Loaded spaCy model: {model_name}")
                    break
                except OSError:
                    continue
            
            if self.nlp_model is None:
                logger.warning("No spaCy model found, using basic NER")
                
        except Exception as e:
            logger.error(f"Error initializing NLP models: {e}")
    
    def _load_academic_patterns(self) -> Dict[str, List[str]]:
        """Load academic domain-specific patterns."""
        return {
            'mathematical_terms': [
                r'\b(theorem|lemma|corollary|proof|axiom|definition|proposition)\b',
                r'\b(equation|formula|function|variable|constant|parameter)\b',
                r'\b(matrix|vector|scalar|tensor|derivative|integral)\b',
                r'\b(algorithm|complexity|optimization|convergence)\b'
            ],
            'scientific_terms': [
                r'\b(hypothesis|theory|experiment|methodology|analysis)\b',
                r'\b(sample|population|variable|correlation|significance)\b',
                r'\b(model|framework|approach|technique|procedure)\b',
                r'\b(result|finding|conclusion|implication|discussion)\b'
            ],
            'cs_terms': [
                r'\b(algorithm|data structure|complexity|runtime|memory)\b',
                r'\b(class|object|method|function|variable|parameter)\b',
                r'\b(database|query|index|transaction|normalization)\b',
                r'\b(network|protocol|security|encryption|authentication)\b'
            ],
            'citations': [
                r'\b[A-Z][a-zA-Z-]+ et al\.?,? \(\d{4}\)',
                r'\([A-Z][a-zA-Z-]+ \d{4}\)',
                r'\[[0-9]+\]',
                r'\bdoi:\s*[0-9.]+/[^\s]+',
                r'\barXiv:\s*[0-9]{4}\.[0-9]{4}'
            ]
        }
    
    def extract_all(self, text: str, language: str = 'en') -> ExtractionResult:
        """Perform comprehensive text analysis and extraction."""
        start_time = time.time()
        
        try:
            # Basic text statistics
            text_stats = self._calculate_text_stats(text)
            text_length_processed.inc(len(text))
            
            # Entity extraction
            entities = self._extract_entities(text)
            
            # Relationship extraction
            relationships = self._extract_relationships(text, entities)
            
            # Key phrase extraction
            key_phrases = self._extract_key_phrases(text)
            
            # Readability analysis
            readability = self._analyze_readability(text)
            
            # Semantic analysis
            semantic_analysis = self._analyze_semantics(text)
            
            processing_time = time.time() - start_time
            
            return ExtractionResult(
                status="success",
                text_stats=text_stats,
                entities=entities,
                relationships=relationships,
                key_phrases=key_phrases,
                readability=readability,
                semantic_analysis=semantic_analysis,
                processing_time=processing_time,
                language=language
            )
            
        except Exception as e:
            logger.error(f"Extraction failed: {e}")
            return ExtractionResult(
                status="error",
                text_stats={"error": str(e)},
                entities=[],
                relationships=[],
                key_phrases=[],
                readability=ReadabilityMetrics(
                    flesch_reading_ease=0,
                    flesch_kincaid_grade=0,
                    gunning_fog=0,
                    coleman_liau=0,
                    automated_readability=0,
                    avg_sentence_length=0,
                    avg_word_length=0
                ),
                semantic_analysis=SemanticAnalysis(
                    sentiment={"polarity": 0, "subjectivity": 0},
                    key_concepts=[],
                    topic_categories=[],
                    semantic_density=0,
                    coherence_score=0
                ),
                processing_time=time.time() - start_time,
                language=language
            )
    
    def _calculate_text_stats(self, text: str) -> Dict[str, Any]:
        """Calculate basic text statistics."""
        sentences = sent_tokenize(text)
        words = word_tokenize(text)
        
        # Filter words (remove punctuation and short words)
        clean_words = [word.lower() for word in words if word.isalpha() and len(word) > 2]
        
        return {
            "character_count": len(text),
            "word_count": len(words),
            "sentence_count": len(sentences),
            "paragraph_count": len([p for p in text.split('\n\n') if p.strip()]),
            "avg_words_per_sentence": len(words) / len(sentences) if sentences else 0,
            "avg_chars_per_word": sum(len(word) for word in clean_words) / len(clean_words) if clean_words else 0,
            "unique_words": len(set(clean_words)),
            "lexical_diversity": len(set(clean_words)) / len(clean_words) if clean_words else 0
        }
    
    def _extract_entities(self, text: str) -> List[Entity]:
        """Extract named entities and domain-specific terms."""
        entities = []
        
        # Use spaCy if available
        if self.nlp_model:
            entities.extend(self._extract_entities_spacy(text))
        
        # Use NLTK as fallback/supplement
        entities.extend(self._extract_entities_nltk(text))
        
        # Extract academic domain entities
        entities.extend(self._extract_academic_entities(text))
        
        # Remove duplicates and sort by confidence
        unique_entities = {}
        for entity in entities:
            key = (entity.text.lower(), entity.label)
            if key not in unique_entities or entity.confidence > unique_entities[key].confidence:
                unique_entities[key] = entity
        
        return sorted(unique_entities.values(), key=lambda x: x.confidence, reverse=True)
    
    def _extract_entities_spacy(self, text: str) -> List[Entity]:
        """Extract entities using spaCy."""
        entities = []
        
        try:
            doc = self.nlp_model(text)
            
            for ent in doc.ents:
                entities_extracted.labels(type=ent.label_).inc()
                entities.append(Entity(
                    text=ent.text,
                    label=ent.label_,
                    start=ent.start_char,
                    end=ent.end_char,
                    confidence=0.9,  # spaCy doesn't provide confidence scores
                    description=spacy.explain(ent.label_)
                ))
                
        except Exception as e:
            logger.error(f"spaCy entity extraction failed: {e}")
        
        return entities
    
    def _extract_entities_nltk(self, text: str) -> List[Entity]:
        """Extract entities using NLTK."""
        entities = []
        
        try:
            sentences = sent_tokenize(text)
            
            for sent in sentences:
                words = word_tokenize(sent)
                pos_tags = pos_tag(words)
                chunks = ne_chunk(pos_tags)
                
                current_chunk = []
                current_label = None
                
                for i, chunk in enumerate(chunks):
                    if hasattr(chunk, 'label'):
                        if current_chunk and current_label != chunk.label():
                            # Save previous chunk
                            entity_text = ' '.join([token for token, pos in current_chunk])
                            entities.append(Entity(
                                text=entity_text,
                                label=current_label,
                                start=0,  # NLTK doesn't provide character positions
                                end=0,
                                confidence=0.7
                            ))
                        
                        current_chunk = [(token, pos) for token, pos in chunk]
                        current_label = chunk.label()
                    else:
                        if current_chunk:
                            entity_text = ' '.join([token for token, pos in current_chunk])
                            entities.append(Entity(
                                text=entity_text,
                                label=current_label,
                                start=0,
                                end=0,
                                confidence=0.7
                            ))
                            current_chunk = []
                            current_label = None
                
                # Handle last chunk
                if current_chunk:
                    entity_text = ' '.join([token for token, pos in current_chunk])
                    entities.append(Entity(
                        text=entity_text,
                        label=current_label,
                        start=0,
                        end=0,
                        confidence=0.7
                    ))
                    
        except Exception as e:
            logger.error(f"NLTK entity extraction failed: {e}")
        
        return entities
    
    def _extract_academic_entities(self, text: str) -> List[Entity]:
        """Extract academic domain-specific entities."""
        entities = []
        
        for category, patterns in self.academic_patterns.items():
            for pattern in patterns:
                matches = re.finditer(pattern, text, re.IGNORECASE)
                for match in matches:
                    entities.append(Entity(
                        text=match.group(),
                        label=category.upper(),
                        start=match.start(),
                        end=match.end(),
                        confidence=0.8,
                        description=f"Academic {category.replace('_', ' ')}"
                    ))
        
        return entities
    
    def _extract_relationships(self, text: str, entities: List[Entity]) -> List[Relationship]:
        """Extract relationships between entities."""
        relationships = []
        
        # Simple co-occurrence based relationships
        entity_pairs = []
        for i, ent1 in enumerate(entities):
            for ent2 in entities[i+1:]:
                if abs(ent1.start - ent2.start) < 100:  # Within 100 characters
                    entity_pairs.append((ent1, ent2))
        
        # Pattern-based relationship extraction
        relationship_patterns = [
            (r'(.+?) is (?:a|an) (.+)', 'IS_A'),
            (r'(.+?) has (.+)', 'HAS'),
            (r'(.+?) uses (.+)', 'USES'),
            (r'(.+?) contains (.+)', 'CONTAINS'),
            (r'(.+?) depends on (.+)', 'DEPENDS_ON'),
            (r'(.+?) implements (.+)', 'IMPLEMENTS'),
            (r'(.+?) extends (.+)', 'EXTENDS')
        ]
        
        sentences = sent_tokenize(text)
        for sentence in sentences:
            for pattern, relation_type in relationship_patterns:
                matches = re.finditer(pattern, sentence, re.IGNORECASE)
                for match in matches:
                    source = match.group(1).strip()
                    target = match.group(2).strip()
                    
                    relationships.append(Relationship(
                        source=source,
                        target=target,
                        relation_type=relation_type,
                        confidence=0.6,
                        context=sentence
                    ))
        
        return relationships[:20]  # Limit to top 20 relationships
    
    def _extract_key_phrases(self, text: str) -> List[KeyPhrase]:
        """Extract key phrases and terms."""
        key_phrases = []
        
        try:
            # Tokenize and clean text
            words = word_tokenize(text.lower())
            words = [word for word in words if word.isalpha() and word not in self.stop_words]
            
            # Get word frequencies
            word_freq = Counter(words)
            
            # Extract n-grams
            bigrams = [f"{words[i]} {words[i+1]}" for i in range(len(words)-1)]
            trigrams = [f"{words[i]} {words[i+1]} {words[i+2]}" for i in range(len(words)-2)]
            
            bigram_freq = Counter(bigrams)
            trigram_freq = Counter(trigrams)
            
            # Combine and score phrases
            all_phrases = {}
            
            # Single words
            for word, freq in word_freq.most_common(50):
                if len(word) > 3:
                    importance = freq * len(word) / len(words)
                    all_phrases[word] = KeyPhrase(
                        phrase=word,
                        frequency=freq,
                        importance=importance,
                        context=[]
                    )
            
            # Bigrams
            for phrase, freq in bigram_freq.most_common(30):
                if freq > 1:
                    importance = freq * 2 / len(bigrams)
                    all_phrases[phrase] = KeyPhrase(
                        phrase=phrase,
                        frequency=freq,
                        importance=importance,
                        context=[]
                    )
            
            # Trigrams
            for phrase, freq in trigram_freq.most_common(20):
                if freq > 1:
                    importance = freq * 3 / len(trigrams)
                    all_phrases[phrase] = KeyPhrase(
                        phrase=phrase,
                        frequency=freq,
                        importance=importance,
                        context=[]
                    )
            
            # Sort by importance
            key_phrases = sorted(all_phrases.values(), key=lambda x: x.importance, reverse=True)[:30]
            
        except Exception as e:
            logger.error(f"Key phrase extraction failed: {e}")
        
        return key_phrases
    
    def _analyze_readability(self, text: str) -> ReadabilityMetrics:
        """Analyze text readability using multiple metrics."""
        try:
            return ReadabilityMetrics(
                flesch_reading_ease=textstat.flesch_reading_ease(text),
                flesch_kincaid_grade=textstat.flesch_kincaid_grade(text),
                gunning_fog=textstat.gunning_fog(text),
                coleman_liau=textstat.coleman_liau_index(text),
                automated_readability=textstat.automated_readability_index(text),
                avg_sentence_length=textstat.avg_sentence_length(text),
                avg_word_length=textstat.avg_letter_per_word(text)
            )
        except Exception as e:
            logger.error(f"Readability analysis failed: {e}")
            return ReadabilityMetrics(
                flesch_reading_ease=0,
                flesch_kincaid_grade=0,
                gunning_fog=0,
                coleman_liau=0,
                automated_readability=0,
                avg_sentence_length=0,
                avg_word_length=0
            )
    
    def _analyze_semantics(self, text: str) -> SemanticAnalysis:
        """Perform semantic analysis of the text."""
        try:
            blob = TextBlob(text)
            
            # Sentiment analysis
            sentiment = {
                "polarity": blob.sentiment.polarity,
                "subjectivity": blob.sentiment.subjectivity
            }
            
            # Extract key concepts (most frequent meaningful words)
            words = [word.lower() for word in word_tokenize(text) 
                    if word.isalpha() and word not in self.stop_words and len(word) > 3]
            
            key_concepts = [word for word, count in Counter(words).most_common(10)]
            
            # Simple topic categorization based on keywords
            topic_keywords = {
                'mathematics': ['theorem', 'proof', 'equation', 'function', 'calculus', 'algebra', 'geometry'],
                'computer_science': ['algorithm', 'programming', 'software', 'computer', 'data', 'system'],
                'physics': ['energy', 'force', 'quantum', 'particle', 'wave', 'field', 'physics'],
                'biology': ['cell', 'organism', 'species', 'evolution', 'protein', 'gene', 'biology'],
                'chemistry': ['molecule', 'atom', 'reaction', 'compound', 'element', 'chemistry'],
                'literature': ['author', 'novel', 'poem', 'character', 'theme', 'literary', 'literature'],
                'history': ['historical', 'century', 'war', 'civilization', 'culture', 'period', 'history']
            }
            
            topic_categories = []
            text_lower = text.lower()
            for topic, keywords in topic_keywords.items():
                if sum(1 for keyword in keywords if keyword in text_lower) >= 2:
                    topic_categories.append(topic)
            
            # Calculate semantic density (ratio of meaningful words to total words)
            total_words = len(word_tokenize(text))
            meaningful_words = len([word for word in word_tokenize(text) 
                                 if word.isalpha() and word not in self.stop_words])
            semantic_density = meaningful_words / total_words if total_words > 0 else 0
            
            # Simple coherence score based on sentence similarity
            sentences = sent_tokenize(text)
            coherence_score = 0.7 if len(sentences) > 1 else 1.0  # Placeholder implementation
            
            return SemanticAnalysis(
                sentiment=sentiment,
                key_concepts=key_concepts,
                topic_categories=topic_categories,
                semantic_density=semantic_density,
                coherence_score=coherence_score
            )
            
        except Exception as e:
            logger.error(f"Semantic analysis failed: {e}")
            return SemanticAnalysis(
                sentiment={"polarity": 0, "subjectivity": 0},
                key_concepts=[],
                topic_categories=[],
                semantic_density=0,
                coherence_score=0
            )
    
    def extract_entities_only(self, text: str) -> List[Entity]:
        """Extract only entities (lightweight operation)."""
        return self._extract_entities(text)
    
    def extract_key_phrases_only(self, text: str) -> List[KeyPhrase]:
        """Extract only key phrases (lightweight operation)."""
        return self._extract_key_phrases(text)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get service statistics."""
        return {
            "service": "langextract",
            "status": "operational",
            "models": {
                "spacy_model": self.nlp_model.meta['name'] if self.nlp_model else None,
                "nltk_data": "available"
            },
            "config": {
                "stop_words_count": len(self.stop_words),
                "academic_patterns": len(self.academic_patterns)
            }
        }


# Flask Application
app = Flask(__name__)
CORS(app)
api = Api(app)

# Load configuration
config = {
    'language': os.getenv('DEFAULT_LANGUAGE', 'en'),
    'max_text_length': int(os.getenv('MAX_TEXT_LENGTH', '1000000')),  # 1MB
    'enable_relationships': os.getenv('ENABLE_RELATIONSHIPS', 'true').lower() == 'true',
    'enable_semantics': os.getenv('ENABLE_SEMANTICS', 'true').lower() == 'true'
}

# Initialize service
service = LangExtractService(config)

# API Resources
class HealthCheck(Resource):
    def get(self):
        return {'status': 'healthy', 'service': 'langextract'}


class ExtractAll(Resource):
    @request_duration.time()
    def post(self):
        request_count.labels(method='POST', endpoint='/extract').inc()
        
        data = request.get_json()
        text = data.get('text', '')
        language = data.get('language', 'en')
        
        if not text:
            return {'error': 'No text provided'}, 400
        
        if len(text) > config['max_text_length']:
            return {'error': f'Text too long (max {config["max_text_length"]} characters)'}, 400
        
        try:
            result = service.extract_all(text, language)
            return result.model_dump()
            
        except Exception as e:
            logger.error(f"Extraction error: {e}")
            return {'error': f'Extraction failed: {str(e)}'}, 500


class ExtractEntities(Resource):
    @request_duration.time()
    def post(self):
        request_count.labels(method='POST', endpoint='/entities').inc()
        
        data = request.get_json()
        text = data.get('text', '')
        
        if not text:
            return {'error': 'No text provided'}, 400
        
        try:
            entities = service.extract_entities_only(text)
            return {
                'status': 'success',
                'entities': [entity.model_dump() for entity in entities],
                'count': len(entities)
            }
            
        except Exception as e:
            logger.error(f"Entity extraction error: {e}")
            return {'error': f'Entity extraction failed: {str(e)}'}, 500


class ExtractKeyPhrases(Resource):
    @request_duration.time()
    def post(self):
        request_count.labels(method='POST', endpoint='/keyphrases').inc()
        
        data = request.get_json()
        text = data.get('text', '')
        
        if not text:
            return {'error': 'No text provided'}, 400
        
        try:
            key_phrases = service.extract_key_phrases_only(text)
            return {
                'status': 'success',
                'key_phrases': [phrase.model_dump() for phrase in key_phrases],
                'count': len(key_phrases)
            }
            
        except Exception as e:
            logger.error(f"Key phrase extraction error: {e}")
            return {'error': f'Key phrase extraction failed: {str(e)}'}, 500


class Stats(Resource):
    def get(self):
        return service.get_stats()


class Metrics(Resource):
    def get(self):
        return generate_latest().decode('utf-8'), 200, {'Content-Type': 'text/plain'}


# Register API endpoints
api.add_resource(HealthCheck, '/health')
api.add_resource(ExtractAll, '/extract')
api.add_resource(ExtractEntities, '/entities')
api.add_resource(ExtractKeyPhrases, '/keyphrases')
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
    port = int(os.getenv('SERVICE_PORT', '8013'))
    debug = os.getenv('DEBUG', 'false').lower() == 'true'
    
    logger.info(f"Starting LangExtract NLP service on port {port}")
    app.run(host='0.0.0.0', port=port, debug=debug)