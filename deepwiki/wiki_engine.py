"""
Wiki Engine for DeepWiki Service
Core wiki functionality implementation
"""

import os
import json
import hashlib
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
import re

import networkx as nx
from slugify import slugify
from fuzzywuzzy import fuzz
from whoosh import index
from whoosh.fields import Schema, TEXT, ID, DATETIME, KEYWORD
from whoosh.qparser import QueryParser, MultifieldParser
from sqlalchemy import create_engine, Column, String, Text, DateTime, Integer, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import markdown
from markdown.extensions import Extension
from markdown.preprocessors import Preprocessor
import frontmatter
import bleach

Base = declarative_base()

class WikiArticle(Base):
    """Database model for wiki articles."""
    __tablename__ = 'wiki_articles'
    
    id = Column(Integer, primary_key=True)
    article_id = Column(String(255), unique=True, index=True)
    title = Column(String(500))
    content = Column(Text)
    markdown_content = Column(Text)
    tags = Column(JSON)
    categories = Column(JSON)
    links_to = Column(JSON)
    linked_from = Column(JSON)
    article_metadata = Column(JSON)
    version = Column(Integer, default=1)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    author = Column(String(255))

class WikiLink(Base):
    """Database model for wiki links."""
    __tablename__ = 'wiki_links'
    
    id = Column(Integer, primary_key=True)
    source_id = Column(String(255), index=True)
    target_id = Column(String(255), index=True)
    link_type = Column(String(50))  # auto, manual, reference, etc.
    weight = Column(Integer, default=1)
    created_at = Column(DateTime, default=datetime.utcnow)

class WikiEngine:
    """Core wiki engine implementation."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.data_path = Path(config.get('data_path', '/data/wiki'))
        self.data_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize database
        self.init_database()
        
        # Initialize search index
        self.init_search_index()
        
        # Initialize knowledge graph
        self.knowledge_graph = nx.DiGraph()
        self.load_knowledge_graph()
        
        # Markdown processor
        self.md = markdown.Markdown(extensions=[
            'extra', 'codehilite', 'toc', 'tables', 'footnotes',
            'meta', 'admonition', 'nl2br', 'sane_lists'
        ])
    
    def init_database(self):
        """Initialize database connection."""
        db_url = self.config.get('database_url', 'sqlite:///data/wiki/wiki.db')
        self.engine = create_engine(db_url)
        Base.metadata.create_all(self.engine)
        Session = sessionmaker(bind=self.engine)
        self.session = Session()
    
    def init_search_index(self):
        """Initialize Whoosh search index."""
        index_dir = self.data_path / 'search_index'
        
        if not index_dir.exists():
            index_dir.mkdir(parents=True)
            
            schema = Schema(
                article_id=ID(stored=True, unique=True),
                title=TEXT(stored=True),
                content=TEXT(stored=True),
                tags=KEYWORD(stored=True, commas=True),
                categories=KEYWORD(stored=True, commas=True),
                created_at=DATETIME(stored=True)
            )
            
            self.search_index = index.create_in(str(index_dir), schema)
        else:
            self.search_index = index.open_dir(str(index_dir))
        
        self.search_writer = self.search_index.writer()
    
    def load_knowledge_graph(self):
        """Load knowledge graph from database."""
        articles = self.session.query(WikiArticle).all()
        for article in articles:
            self.knowledge_graph.add_node(
                article.article_id,
                title=article.title,
                tags=article.tags or [],
                categories=article.categories or []
            )
        
        links = self.session.query(WikiLink).all()
        for link in links:
            self.knowledge_graph.add_edge(
                link.source_id,
                link.target_id,
                type=link.link_type,
                weight=link.weight
            )
    
    def create_article(self, title: str, content: str, 
                      metadata: Optional[Dict] = None) -> Dict[str, Any]:
        """Create a new wiki article."""
        # Generate article ID
        article_id = slugify(title)
        
        # Check if article exists
        existing = self.session.query(WikiArticle).filter_by(
            article_id=article_id
        ).first()
        
        if existing:
            return {
                'status': 'error',
                'message': f'Article with ID {article_id} already exists'
            }
        
        # Process content
        processed_content, extracted_links = self.process_content(content)
        html_content = self.md.convert(processed_content)
        
        # Extract metadata
        if metadata is None:
            metadata = {}
        
        tags = metadata.get('tags', [])
        categories = metadata.get('categories', [])
        author = metadata.get('author', 'anonymous')
        
        # Create article
        article = WikiArticle(
            article_id=article_id,
            title=title,
            content=html_content,
            markdown_content=content,
            tags=tags,
            categories=categories,
            links_to=extracted_links,
            linked_from=[],
            article_metadata=metadata,
            author=author
        )
        
        self.session.add(article)
        self.session.commit()
        
        # Update search index
        self.search_writer.add_document(
            article_id=article_id,
            title=title,
            content=content,
            tags=','.join(tags),
            categories=','.join(categories),
            created_at=article.created_at
        )
        self.search_writer.commit()
        
        # Update knowledge graph
        self.knowledge_graph.add_node(
            article_id,
            title=title,
            tags=tags,
            categories=categories
        )
        
        # Create auto-links
        self.create_auto_links(article_id, content)
        
        return {
            'status': 'success',
            'article_id': article_id,
            'title': title,
            'version': 1,
            'links_created': len(extracted_links)
        }
    
    def update_article(self, article_id: str, updates: Dict[str, Any]) -> Dict[str, Any]:
        """Update an existing article."""
        article = self.session.query(WikiArticle).filter_by(
            article_id=article_id
        ).first()
        
        if not article:
            return {
                'status': 'error',
                'message': f'Article {article_id} not found'
            }
        
        # Update fields
        if 'title' in updates:
            article.title = updates['title']
        
        if 'content' in updates:
            content = updates['content']
            processed_content, extracted_links = self.process_content(content)
            article.markdown_content = content
            article.content = self.md.convert(processed_content)
            article.links_to = extracted_links
        
        if 'tags' in updates:
            article.tags = updates['tags']
        
        if 'categories' in updates:
            article.categories = updates['categories']
        
        article.version += 1
        article.updated_at = datetime.utcnow()
        
        self.session.commit()
        
        # Update search index
        self.update_search_index(article)
        
        return {
            'status': 'success',
            'article_id': article_id,
            'version': article.version
        }
    
    def get_article(self, article_id: str) -> Optional[Dict[str, Any]]:
        """Get an article by ID."""
        article = self.session.query(WikiArticle).filter_by(
            article_id=article_id
        ).first()
        
        if not article:
            return None
        
        return {
            'article_id': article.article_id,
            'title': article.title,
            'content': article.content,
            'markdown_content': article.markdown_content,
            'tags': article.tags or [],
            'categories': article.categories or [],
            'links_to': article.links_to or [],
            'linked_from': article.linked_from or [],
            'metadata': article.article_metadata or {},
            'version': article.version,
            'created_at': article.created_at.isoformat(),
            'updated_at': article.updated_at.isoformat(),
            'author': article.author
        }
    
    def search_articles(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Search articles using full-text search."""
        with self.search_index.searcher() as searcher:
            parser = MultifieldParser(
                ["title", "content", "tags", "categories"],
                self.search_index.schema
            )
            q = parser.parse(query)
            results = searcher.search(q, limit=limit)
            
            articles = []
            for hit in results:
                article = self.get_article(hit['article_id'])
                if article:
                    articles.append({
                        'article_id': article['article_id'],
                        'title': article['title'],
                        'score': hit.score,
                        'snippet': self.create_snippet(article['markdown_content'], query)
                    })
            
            return articles
    
    def find_similar_articles(self, article_id: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Find articles similar to the given article."""
        article = self.session.query(WikiArticle).filter_by(
            article_id=article_id
        ).first()
        
        if not article:
            return []
        
        # Use tags and categories for similarity
        similar = []
        all_articles = self.session.query(WikiArticle).filter(
            WikiArticle.article_id != article_id
        ).all()
        
        for other in all_articles:
            score = 0
            
            # Tag similarity
            if article.tags and other.tags:
                tag_overlap = set(article.tags) & set(other.tags)
                score += len(tag_overlap) * 2
            
            # Category similarity
            if article.categories and other.categories:
                cat_overlap = set(article.categories) & set(other.categories)
                score += len(cat_overlap) * 3
            
            # Title similarity
            title_sim = fuzz.ratio(article.title, other.title) / 100
            score += title_sim
            
            if score > 0:
                similar.append({
                    'article_id': other.article_id,
                    'title': other.title,
                    'similarity_score': score
                })
        
        # Sort by similarity score
        similar.sort(key=lambda x: x['similarity_score'], reverse=True)
        
        return similar[:limit]
    
    def create_auto_links(self, article_id: str, content: str):
        """Automatically create links based on content."""
        # Find mentions of other articles
        all_articles = self.session.query(WikiArticle).filter(
            WikiArticle.article_id != article_id
        ).all()
        
        for other in all_articles:
            # Check if title is mentioned
            if other.title.lower() in content.lower():
                self.create_link(article_id, other.article_id, 'auto')
            
            # Check for tag matches
            article = self.session.query(WikiArticle).filter_by(
                article_id=article_id
            ).first()
            
            if article and article.tags and other.tags:
                tag_overlap = set(article.tags) & set(other.tags)
                if tag_overlap:
                    self.create_link(article_id, other.article_id, 'tag_based')
    
    def create_link(self, source_id: str, target_id: str, 
                   link_type: str = 'manual') -> bool:
        """Create a link between two articles."""
        # Check if link exists
        existing = self.session.query(WikiLink).filter_by(
            source_id=source_id,
            target_id=target_id
        ).first()
        
        if existing:
            existing.weight += 1
        else:
            link = WikiLink(
                source_id=source_id,
                target_id=target_id,
                link_type=link_type
            )
            self.session.add(link)
            
            # Update knowledge graph
            self.knowledge_graph.add_edge(
                source_id,
                target_id,
                type=link_type,
                weight=1
            )
        
        self.session.commit()
        return True
    
    def get_link_graph(self, article_id: Optional[str] = None, 
                      depth: int = 2) -> Dict[str, Any]:
        """Get the link graph for visualization."""
        if article_id:
            # Get subgraph around specific article
            nodes = set([article_id])
            for _ in range(depth):
                new_nodes = set()
                for node in nodes:
                    new_nodes.update(self.knowledge_graph.predecessors(node))
                    new_nodes.update(self.knowledge_graph.successors(node))
                nodes.update(new_nodes)
            
            subgraph = self.knowledge_graph.subgraph(nodes)
        else:
            subgraph = self.knowledge_graph
        
        # Convert to serializable format
        nodes = []
        for node_id in subgraph.nodes():
            node_data = subgraph.nodes[node_id]
            nodes.append({
                'id': node_id,
                'label': node_data.get('title', node_id),
                'tags': node_data.get('tags', []),
                'categories': node_data.get('categories', [])
            })
        
        edges = []
        for source, target, data in subgraph.edges(data=True):
            edges.append({
                'source': source,
                'target': target,
                'type': data.get('type', 'unknown'),
                'weight': data.get('weight', 1)
            })
        
        return {
            'nodes': nodes,
            'edges': edges,
            'node_count': len(nodes),
            'edge_count': len(edges)
        }
    
    def process_content(self, content: str) -> Tuple[str, List[str]]:
        """Process content to extract links and enhance formatting."""
        # Extract wiki links [[Article Name]]
        wiki_link_pattern = r'\[\[([^\]]+)\]\]'
        links = re.findall(wiki_link_pattern, content)
        
        # Convert wiki links to markdown links
        def replace_wiki_link(match):
            article_name = match.group(1)
            article_id = slugify(article_name)
            return f'[{article_name}](/wiki/{article_id})'
        
        processed = re.sub(wiki_link_pattern, replace_wiki_link, content)
        
        # Extract regular markdown links
        md_link_pattern = r'\[([^\]]+)\]\(([^\)]+)\)'
        md_links = re.findall(md_link_pattern, content)
        links.extend([link[1] for link in md_links if link[1].startswith('/wiki/')])
        
        return processed, links
    
    def create_snippet(self, content: str, query: str, length: int = 200) -> str:
        """Create a snippet of content around the query terms."""
        content_lower = content.lower()
        query_lower = query.lower()
        
        # Find position of query in content
        pos = content_lower.find(query_lower)
        
        if pos == -1:
            # Query not found, return beginning of content
            return content[:length] + '...' if len(content) > length else content
        
        # Extract snippet around query
        start = max(0, pos - length // 2)
        end = min(len(content), pos + len(query) + length // 2)
        
        snippet = content[start:end]
        
        # Add ellipsis if needed
        if start > 0:
            snippet = '...' + snippet
        if end < len(content):
            snippet = snippet + '...'
        
        return snippet
    
    def update_search_index(self, article: WikiArticle):
        """Update search index for an article."""
        # Delete old document
        self.search_writer.delete_by_term('article_id', article.article_id)
        
        # Add updated document
        self.search_writer.add_document(
            article_id=article.article_id,
            title=article.title,
            content=article.markdown_content,
            tags=','.join(article.tags or []),
            categories=','.join(article.categories or []),
            created_at=article.created_at
        )
        self.search_writer.commit()
    
    def export_wiki(self, format: str = 'json') -> Any:
        """Export entire wiki in specified format."""
        articles = self.session.query(WikiArticle).all()
        
        if format == 'json':
            data = {
                'articles': [],
                'links': [],
                'metadata': {
                    'export_date': datetime.utcnow().isoformat(),
                    'article_count': len(articles),
                    'format_version': '1.0'
                }
            }
            
            for article in articles:
                data['articles'].append({
                    'article_id': article.article_id,
                    'title': article.title,
                    'content': article.markdown_content,
                    'tags': article.tags,
                    'categories': article.categories,
                    'metadata': article.article_metadata,
                    'created_at': article.created_at.isoformat(),
                    'updated_at': article.updated_at.isoformat()
                })
            
            links = self.session.query(WikiLink).all()
            for link in links:
                data['links'].append({
                    'source': link.source_id,
                    'target': link.target_id,
                    'type': link.link_type,
                    'weight': link.weight
                })
            
            return data
        
        else:
            raise ValueError(f"Unsupported export format: {format}")
    
    def import_wiki(self, data: Dict[str, Any]):
        """Import wiki from exported data."""
        # Clear existing data
        self.session.query(WikiArticle).delete()
        self.session.query(WikiLink).delete()
        self.session.commit()
        
        # Import articles
        for article_data in data.get('articles', []):
            article = WikiArticle(
                article_id=article_data['article_id'],
                title=article_data['title'],
                markdown_content=article_data['content'],
                content=self.md.convert(article_data['content']),
                tags=article_data.get('tags', []),
                categories=article_data.get('categories', []),
                metadata=article_data.get('metadata', {}),
                created_at=datetime.fromisoformat(article_data['created_at']),
                updated_at=datetime.fromisoformat(article_data['updated_at'])
            )
            self.session.add(article)
        
        # Import links
        for link_data in data.get('links', []):
            link = WikiLink(
                source_id=link_data['source'],
                target_id=link_data['target'],
                link_type=link_data.get('type', 'manual'),
                weight=link_data.get('weight', 1)
            )
            self.session.add(link)
        
        self.session.commit()
        
        # Rebuild search index and knowledge graph
        self.rebuild_search_index()
        self.load_knowledge_graph()