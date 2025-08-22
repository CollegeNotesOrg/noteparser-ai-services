#!/usr/bin/env python3
"""
Dolphin PDF Processing Service
Advanced PDF processing with layout preservation and intelligent content extraction
"""

import os
import sys
import logging
import tempfile
import io
from typing import Dict, Any, List, Optional, Union
from pathlib import Path
import asyncio

from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from flask_restful import Api, Resource
import fitz  # PyMuPDF
import pdfplumber
import pytesseract
from PIL import Image
import cv2
import numpy as np
from pydantic import BaseModel, Field
from prometheus_client import Counter, Histogram, generate_latest
import time
import json

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Metrics
request_count = Counter('dolphin_requests_total', 'Total requests', ['method', 'endpoint'])
request_duration = Histogram('dolphin_request_duration_seconds', 'Request duration')
pdf_pages_processed = Counter('dolphin_pdf_pages_processed_total', 'PDF pages processed')
extraction_errors = Counter('dolphin_extraction_errors_total', 'Extraction errors', ['type'])


class DocumentMetadata(BaseModel):
    """Document metadata structure."""
    title: Optional[str] = None
    author: Optional[str] = None
    subject: Optional[str] = None
    creator: Optional[str] = None
    producer: Optional[str] = None
    creation_date: Optional[str] = None
    modification_date: Optional[str] = None
    page_count: int = 0
    file_size: int = 0
    language: Optional[str] = None


class PageContent(BaseModel):
    """Page content structure."""
    page_number: int
    text: str
    images: List[Dict[str, Any]] = []
    tables: List[Dict[str, Any]] = []
    annotations: List[Dict[str, Any]] = []
    layout_boxes: List[Dict[str, Any]] = []
    confidence_score: float = 1.0


class ProcessingResult(BaseModel):
    """PDF processing result structure."""
    status: str
    document_metadata: DocumentMetadata
    pages: List[PageContent]
    processing_time: float
    extraction_method: str
    quality_metrics: Dict[str, Any]


class DolphinPDFService:
    """Advanced PDF processing service with layout preservation."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.temp_dir = Path(config.get('temp_dir', '/tmp/dolphin'))
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        
        # OCR configuration
        self.ocr_enabled = config.get('ocr_enabled', True)
        self.ocr_language = config.get('ocr_language', 'eng')
        
        # Processing options
        self.preserve_layout = config.get('preserve_layout', True)
        self.extract_images = config.get('extract_images', True)
        self.extract_tables = config.get('extract_tables', True)
        self.quality_threshold = config.get('quality_threshold', 0.7)
        
    def process_pdf(self, file_data: bytes, filename: str = None) -> ProcessingResult:
        """Process PDF with advanced extraction capabilities."""
        start_time = time.time()
        
        try:
            # Create temporary file
            with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as temp_file:
                temp_file.write(file_data)
                temp_path = Path(temp_file.name)
            
            # Extract metadata
            metadata = self._extract_metadata(temp_path)
            
            # Process pages
            pages = []
            extraction_method = "hybrid"  # Default to hybrid approach
            
            # Try PyMuPDF first for text extraction
            try:
                pages = self._process_with_pymupdf(temp_path)
                if self._assess_extraction_quality(pages) < self.quality_threshold:
                    logger.info("PyMuPDF quality low, trying pdfplumber")
                    pages = self._process_with_pdfplumber(temp_path)
                    extraction_method = "pdfplumber"
                    
                    if self._assess_extraction_quality(pages) < self.quality_threshold:
                        logger.info("pdfplumber quality low, using OCR")
                        pages = self._process_with_ocr(temp_path)
                        extraction_method = "ocr"
                else:
                    extraction_method = "pymupdf"
                    
            except Exception as e:
                logger.warning(f"Primary extraction failed: {e}, falling back to OCR")
                pages = self._process_with_ocr(temp_path)
                extraction_method = "ocr"
            
            # Calculate quality metrics
            quality_metrics = self._calculate_quality_metrics(pages)
            
            processing_time = time.time() - start_time
            pdf_pages_processed.inc(len(pages))
            
            # Clean up
            temp_path.unlink()
            
            return ProcessingResult(
                status="success",
                document_metadata=metadata,
                pages=pages,
                processing_time=processing_time,
                extraction_method=extraction_method,
                quality_metrics=quality_metrics
            )
            
        except Exception as e:
            extraction_errors.labels(type=type(e).__name__).inc()
            logger.error(f"PDF processing failed: {e}")
            
            return ProcessingResult(
                status="error",
                document_metadata=DocumentMetadata(),
                pages=[],
                processing_time=time.time() - start_time,
                extraction_method="failed",
                quality_metrics={"error": str(e)}
            )
    
    def _extract_metadata(self, pdf_path: Path) -> DocumentMetadata:
        """Extract document metadata."""
        try:
            doc = fitz.open(str(pdf_path))
            metadata = doc.metadata
            
            return DocumentMetadata(
                title=metadata.get('title'),
                author=metadata.get('author'),
                subject=metadata.get('subject'),
                creator=metadata.get('creator'),
                producer=metadata.get('producer'),
                creation_date=metadata.get('creationDate'),
                modification_date=metadata.get('modDate'),
                page_count=len(doc),
                file_size=pdf_path.stat().st_size,
                language=metadata.get('language', 'unknown')
            )
            
        except Exception as e:
            logger.error(f"Metadata extraction failed: {e}")
            return DocumentMetadata(
                page_count=0,
                file_size=pdf_path.stat().st_size if pdf_path.exists() else 0
            )
    
    def _process_with_pymupdf(self, pdf_path: Path) -> List[PageContent]:
        """Process PDF using PyMuPDF with layout preservation."""
        pages = []
        
        try:
            doc = fitz.open(str(pdf_path))
            
            for page_num in range(len(doc)):
                page = doc[page_num]
                
                # Extract text with layout information
                text_dict = page.get_text("dict")
                text_content = self._extract_structured_text(text_dict)
                
                # Extract images
                images = []
                if self.extract_images:
                    images = self._extract_images_pymupdf(page, page_num)
                
                # Extract layout boxes
                layout_boxes = self._extract_layout_boxes(text_dict)
                
                # Extract annotations
                annotations = self._extract_annotations(page)
                
                # Calculate confidence based on text extraction quality
                confidence = self._calculate_text_confidence(text_content)
                
                pages.append(PageContent(
                    page_number=page_num + 1,
                    text=text_content,
                    images=images,
                    tables=[],  # Tables will be extracted separately if needed
                    annotations=annotations,
                    layout_boxes=layout_boxes,
                    confidence_score=confidence
                ))
            
            doc.close()
            
        except Exception as e:
            logger.error(f"PyMuPDF processing failed: {e}")
            raise
        
        return pages
    
    def _process_with_pdfplumber(self, pdf_path: Path) -> List[PageContent]:
        """Process PDF using pdfplumber for better table extraction."""
        pages = []
        
        try:
            with pdfplumber.open(str(pdf_path)) as pdf:
                for page_num, page in enumerate(pdf.pages):
                    # Extract text
                    text_content = page.extract_text() or ""
                    
                    # Extract tables
                    tables = []
                    if self.extract_tables:
                        tables = self._extract_tables_pdfplumber(page)
                    
                    # Extract images (basic)
                    images = []
                    if self.extract_images:
                        images = self._extract_images_pdfplumber(page, page_num)
                    
                    confidence = self._calculate_text_confidence(text_content)
                    
                    pages.append(PageContent(
                        page_number=page_num + 1,
                        text=text_content,
                        images=images,
                        tables=tables,
                        annotations=[],
                        layout_boxes=[],
                        confidence_score=confidence
                    ))
                    
        except Exception as e:
            logger.error(f"pdfplumber processing failed: {e}")
            raise
        
        return pages
    
    def _process_with_ocr(self, pdf_path: Path) -> List[PageContent]:
        """Process PDF using OCR as fallback."""
        pages = []
        
        if not self.ocr_enabled:
            logger.warning("OCR disabled, returning empty results")
            return pages
        
        try:
            doc = fitz.open(str(pdf_path))
            
            for page_num in range(len(doc)):
                page = doc[page_num]
                
                # Convert page to image
                pix = page.get_pixmap()
                img_data = pix.tobytes("png")
                image = Image.open(io.BytesIO(img_data))
                
                # Preprocess image for better OCR
                cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
                processed_image = self._preprocess_for_ocr(cv_image)
                
                # Perform OCR
                try:
                    ocr_config = f'--oem 3 --psm 6 -l {self.ocr_language}'
                    text_content = pytesseract.image_to_string(
                        processed_image,
                        config=ocr_config
                    )
                    
                    # Get confidence data
                    data = pytesseract.image_to_data(
                        processed_image,
                        output_type=pytesseract.Output.DICT,
                        config=ocr_config
                    )
                    
                    confidence = self._calculate_ocr_confidence(data)
                    
                except Exception as e:
                    logger.error(f"OCR failed for page {page_num}: {e}")
                    text_content = ""
                    confidence = 0.0
                
                pages.append(PageContent(
                    page_number=page_num + 1,
                    text=text_content,
                    images=[],
                    tables=[],
                    annotations=[],
                    layout_boxes=[],
                    confidence_score=confidence
                ))
            
            doc.close()
            
        except Exception as e:
            logger.error(f"OCR processing failed: {e}")
            raise
        
        return pages
    
    def _extract_structured_text(self, text_dict: Dict) -> str:
        """Extract structured text from PyMuPDF text dictionary."""
        text_blocks = []
        
        for block in text_dict.get("blocks", []):
            if "lines" in block:  # Text block
                block_text = []
                for line in block["lines"]:
                    line_text = []
                    for span in line["spans"]:
                        line_text.append(span["text"])
                    block_text.append("".join(line_text))
                text_blocks.append("\n".join(block_text))
        
        return "\n\n".join(text_blocks)
    
    def _extract_images_pymupdf(self, page, page_num: int) -> List[Dict[str, Any]]:
        """Extract images from PDF page using PyMuPDF."""
        images = []
        
        try:
            image_list = page.get_images()
            
            for img_index, img in enumerate(image_list):
                xref = img[0]
                base_image = page.parent.extract_image(xref)
                
                images.append({
                    "index": img_index,
                    "page": page_num + 1,
                    "width": base_image["width"],
                    "height": base_image["height"],
                    "colorspace": base_image["colorspace"],
                    "ext": base_image["ext"],
                    "size": len(base_image["image"])
                })
                
        except Exception as e:
            logger.error(f"Image extraction failed: {e}")
        
        return images
    
    def _extract_images_pdfplumber(self, page, page_num: int) -> List[Dict[str, Any]]:
        """Extract images from PDF page using pdfplumber."""
        images = []
        
        try:
            for img_index, img in enumerate(page.images):
                images.append({
                    "index": img_index,
                    "page": page_num + 1,
                    "x0": img["x0"],
                    "y0": img["y0"],
                    "x1": img["x1"],
                    "y1": img["y1"],
                    "width": img["width"],
                    "height": img["height"]
                })
                
        except Exception as e:
            logger.error(f"Image extraction failed: {e}")
        
        return images
    
    def _extract_tables_pdfplumber(self, page) -> List[Dict[str, Any]]:
        """Extract tables from PDF page."""
        tables = []
        
        try:
            page_tables = page.extract_tables()
            
            for table_index, table in enumerate(page_tables):
                if table:
                    tables.append({
                        "index": table_index,
                        "rows": len(table),
                        "columns": len(table[0]) if table else 0,
                        "data": table
                    })
                    
        except Exception as e:
            logger.error(f"Table extraction failed: {e}")
        
        return tables
    
    def _extract_layout_boxes(self, text_dict: Dict) -> List[Dict[str, Any]]:
        """Extract layout bounding boxes."""
        layout_boxes = []
        
        try:
            for block in text_dict.get("blocks", []):
                if "lines" in block:
                    bbox = block["bbox"]
                    layout_boxes.append({
                        "type": "text_block",
                        "x0": bbox[0],
                        "y0": bbox[1],
                        "x1": bbox[2],
                        "y1": bbox[3]
                    })
                    
        except Exception as e:
            logger.error(f"Layout extraction failed: {e}")
        
        return layout_boxes
    
    def _extract_annotations(self, page) -> List[Dict[str, Any]]:
        """Extract annotations from PDF page."""
        annotations = []
        
        try:
            for annot in page.annots():
                annotations.append({
                    "type": annot.type[1],
                    "content": annot.info.get("content", ""),
                    "bbox": list(annot.rect),
                    "author": annot.info.get("title", "")
                })
                
        except Exception as e:
            logger.error(f"Annotation extraction failed: {e}")
        
        return annotations
    
    def _preprocess_for_ocr(self, image: np.ndarray) -> np.ndarray:
        """Preprocess image for better OCR results."""
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply denoising
        denoised = cv2.fastNlMeansDenoising(gray)
        
        # Apply threshold to get binary image
        _, binary = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Apply morphological operations to clean up
        kernel = np.ones((1, 1), np.uint8)
        cleaned = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        
        return cleaned
    
    def _calculate_text_confidence(self, text: str) -> float:
        """Calculate confidence score based on text quality."""
        if not text:
            return 0.0
        
        # Basic metrics for text quality
        word_count = len(text.split())
        char_count = len(text)
        
        if char_count == 0:
            return 0.0
        
        # Calculate ratio of alphanumeric characters
        alnum_ratio = sum(1 for c in text if c.isalnum()) / char_count
        
        # Calculate average word length
        avg_word_length = char_count / word_count if word_count > 0 else 0
        
        # Combine metrics (simple heuristic)
        confidence = min(1.0, (alnum_ratio * 0.7 + min(avg_word_length / 5, 1.0) * 0.3))
        
        return confidence
    
    def _calculate_ocr_confidence(self, ocr_data: Dict) -> float:
        """Calculate OCR confidence from pytesseract data."""
        confidences = [int(conf) for conf in ocr_data['conf'] if int(conf) > 0]
        
        if not confidences:
            return 0.0
        
        return sum(confidences) / len(confidences) / 100.0
    
    def _assess_extraction_quality(self, pages: List[PageContent]) -> float:
        """Assess overall extraction quality."""
        if not pages:
            return 0.0
        
        total_confidence = sum(page.confidence_score for page in pages)
        return total_confidence / len(pages)
    
    def _calculate_quality_metrics(self, pages: List[PageContent]) -> Dict[str, Any]:
        """Calculate comprehensive quality metrics."""
        if not pages:
            return {"overall_quality": 0.0}
        
        total_pages = len(pages)
        total_text_length = sum(len(page.text) for page in pages)
        avg_confidence = sum(page.confidence_score for page in pages) / total_pages
        
        pages_with_text = sum(1 for page in pages if page.text.strip())
        pages_with_images = sum(1 for page in pages if page.images)
        pages_with_tables = sum(1 for page in pages if page.tables)
        
        return {
            "overall_quality": avg_confidence,
            "total_pages": total_pages,
            "avg_text_length": total_text_length / total_pages if total_pages > 0 else 0,
            "pages_with_content": {
                "text": pages_with_text,
                "images": pages_with_images,
                "tables": pages_with_tables
            },
            "content_coverage": {
                "text": pages_with_text / total_pages if total_pages > 0 else 0,
                "images": pages_with_images / total_pages if total_pages > 0 else 0,
                "tables": pages_with_tables / total_pages if total_pages > 0 else 0
            }
        }
    
    def get_stats(self) -> Dict[str, Any]:
        """Get service statistics."""
        return {
            "service": "dolphin",
            "status": "operational",
            "config": {
                "ocr_enabled": self.ocr_enabled,
                "ocr_language": self.ocr_language,
                "preserve_layout": self.preserve_layout,
                "extract_images": self.extract_images,
                "extract_tables": self.extract_tables,
                "quality_threshold": self.quality_threshold
            },
            "temp_dir": str(self.temp_dir),
            "temp_dir_size": sum(f.stat().st_size for f in self.temp_dir.rglob('*') if f.is_file())
        }


# Flask Application
app = Flask(__name__)
CORS(app)
api = Api(app)

# Load configuration
config = {
    'temp_dir': os.getenv('TEMP_DIR', '/tmp/dolphin'),
    'ocr_enabled': os.getenv('OCR_ENABLED', 'true').lower() == 'true',
    'ocr_language': os.getenv('OCR_LANGUAGE', 'eng'),
    'preserve_layout': os.getenv('PRESERVE_LAYOUT', 'true').lower() == 'true',
    'extract_images': os.getenv('EXTRACT_IMAGES', 'true').lower() == 'true',
    'extract_tables': os.getenv('EXTRACT_TABLES', 'true').lower() == 'true',
    'quality_threshold': float(os.getenv('QUALITY_THRESHOLD', '0.7'))
}

# Initialize service
service = DolphinPDFService(config)

# API Resources
class HealthCheck(Resource):
    def get(self):
        return {'status': 'healthy', 'service': 'dolphin'}


class ProcessPDF(Resource):
    @request_duration.time()
    def post(self):
        request_count.labels(method='POST', endpoint='/process').inc()
        
        if 'file' not in request.files:
            return {'error': 'No file provided'}, 400
        
        file = request.files['file']
        if file.filename == '':
            return {'error': 'No file selected'}, 400
        
        if not file.filename.lower().endswith('.pdf'):
            return {'error': 'File must be a PDF'}, 400
        
        try:
            file_data = file.read()
            result = service.process_pdf(file_data, file.filename)
            
            # Convert Pydantic model to dict for JSON response
            return result.model_dump()
            
        except Exception as e:
            logger.error(f"PDF processing error: {e}")
            return {'error': f'Processing failed: {str(e)}'}, 500


class ExtractText(Resource):
    @request_duration.time()
    def post(self):
        request_count.labels(method='POST', endpoint='/extract-text').inc()
        
        if 'file' not in request.files:
            return {'error': 'No file provided'}, 400
        
        file = request.files['file']
        try:
            file_data = file.read()
            result = service.process_pdf(file_data, file.filename)
            
            # Extract only text content
            text_content = []
            for page in result.pages:
                text_content.append({
                    'page': page.page_number,
                    'text': page.text,
                    'confidence': page.confidence_score
                })
            
            return {
                'status': result.status,
                'extraction_method': result.extraction_method,
                'pages': text_content,
                'metadata': result.document_metadata.model_dump()
            }
            
        except Exception as e:
            logger.error(f"Text extraction error: {e}")
            return {'error': f'Extraction failed: {str(e)}'}, 500


class Stats(Resource):
    def get(self):
        return service.get_stats()


class Metrics(Resource):
    def get(self):
        return generate_latest().decode('utf-8'), 200, {'Content-Type': 'text/plain'}


# Register API endpoints
api.add_resource(HealthCheck, '/health')
api.add_resource(ProcessPDF, '/process')
api.add_resource(ExtractText, '/extract-text')
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
    port = int(os.getenv('SERVICE_PORT', '8012'))
    debug = os.getenv('DEBUG', 'false').lower() == 'true'
    
    logger.info(f"Starting Dolphin PDF service on port {port}")
    app.run(host='0.0.0.0', port=port, debug=debug)