import logging
from pathlib import Path
from typing import Dict, List, Any
import camelot
import pdfplumber
import numpy as np
import re
import pandas as pd
from sympy import sympify
import torch
import os
import requests
from urllib.parse import urlparse
import fitz 
logger = logging.getLogger(__name__)

try:
    from pdf2image import convert_from_path
    PDF2IMAGE_AVAILABLE = True
except ImportError:
    PDF2IMAGE_AVAILABLE = False
    logger.warning("pdf2image not available. Using PyMuPDF as fallback.")

try:
    from paddleocr import PaddleOCR
    PADDLE_OCR_AVAILABLE = True
except ImportError:
    PADDLE_OCR_AVAILABLE = False
    logger.warning("PaddleOCR not available. Using text extraction from PDF directly.")

try:
    import layoutparser as lp
    LAYOUTPARSER_AVAILABLE = True
except ImportError:
    LAYOUTPARSER_AVAILABLE = False
    logger.warning("LayoutParser not available. Using OCR-only mode.")

class OCRUtils:
    def __init__(self):
        self.ocr = None
        if PADDLE_OCR_AVAILABLE:
            try:
                self.ocr = PaddleOCR(use_angle_cls=True, lang='en', show_log=False)
                logger.info("PaddleOCR initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize PaddleOCR: {e}")
                self.ocr = None

    def extract_text(self, image) -> str:
        if self.ocr is None:
            logger.warning("OCR not available, returning empty text")
            return ""
        
        try:
            result = self.ocr.ocr(np.array(image), cls=True)
            if result and result[0]:
                text_lines = [line[1][0] for line in result[0] if len(line) >= 2 and line[1]]
                return '\n'.join(text_lines)
            return ""
        except Exception as e:
            logger.error(f"OCR extraction failed: {e}")
            return ""

class ParserEngine:
    def __init__(self):
        self.ocr_utils = OCRUtils()
        self.layout_model = None
        
        self._initialize_layout_model()

    def _initialize_layout_model(self):
        """Initialize LayoutParser model with proper error handling"""
        
        if not LAYOUTPARSER_AVAILABLE:
            logger.info("LayoutParser not available. Using OCR-only mode.")
            self.layout_model = None
            return
            
        if os.getenv('DISABLE_LAYOUTPARSER'):
            logger.info("LayoutParser disabled by environment variable")
            self.layout_model = None
            return
        
        try:
            logger.info("Attempting to load LayoutParser model...")
            
            try:
                self.layout_model = lp.models.Detectron2LayoutModel(
                    config_path='lp://PubLayNet/mask_rcnn_X_101_32x8d_FPN_3x/config',
                    model_path='lp://PubLayNet/mask_rcnn_X_101_32x8d_FPN_3x/model',
                    extra_config=["MODEL.ROI_HEADS.SCORE_THRESH_TEST", 0.5],
                    label_map={0: "Text", 1: "Title", 2: "List", 3: "Table", 4: "Figure"}
                )
                logger.info("LayoutParser model loaded successfully")
                return
            except AttributeError:
                # Try alternative API
                try:
                    self.layout_model = lp.Detectron2LayoutModel(
                        'lp://PubLayNet/mask_rcnn_X_101_32x8d_FPN_3x/config',
                        extra_config=["MODEL.ROI_HEADS.SCORE_THRESH_TEST", 0.5],
                        label_map={0: "Text", 1: "Title", 2: "List", 3: "Table", 4: "Figure"}
                    )
                    logger.info("LayoutParser model loaded successfully (alternative API)")
                    return
                except Exception as e2:
                    logger.warning(f"Alternative LayoutParser API failed: {e2}")
            
        except Exception as e:
            logger.error(f"LayoutParser initialization failed: {e}")
            logger.info("Continuing with OCR-only mode")
            
        self.layout_model = None

    def parse_pdf(self, pdf_path: str) -> Dict[str, Any]:
        result = {
            "file_path": pdf_path,
            "text_blocks": [],
            "tables": [],
            "linked_content": [],
            "formulas": [],
            "formula_table_links": [],
            "metadata": {}
        }

        if not os.path.exists(pdf_path):
            logger.error(f"PDF file not found: {pdf_path}")
            return result

        text_blocks, tables = self.extract_text_and_tables(pdf_path)
        result["text_blocks"] = text_blocks
        result["tables"] = tables
        result["linked_content"] = self.link_text_to_tables(text_blocks, tables)
        result["formulas"] = self.extract_formulas(text_blocks)
        result["formula_table_links"] = self.link_formulas_to_tables(result["formulas"], tables)

        result["metadata"] = {
            "num_text_blocks": len(text_blocks),
            "num_tables": len(tables),
            "num_linked_references": len(result["linked_content"]),
            "num_formulas": len(result["formulas"]),
            "parsing_methods": self._get_available_methods(),
            "layout_model_available": self.layout_model is not None,
            "pdf2image_available": PDF2IMAGE_AVAILABLE,
            "paddleocr_available": PADDLE_OCR_AVAILABLE
        }
        return result

    def _get_available_methods(self):
        methods = []
        if self.layout_model:
            methods.append("layout_parser")
        if PADDLE_OCR_AVAILABLE:
            methods.append("paddle_ocr")
        methods.extend(["camelot", "pdfplumber", "pymupdf"])
        return methods

    def extract_text_and_tables(self, pdf_path: str) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        text_blocks = []
        tables = []
        
        images = self._convert_pdf_to_images(pdf_path)
        
        if images:
            text_blocks = self._extract_text_from_images(images)
        else:
            text_blocks = self._extract_text_from_pdf_direct(pdf_path)

        camelot_tables = self.extract_tables_camelot(pdf_path)
        plumber_tables = self.extract_tables_plumber(pdf_path)
        tables = self.merge_tables(camelot_tables, plumber_tables)

        logger.info(f"Extracted {len(text_blocks)} text blocks and {len(tables)} tables")
        return text_blocks, tables

    def _convert_pdf_to_images(self, pdf_path: str):
        """Convert PDF to images with multiple fallback strategies"""
        images = []
        
        if PDF2IMAGE_AVAILABLE:
            try:
                for dpi in [200, 150, 100]:
                    try:
                        images = convert_from_path(pdf_path, dpi=dpi)
                        logger.info(f"PDF converted to {len(images)} images at {dpi} DPI")
                        break
                    except Exception as e:
                        logger.warning(f"PDF conversion failed at {dpi} DPI: {e}")
                        continue
                        
                if images:
                    return images
                    
            except Exception as e:
                logger.error(f"pdf2image failed completely: {e}")
        
        try:
            logger.info("Trying PyMuPDF for PDF processing...")
            doc = fitz.open(pdf_path)
            images = []
            
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                mat = fitz.Matrix(2.0, 2.0)  # 2x zoom
                pix = page.get_pixmap(matrix=mat)
                img_data = pix.tobytes("png")
                
                try:
                    from PIL import Image
                    import io
                    img = Image.open(io.BytesIO(img_data))
                    images.append(img)
                except ImportError:
                    logger.warning("PIL not available, cannot convert PyMuPDF output to images")
                    break
            
            doc.close()
            
            if images:
                logger.info(f"PyMuPDF converted PDF to {len(images)} images")
                return images
                
        except Exception as e:
            logger.error(f"PyMuPDF fallback failed: {e}")
        
        logger.warning("All PDF to image conversion methods failed")
        return []

    def _extract_text_from_images(self, images):
        """Extract text from images using LayoutParser or OCR"""
        text_blocks = []
        
        if self.layout_model and LAYOUTPARSER_AVAILABLE:
            logger.info("Using LayoutParser for text extraction")
            for idx, image in enumerate(images):
                try:
                    layout = self.layout_model.detect(np.array(image))
                    for block in layout:
                        if block.type in ["Text", "Title", "List"]:
                            text = self.ocr_utils.extract_text(block.crop_image(np.array(image)))
                            if text.strip():
                                text_blocks.append({
                                    "page": idx + 1,
                                    "type": block.type,
                                    "text": text.strip(),
                                    "bbox": [block.block.x_1, block.block.y_1, block.block.x_2, block.block.y_2],
                                    "confidence": float(block.score)
                                })
                except Exception as e:
                    logger.error(f"Error processing page {idx + 1} with LayoutParser: {e}")
                    self._extract_full_page_ocr(image, idx + 1, text_blocks)
        else:
            logger.info("Using OCR-only text extraction")
            for idx, image in enumerate(images):
                self._extract_full_page_ocr(image, idx + 1, text_blocks)
        
        return text_blocks

    def _extract_full_page_ocr(self, image, page_num, text_blocks):
        """Extract text from full page using OCR"""
        try:
            full_text = self.ocr_utils.extract_text(image)
            if full_text.strip():
                text_blocks.append({
                    "page": page_num,
                    "type": "FullPageText",
                    "text": full_text.strip(),
                    "bbox": [0, 0, getattr(image, 'width', 0), getattr(image, 'height', 0)],
                    "confidence": 0.5
                })
        except Exception as e:
            logger.error(f"OCR failed for page {page_num}: {e}")

    def _extract_text_from_pdf_direct(self, pdf_path: str):
        """Extract text directly from PDF without images"""
        text_blocks = []
        
        try:
            logger.info("Extracting text directly from PDF using pdfplumber")
            with pdfplumber.open(pdf_path) as pdf:
                for page_num, page in enumerate(pdf.pages):
                    text = page.extract_text()
                    if text and text.strip():
                        text_blocks.append({
                            "page": page_num + 1,
                            "type": "DirectText",
                            "text": text.strip(),
                            "bbox": [0, 0, page.width or 0, page.height or 0],
                            "confidence": 0.8
                        })
        except Exception as e:
            logger.error(f"pdfplumber text extraction failed: {e}")
        
        if not text_blocks:
            try:
                logger.info("Extracting text directly from PDF using PyMuPDF")
                doc = fitz.open(pdf_path)
                for page_num in range(len(doc)):
                    page = doc.load_page(page_num)
                    text = page.get_text()
                    if text and text.strip():
                        text_blocks.append({
                            "page": page_num + 1,
                            "type": "DirectText",
                            "text": text.strip(),
                            "bbox": [0, 0, page.rect.width, page.rect.height],
                            "confidence": 0.7
                        })
                doc.close()
            except Exception as e:
                logger.error(f"PyMuPDF text extraction failed: {e}")
        
        return text_blocks

    def extract_tables_camelot(self, pdf_path: str) -> List[Dict[str, Any]]:
        try:
            logger.info("Extracting tables with Camelot...")
            camelot_tables = camelot.read_pdf(pdf_path, flavor='lattice', pages='all')
            tables = []
            for i, table in enumerate(camelot_tables):
                df = table.df.dropna(how='all').dropna(axis=1, how='all').fillna('').astype(str).applymap(str.strip)
                if not df.empty:  # Only add non-empty tables
                    tables.append({
                        "table_id": f"camelot_{i}",
                        "page": table.page,
                        "bbox": list(table._bbox),
                        "data": df.to_dict("records"),
                        "headers": df.columns.tolist(),
                        "shape": df.shape,
                        "accuracy": float(table.accuracy),
                        "whitespace": float(table.whitespace),
                        "extraction_method": "camelot"
                    })
            logger.info(f"Camelot extracted {len(tables)} tables")
            return tables
        except Exception as e:
            logger.error(f"Camelot error: {e}")
            return []

    def extract_tables_plumber(self, pdf_path: str) -> List[Dict[str, Any]]:
        tables = []
        try:
            logger.info("Extracting tables with pdfplumber...")
            with pdfplumber.open(pdf_path) as pdf:
                for page_num, page in enumerate(pdf.pages):
                    page_tables = page.extract_tables()
                    for i, table_data in enumerate(page_tables):
                        if table_data and len(table_data) > 1:
                            # Clean and validate table data
                            cleaned_data = [[cell if cell is not None else '' for cell in row] for row in table_data]
                            if all(len(row) == len(cleaned_data[0]) for row in cleaned_data):  # Ensure consistent column count
                                df = pd.DataFrame(cleaned_data[1:], columns=cleaned_data[0]).fillna('').astype(str)
                                if not df.empty:  # Only add non-empty tables
                                    tables.append({
                                        "table_id": f"plumber_{page_num}_{i}",
                                        "page": page_num + 1,
                                        "bbox": [0, 0, page.width, page.height],
                                        "data": df.to_dict("records"),
                                        "headers": df.columns.tolist(),
                                        "shape": df.shape,
                                        "extraction_method": "pdfplumber"
                                    })
            logger.info(f"pdfplumber extracted {len(tables)} tables")
        except Exception as e:
            logger.error(f"Plumber error: {e}")
        return tables

    def merge_tables(self, camelot_tables, plumber_tables):
        """Merge tables from different extraction methods, avoiding duplicates"""
        merged = camelot_tables[:]
        for pt in plumber_tables:
            is_duplicate = False
            for ct in camelot_tables:
                if (abs(pt["page"] - ct["page"]) == 0 and 
                    abs(pt["bbox"][0] - ct["bbox"][0]) < 50 and
                    pt["shape"] == ct["shape"]):
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                merged.append(pt)
        
        logger.info(f"Merged to {len(merged)} unique tables")
        return merged

    def link_text_to_tables(self, text_blocks: List[Dict], tables: List[Dict]) -> List[Dict[str, Any]]:
        links = []
        table_refs = [
            r'(?i)table\s+(\d+)', r'(?i)tab\.\s*(\d+)', r'(?i)see\s+table\s+(\d+)',
            r'(?i)in\s+table\s+(\d+)', r'(?i)table\s+(\d+)\s+shows', r'(?i)as\s+shown\s+in\s+table\s+(\d+)'
        ]
        
        tables_by_page = {}
        for t in tables:
            tables_by_page.setdefault(t["page"], []).append(t)

        for block in text_blocks:
            text, page = block["text"], block["page"]
            for pattern in table_refs:
                for match in re.finditer(pattern, text):
                    try:
                        table_num = int(match.group(1))
                        ref_text = match.group(0)
                        ref_tables = tables_by_page.get(page, [])
                        
                        target_table = None
                        if len(ref_tables) >= table_num:
                            target_table = ref_tables[table_num - 1]
                        elif ref_tables:  
                            target_table = ref_tables[0]
                        
                        if target_table:
                            links.append({
                                "reference_text": ref_text,
                                "reference_page": page,
                                "reference_bbox": block.get("bbox"),
                                "table_id": target_table["table_id"],
                                "table_page": target_table["page"],
                                "table_bbox": target_table["bbox"],
                                "link_type": "table_reference",
                                "confidence": 1.0 if len(ref_tables) >= table_num else 0.7
                            })
                    except (ValueError, IndexError) as e:
                        logger.debug(f"Error processing table reference: {e}")
                        continue
        
        logger.info(f"Found {len(links)} text-to-table links")
        return links

    def extract_formulas(self, text_blocks: List[Dict]) -> List[Dict[str, Any]]:
        formulas = []
        math_patterns = [
            r'[a-zA-Z]\s*=\s*[^=]+',  # Basic equations
            r'\b\w+\s*\(\s*\w+\s*\)',  # Functions
            r'∑|∫|∏|√|∞|±|≤|≥|≠|≈',  # Math symbols
            r'\b\d+\s*[+\-*/]\s*\d+',  # Simple arithmetic
        ]
        
        for block in text_blocks:
            text = block["text"]
            
            has_math = any(re.search(pattern, text) for pattern in math_patterns)
            
            if has_math:
                try:
                    expr = sympify(text)
                    formulas.append({
                        "formula_text": text,
                        "parsed": str(expr),
                        "variables": [str(s) for s in expr.free_symbols],
                        "page": block["page"],
                        "bbox": block.get("bbox"),
                        "confidence": 0.9
                    })
                except Exception:
                    formulas.append({
                        "formula_text": text,
                        "parsed": None,
                        "variables": [],
                        "page": block["page"],
                        "bbox": block.get("bbox"),
                        "confidence": 0.5
                    })
        
        logger.info(f"Extracted {len(formulas)} potential formulas")
        return formulas

    def link_formulas_to_tables(self, formulas: List[Dict], tables: List[Dict]) -> List[Dict]:
        links = []
        for formula in formulas:
            variables = formula.get("variables", [])
            if not variables:
                continue
                
            for table in tables:
                headers = [h.lower().strip() for h in table.get("headers", []) if h]
                if not headers:
                    continue
                    
                matched_vars = []
                for var in variables:
                    var_lower = var.lower().strip()
                    if var_lower in headers or any(var_lower in header for header in headers):
                        matched_vars.append(var)
                
                if matched_vars:
                    links.append({
                        "formula": formula["formula_text"],
                        "formula_page": formula["page"],
                        "matched_variables": matched_vars,
                        "table_id": table["table_id"],
                        "table_page": table["page"],
                        "confidence": len(matched_vars) / len(variables) if variables else 0
                    })
        
        logger.info(f"Found {len(links)} formula-to-table links")
        return links
