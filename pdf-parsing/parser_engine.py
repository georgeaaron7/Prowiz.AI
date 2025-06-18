import logging
from pathlib import Path
from typing import Dict, List, Any
import camelot
import pdfplumber
import layoutparser as lp
import numpy as np
from pdf2image import convert_from_path
from paddleocr import PaddleOCR
import re
import pandas as pd
from sympy import sympify

logger = logging.getLogger(__name__)

class OCRUtils:
    def __init__(self):
        self.ocr = PaddleOCR(use_angle_cls=True, lang='en', show_log=False)

    def extract_text(self, image) -> str:
        result = self.ocr.ocr(np.array(image), cls=True)
        text_lines = [line[1][0] for line in result[0] if len(line) >= 2 and line[1]]
        return '\n'.join(text_lines)

class ParserEngine:
    def __init__(self):
        self.ocr_utils = OCRUtils()
        try:
            self.layout_model = lp.Detectron2LayoutModel(
                'lp://PubLayNet/mask_rcnn_X_101_32x8d_FPN_3x/config',
                extra_config=["MODEL.ROI_HEADS.SCORE_THRESH_TEST", 0.5],
                label_map={0: "Text", 1: "Title", 2: "List", 3: "Table", 4: "Figure"}
            )
        except Exception as e:
            logger.error(f"LayoutParser model load failed: {e}")
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
            "parsing_methods": ["layout_parser", "camelot", "plumber"]
        }
        return result

    def extract_text_and_tables(self, pdf_path: str) -> (List[Dict[str, Any]], List[Dict[str, Any]]):
        text_blocks = []
        tables = []
        images = convert_from_path(pdf_path, dpi=200)

        if self.layout_model:
            for idx, image in enumerate(images):
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

        if not text_blocks:
            for idx, image in enumerate(images):
                full_text = self.ocr_utils.extract_text(image)
                if full_text.strip():
                    text_blocks.append({
                        "page": idx + 1,
                        "type": "FullPageText",
                        "text": full_text.strip(),
                        "bbox": [0, 0, image.width, image.height],
                        "confidence": 0.5
                    })

        camelot_tables = self.extract_tables_camelot(pdf_path)
        plumber_tables = self.extract_tables_plumber(pdf_path)
        tables = self.merge_tables(camelot_tables, plumber_tables)

        return text_blocks, tables

    def extract_tables_camelot(self, pdf_path: str) -> List[Dict[str, Any]]:
        try:
            camelot_tables = camelot.read_pdf(pdf_path, flavor='lattice', pages='all')
            tables = []
            for i, table in enumerate(camelot_tables):
                df = table.df.dropna(how='all').dropna(axis=1, how='all').fillna('').astype(str).applymap(str.strip)
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
            return tables
        except Exception as e:
            logger.error(f"Camelot error: {e}")
            return []

    def extract_tables_plumber(self, pdf_path: str) -> List[Dict[str, Any]]:
        tables = []
        try:
            with pdfplumber.open(pdf_path) as pdf:
                for page_num, page in enumerate(pdf.pages):
                    page_tables = page.extract_tables()
                    for i, table_data in enumerate(page_tables):
                        if table_data and len(table_data) > 1:
                            df = pd.DataFrame(table_data[1:], columns=table_data[0]).fillna('').astype(str)
                            tables.append({
                                "table_id": f"plumber_{page_num}_{i}",
                                "page": page_num + 1,
                                "bbox": [0, 0, page.width, page.height],
                                "data": df.to_dict("records"),
                                "headers": df.columns.tolist(),
                                "shape": df.shape,
                                "extraction_method": "pdfplumber"
                            })
        except Exception as e:
            logger.error(f"Plumber error: {e}")
        return tables

    def merge_tables(self, camelot_tables, plumber_tables):
        merged = camelot_tables[:]
        for pt in plumber_tables:
            if not any(abs(pt["page"] - ct["page"]) == 0 and abs(pt["bbox"][0] - ct["bbox"][0]) < 50 for ct in camelot_tables):
                merged.append(pt)
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
                    table_num = int(match.group(1))
                    ref_text = match.group(0)
                    ref_table = tables_by_page.get(page, [])
                    if len(ref_table) >= table_num:
                        table = ref_table[table_num - 1]
                        links.append({
                            "reference_text": ref_text,
                            "reference_page": page,
                            "reference_bbox": block.get("bbox"),
                            "table_id": table["table_id"],
                            "table_page": table["page"],
                            "table_bbox": table["bbox"],
                            "link_type": "table_reference",
                            "confidence": 1.0
                        })
        return links

    def extract_formulas(self, text_blocks: List[Dict]) -> List[Dict[str, Any]]:
        formulas = []
        for block in text_blocks:
            text = block["text"]
            try:
                expr = sympify(text)
                formulas.append({
                    "formula_text": text,
                    "parsed": str(expr),
                    "variables": [str(s) for s in expr.free_symbols],
                    "page": block["page"],
                    "bbox": block.get("bbox")
                })
            except Exception:
                continue
        return formulas

    def link_formulas_to_tables(self, formulas: List[Dict], tables: List[Dict]) -> List[Dict]:
        links = []
        for formula in formulas:
            for table in tables:
                headers = [h.lower() for h in table.get("headers", [])]
                matched_vars = [var for var in formula["variables"] if var.lower() in headers]
                if matched_vars:
                    links.append({
                        "formula": formula["formula_text"],
                        "formula_page": formula["page"],
                        "matched_variables": matched_vars,
                        "table_id": table["table_id"],
                        "table_page": table["page"]
                    })
        return links
