# table_extractor.py
import cv2
import numpy as np
import pandas as pd
import pytesseract
from PIL import Image
import sqlite3
import json
import os
from typing import List, Dict, Tuple

class TableExtractor:
    def __init__(self, db_path="rag_database.db"):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Initialize SQLite database with tables schema"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create tables table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS tables (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            page_number INTEGER,
            source_file TEXT,
            table_data TEXT,
            description TEXT,
            columns TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        ''')
        
        conn.commit()
        conn.close()
    
    def extract_table_from_image(self, image_path: str) -> List[List[str]]:
        """
        Extract table data from image using computer vision and OCR
        """
        try:
            # Read and preprocess image
            image = cv2.imread(image_path)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Apply threshold
            _, binary = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY_INV)
            
            # Detect horizontal and vertical lines
            horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
            vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 40))
            
            horizontal_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, horizontal_kernel)
            vertical_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, vertical_kernel)
            
            # Find intersections to locate cells
            table_mask = cv2.addWeighted(horizontal_lines, 0.5, vertical_lines, 0.5, 0.0)
            
            # Find contours of table cells
            contours, _ = cv2.findContours(table_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Filter and sort contours by position
            cell_contours = []
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > 500:  # Filter small contours
                    x, y, w, h = cv2.boundingRect(contour)
                    cell_contours.append((x, y, w, h))
            
            # Sort cells by position (top to bottom, left to right)
            cell_contours.sort(key=lambda x: (x[1], x[0]))
            
            # Extract text from each cell
            table_data = []
            current_row = []
            current_y = None
            
            for x, y, w, h in cell_contours:
                if current_y is None:
                    current_y = y
                
                # If we've moved to a new row (y coordinate changed significantly)
                if abs(y - current_y) > 20 and current_row:
                    table_data.append(current_row)
                    current_row = []
                    current_y = y
                
                # Extract text from cell
                cell_image = gray[y:y+h, x:x+w]
                cell_text = pytesseract.image_to_string(cell_image, config='--psm 8').strip()
                current_row.append(cell_text)
            
            # Add the last row
            if current_row:
                table_data.append(current_row)
            
            return table_data
            
        except Exception as e:
            print(f"Error extracting table from {image_path}: {str(e)}")
            return []
    
    def generate_table_description(self, table_data: List[List[str]]) -> str:
        """
        Generate a brief description of the table content
        """
        if not table_data:
            return "Empty table"
        
        rows = len(table_data)
        cols = len(table_data[0]) if table_data else 0
        
        # Try to identify headers (first row)
        headers = table_data[0] if table_data else []
        
        # Sample some data to understand content
        description_parts = [f"Table with {rows} rows and {cols} columns"]
        
        if headers:
            header_text = ", ".join([h for h in headers if h.strip()])
            if header_text:
                description_parts.append(f"Headers: {header_text}")
        
        # Analyze data types in first few rows
        if len(table_data) > 1:
            sample_data = []
            for i in range(1, min(4, len(table_data))):  # Sample first 3 data rows
                row_text = ", ".join([cell for cell in table_data[i] if cell.strip()])
                if row_text:
                    sample_data.append(row_text[:100])  # Limit length
            
            if sample_data:
                description_parts.append(f"Sample data: {' | '.join(sample_data)}")
        
        return ". ".join(description_parts)
    
    def store_table(self, page_number: int, source_file: str, table_data: List[List[str]]):
        """
        Store extracted table data in SQLite database
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Convert table data to JSON for storage
            table_json = json.dumps(table_data)
            
            # Generate description
            description = self.generate_table_description(table_data)
            
            # Get column names (first row if available)
            columns = json.dumps(table_data[0]) if table_data else json.dumps([])
            
            # Insert into database
            cursor.execute('''
            INSERT INTO tables (page_number, source_file, table_data, description, columns)
            VALUES (?, ?, ?, ?, ?)
            ''', (page_number, source_file, table_json, description, columns))
            
            conn.commit()
            conn.close()
            
            print(f"Stored table from page {page_number} in database")
            
        except Exception as e:
            print(f"Error storing table: {str(e)}")
    
    def process_table_pages(self, table_pages: List[Dict]):
        """
        Process all pages containing tables
        """
        for page_info in table_pages:
            page_number = page_info['page']
            image_path = page_info['path']
            
            print(f"Extracting table from page {page_number}")
            
            # Extract table data
            table_data = self.extract_table_from_image(image_path)
            
            if table_data:
                # Store in database
                source_file = os.path.basename(image_path)
                self.store_table(page_number, source_file, table_data)
            else:
                print(f"No table data extracted from page {page_number}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python table_extractor.py <content_analysis.json>")
        sys.exit(1)
    
    # Load content analysis results
    with open(sys.argv[1], 'r') as f:
        analysis_results = json.load(f)
    
    extractor = TableExtractor()
    extractor.process_table_pages(analysis_results['table_pages'])
    
    print("Table extraction and storage complete!")