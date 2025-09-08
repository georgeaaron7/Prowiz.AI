from sentence_transformers import SentenceTransformer
import pytesseract
from PIL import Image
import cv2
import numpy as np
import logging
import json
import uuid
from pathlib import Path
import re

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TextExtractor:
    def close(self):
        """No vector DB connection to close for FAISS"""
        pass

    def __init__(self):
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        
    def extract_text_from_image(self, image_path):
        """Extract text from a single image using Tesseract OCR"""
        try:
            # Load the image
            img = Image.open(image_path)
            
            # Convert the image to RGB (in case it's in a different format)
            img = img.convert("RGB")
            
            # Perform OCR on the image
            text = pytesseract.image_to_string(img)
            
            return text.strip()
        except Exception as e:
            logger.error(f"Error extracting text from image {image_path}: {e}")
            return ""
    
    def extract_text_from_pdf(self, pdf_path):
        """Extract text from a PDF file (multi-page)"""
        all_text = []
        try:
            # Read the PDF file
            pdf = cv2.PDF(pdf_path)
            
            # Iterate over each page in the PDF
            for page_number in range(len(pdf)):
                # Extract text from the page
                text = pytesseract.image_to_string(pdf[page_number])
                all_text.append({"page_number": page_number + 1, "text": text})
                
        except Exception as e:
            logger.error(f"Error extracting text from PDF {pdf_path}: {e}")
        
        return all_text
    
    def chunk_text(self, text, max_length=500):
        """Chunk long text into smaller parts"""
        chunks = []
        words = text.split()
        current_chunk = []
        current_length = 0
        
        for word in words:
            word_length = len(word) + 1  # +1 for the space or punctuation
            if current_length + word_length > max_length:
                # Current chunk is full, save it and start a new one
                chunks.append(" ".join(current_chunk))
                current_chunk = [word]
                current_length = word_length
            else:
                # Add the word to the current chunk
                current_chunk.append(word)
                current_length += word_length
        
        # Don't forget to add the last chunk
        if current_chunk:
            chunks.append(" ".join(current_chunk))
        
        return chunks
    
    def process_and_store(self, source_path, document_id, is_image=True):
        """Process an image or PDF file and store the extracted text in chunks"""
        try:
            if is_image:
                # Extract text from a single image
                text = self.extract_text_from_image(source_path)
                logger.info(f"Extracted text from image: {text[:100]}...")  # Log first 100 chars
            else:
                # Extract text from a PDF file
                all_text = self.extract_text_from_pdf(source_path)
                logger.info(f"Extracted text from PDF with {len(all_text)} pages")
                
                # For each page, extract and log the text
                for page in all_text:
                    logger.info(f"Page {page['page_number']} text: {page['text'][:100]}...")  # Log first 100 chars
            
            # Here you would store the extracted text in your vector database (e.g., FAISS)
            # This part is omitted as it depends on your specific database setup and requirements
            
        except Exception as e:
            logger.error(f"Error processing and storing data from {source_path}: {e}")
                    
                    
                    # Removed undefined 'collection' block
            
            logger.info(f"Stored {chunk_counter} text chunks from page {page_number}")
            return True
            
        except Exception as e:
            logger.error(f"Error storing text in Weaviate: {e}")
            return False
