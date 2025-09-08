import cv2
import numpy as np
from PIL import Image
import re
import json
import gc
from typing import List, Dict, Tuple
import warnings
warnings.filterwarnings("ignore")

# Multiple OCR options with different resource requirements
try:
    import easyocr
    EASYOCR_AVAILABLE = True
except ImportError:
    EASYOCR_AVAILABLE = False

try:
    import pytesseract
    PYTESSERACT_AVAILABLE = True
except ImportError:
    PYTESSERACT_AVAILABLE = False

class ContentDetector:
    def __init__(self, ocr_mode="lightweight", use_gpu=False, batch_size=4):
        """
        Initialize with different OCR modes:
        - 'lightweight': Uses pytesseract (CPU only, fast)
        - 'cpu_accurate': Uses EasyOCR on CPU (slower but more accurate)  
        - 'gpu_optimized': Uses EasyOCR on GPU with memory management
        """
        self.ocr_mode = ocr_mode
        self.use_gpu = use_gpu
        self.batch_size = batch_size
        self.ocr_reader = None
        
        self._init_ocr_engine()
    
    def _init_ocr_engine(self):
        """Initialize OCR engine based on available resources and mode"""
        if self.ocr_mode == "lightweight" and PYTESSERACT_AVAILABLE:
            print("Using lightweight Tesseract OCR (CPU-only)")
            # No initialization needed for pytesseract
            
        elif self.ocr_mode == "cpu_accurate" and EASYOCR_AVAILABLE:
            try:
                self.ocr_reader = easyocr.Reader(['en'], gpu=False, verbose=False)
                print("Using EasyOCR on CPU")
            except Exception as e:
                print(f"Failed to initialize EasyOCR CPU: {e}")
                print("Falling back to Tesseract")
                self.ocr_mode = "lightweight"
                
        elif self.ocr_mode == "gpu_optimized" and EASYOCR_AVAILABLE:
            try:
                if self.use_gpu:
                    self.ocr_reader = easyocr.Reader(['en'], gpu=True, verbose=False)
                    print("Using EasyOCR on GPU")
                else:
                    print("GPU requested but not available, using CPU")
                    self.ocr_reader = easyocr.Reader(['en'], gpu=False, verbose=False)
            except Exception as e:
                print(f"Failed to initialize EasyOCR GPU: {e}")
                print("Falling back to CPU mode")
                try:
                    self.ocr_reader = easyocr.Reader(['en'], gpu=False, verbose=False)
                except:
                    print("Falling back to Tesseract")
                    self.ocr_mode = "lightweight"
        else:
            print("Using default Tesseract OCR")
            self.ocr_mode = "lightweight"
    
    def _clear_memory(self):
        """Clear memory and GPU cache"""
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except:
            pass
    
    def detect_tables(self, image_path: str) -> bool:
        """
        Lightweight table detection using OpenCV (no OCR needed)
        """
        try:
            # Read and preprocess image efficiently
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            if image is None:
                return False
            
            # Resize if image is too large to save memory
            height, width = image.shape
            if width > 1500 or height > 1500:
                scale = min(1500/width, 1500/height)
                new_width = int(width * scale)
                new_height = int(height * scale)
                image = cv2.resize(image, (new_width, new_height))
            
            # Apply threshold
            _, binary = cv2.threshold(image, 128, 255, cv2.THRESH_BINARY_INV)
            
            # Detect lines with smaller kernels for efficiency
            h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (min(30, image.shape[1]//20), 1))
            v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, min(30, image.shape[0]//20)))
            
            horizontal_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, h_kernel)
            vertical_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, v_kernel)
            
            # Combine lines
            table_mask = cv2.addWeighted(horizontal_lines, 0.5, vertical_lines, 0.5, 0.0)
            
            # Find contours
            contours, _ = cv2.findContours(table_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Check for table-like structures
            table_contours = []
            min_area = max(500, (image.shape[0] * image.shape[1]) // 1000)  # Adaptive minimum area
            
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > min_area:
                    x, y, w, h = cv2.boundingRect(contour)
                    aspect_ratio = w / h
                    if 0.2 < aspect_ratio < 5.0:  # Filter out very thin lines
                        table_contours.append(contour)
            
            return len(table_contours) >= 2  # At least 2 table-like structures
            
        except Exception as e:
            print(f"Error detecting tables in {image_path}: {str(e)}")
            return False
        finally:
            self._clear_memory()
    
    def extract_text_lightweight(self, image_path: str) -> str:
        """Lightweight text extraction using Tesseract"""
        if not PYTESSERACT_AVAILABLE:
            return ""
        
        try:
            # Load and preprocess image
            image = Image.open(image_path)
            
            # Convert to RGB if needed
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Resize if too large to speed up OCR
            max_size = 2000
            if image.width > max_size or image.height > max_size:
                image.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
            
            # Use fast OCR configuration
            config = "--oem 3 --psm 6"
            text = pytesseract.image_to_string(image, config=config)
            
            return text.strip()
            
        except Exception as e:
            print(f"Error extracting text from {image_path}: {str(e)}")
            return ""
    
    def extract_text_easyocr(self, image_path: str) -> str:
        """EasyOCR text extraction with memory management"""
        if not self.ocr_reader:
            return self.extract_text_lightweight(image_path)
        
        try:
            # Read image
            image = cv2.imread(image_path)
            if image is None:
                return ""
            
            # Resize if too large
            height, width = image.shape[:2]
            max_dim = 1600  # Smaller max dimension for memory efficiency
            if width > max_dim or height > max_dim:
                scale = max_dim / max(width, height)
                new_width = int(width * scale)
                new_height = int(height * scale)
                image = cv2.resize(image, (new_width, new_height))
            
            # Extract text
            results = self.ocr_reader.readtext(image, paragraph=True, width_ths=0.7)
            
            # Combine results
            text_parts = []
            for result in results:
                if len(result) >= 2:
                    confidence = result[2] if len(result) > 2 else 0.5
                    if confidence > 0.3:  # Only include confident results
                        text_parts.append(result[1])
            
            return " ".join(text_parts)
            
        except Exception as e:
            print(f"Error with EasyOCR on {image_path}: {str(e)}")
            return self.extract_text_lightweight(image_path)  # Fallback
        finally:
            self._clear_memory()
    
    def extract_text(self, image_path: str) -> str:
        """Extract text using selected OCR method"""
        if self.ocr_mode == "lightweight":
            return self.extract_text_lightweight(image_path)
        else:
            return self.extract_text_easyocr(image_path)
    
    def detect_formulas_lightweight(self, text_content: str) -> List[Dict]:
        """Lightweight formula detection using regex patterns"""
        formulas = []
        
        # Comprehensive mathematical patterns
        patterns = {
            'basic_equation': r'[a-zA-Z_][a-zA-Z0-9_]*\s*[=]\s*[^=\n]+',
            'fraction': r'\d+/\d+|\([^)]+\)/\([^)]+\)',
            'power': r'[a-zA-Z0-9]+\^[0-9]+|[a-zA-Z0-9]+\*\*[0-9]+',
            'square_root': r'√[^√\s]+|sqrt\([^)]+\)',
            'trigonometric': r'(sin|cos|tan|cot|sec|csc)\([^)]+\)',
            'logarithm': r'(log|ln|lg)\([^)]+\)',
            'integral': r'∫[^∫]+d[a-zA-Z]|integral\([^)]+\)',
            'derivative': r'd[a-zA-Z]/d[a-zA-Z]|∂[^∂]+/∂[a-zA-Z]',
            'summation': r'∑[^∑]+|sum\([^)]+\)',
            'limit': r'lim[^l]*→[^l]*',
            'greek_letters': r'[αβγδεζηθικλμνξοπρστυφχψω]',
            'complex_formula': r'[a-zA-Z0-9+\-*/()^√∑∫∂=<>≤≥≠±∞]+[=][a-zA-Z0-9+\-*/()^√∑∫∂<>≤≥≠±∞]+'
        }
        
        for pattern_name, pattern in patterns.items():
            matches = re.finditer(pattern, text_content, re.IGNORECASE | re.UNICODE)
            for match in matches:
                formula_text = match.group().strip()
                if len(formula_text) > 3:  # Filter very short matches
                    formulas.append({
                        'formula': formula_text,
                        'type': pattern_name,
                        'position': match.start(),
                        'confidence': 0.7 if pattern_name in ['basic_equation', 'complex_formula'] else 0.5
                    })
        
        # Remove duplicates and sort by position
        unique_formulas = {}
        for formula in formulas:
            key = formula['formula'].lower().replace(' ', '')
            if key not in unique_formulas or unique_formulas[key]['confidence'] < formula['confidence']:
                unique_formulas[key] = formula
        
        return sorted(unique_formulas.values(), key=lambda x: x['position'])
    
    def detect_formulas_advanced(self, image_path: str) -> List[Dict]:
        """Advanced formula detection using OCR + pattern matching"""
        try:
            # Extract text first
            text_content = self.extract_text(image_path)
            if not text_content:
                return []
            
            # Use lightweight detection on extracted text
            formulas = self.detect_formulas_lightweight(text_content)
            
            # Additional image-based formula detection for symbols
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            if image is not None:
                # Resize for efficiency
                height, width = image.shape
                if width > 1200 or height > 1200:
                    scale = min(1200/width, 1200/height)
                    new_width = int(width * scale)
                    new_height = int(height * scale)
                    image = cv2.resize(image, (new_width, new_height))
                
                # Look for mathematical symbols using template matching (simplified)
                # This is a basic approach - could be enhanced with more sophisticated methods
                symbol_indicators = self._detect_math_symbols(image)
                
                if symbol_indicators > 2:  # Threshold for mathematical content
                    # If we found symbols but no formulas, add a generic entry
                    if not formulas:
                        formulas.append({
                            'formula': 'Mathematical expression detected',
                            'type': 'symbol_based',
                            'position': 0,
                            'confidence': 0.4
                        })
            
            return formulas
            
        except Exception as e:
            print(f"Error in advanced formula detection: {e}")
            return self.detect_formulas_lightweight(self.extract_text(image_path))
    
    def _detect_math_symbols(self, image) -> int:
        """Simple mathematical symbol detection using contour analysis"""
        try:
            # Apply threshold
            _, binary = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY_INV)
            
            # Find contours
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            symbol_count = 0
            for contour in contours:
                area = cv2.contourArea(contour)
                if 50 < area < 2000:  # Size range for potential symbols
                    x, y, w, h = cv2.boundingRect(contour)
                    aspect_ratio = w / h
                    
                    # Look for symbol-like characteristics
                    if 0.3 < aspect_ratio < 3.0:
                        # Check if the contour has curves (potential for symbols like ∫, ∑, etc.)
                        hull = cv2.convexHull(contour)
                        hull_area = cv2.contourArea(hull)
                        if hull_area > 0:
                            solidity = area / hull_area
                            if solidity < 0.8:  # Non-convex shapes (likely symbols)
                                symbol_count += 1
            
            return symbol_count
            
        except Exception as e:
            return 0
    
    def detect_formulas(self, image_path: str) -> List[Dict]:
        """Main formula detection method"""
        if self.ocr_mode == "lightweight":
            # Fast detection using only extracted text
            text = self.extract_text(image_path)
            return self.detect_formulas_lightweight(text)
        else:
            # More thorough detection
            return self.detect_formulas_advanced(image_path)
    
    def analyze_images_batch(self, image_paths: List[str]) -> Dict:
        """Batch analyze images with memory management"""
        analysis_results = {
            'table_pages': [],
            'formula_pages': [],
            'text_pages': [],
            'all_content': {}
        }
        
        print(f"Analyzing {len(image_paths)} images in batches of {self.batch_size}")
        
        # Process in batches to manage memory
        for i in range(0, len(image_paths), self.batch_size):
            batch_paths = image_paths[i:i + self.batch_size]
            
            print(f"Processing batch {i//self.batch_size + 1}/{(len(image_paths)-1)//self.batch_size + 1}")
            
            for j, image_path in enumerate(batch_paths):
                page_num = i + j + 1
                
                try:
                    print(f"  Analyzing page {page_num}: {image_path}")
                    
                    # Check for tables (fast, no OCR)
                    has_tables = self.detect_tables(image_path)
                    
                    # Extract text
                    text_content = self.extract_text(image_path)
                    
                    # Detect formulas
                    formulas = self.detect_formulas(image_path) if text_content else []
                    
                    # Store results
                    if has_tables:
                        analysis_results['table_pages'].append({
                            'page': page_num,
                            'path': image_path
                        })
                    
                    if formulas:
                        analysis_results['formula_pages'].append({
                            'page': page_num,
                            'path': image_path,
                            'formulas': formulas
                        })
                    
                    if text_content:
                        analysis_results['text_pages'].append({
                            'page': page_num,
                            'path': image_path,
                            'text': text_content
                        })
                    
                    analysis_results['all_content'][page_num] = {
                        'path': image_path,
                        'has_tables': has_tables,
                        'formulas': formulas,
                        'text': text_content[:200] + "..." if len(text_content) > 200 else text_content
                    }
                    
                except Exception as e:
                    print(f"  Error processing page {page_num}: {e}")
                    # Add minimal entry for failed pages
                    analysis_results['all_content'][page_num] = {
                        'path': image_path,
                        'has_tables': False,
                        'formulas': [],
                        'text': '',
                        'error': str(e)
                    }
            
            # Clear memory after each batch
            self._clear_memory()
        
        return analysis_results
    
    def analyze_images(self, image_paths: List[str]) -> Dict:
        """Main analysis method with automatic batching"""
        if len(image_paths) <= self.batch_size:
            # Small number of images, process normally
            return self._analyze_single_batch(image_paths)
        else:
            # Large number of images, use batch processing
            return self.analyze_images_batch(image_paths)
    
    def _analyze_single_batch(self, image_paths: List[str]) -> Dict:
        """Analyze small batch of images"""
        analysis_results = {
            'table_pages': [],
            'formula_pages': [],
            'text_pages': [],
            'all_content': {}
        }
        
        for i, image_path in enumerate(image_paths):
            page_num = i + 1
            print(f"Analyzing page {page_num}: {image_path}")
            
            try:
                # Check for tables
                has_tables = self.detect_tables(image_path)
                if has_tables:
                    analysis_results['table_pages'].append({
                        'page': page_num,
                        'path': image_path
                    })
                
                # Extract text
                text_content = self.extract_text(image_path)
                if text_content:
                    analysis_results['text_pages'].append({
                        'page': page_num,
                        'path': image_path,
                        'text': text_content
                    })
                
                # Detect formulas
                formulas = self.detect_formulas(image_path)
                if formulas:
                    analysis_results['formula_pages'].append({
                        'page': page_num,
                        'path': image_path,
                        'formulas': formulas
                    })
                
                # Store all content
                analysis_results['all_content'][page_num] = {
                    'path': image_path,
                    'has_tables': has_tables,
                    'formulas': formulas,
                    'text': text_content
                }
                
            except Exception as e:
                print(f"Error processing page {page_num}: {e}")
                analysis_results['all_content'][page_num] = {
                    'path': image_path,
                    'has_tables': False,
                    'formulas': [],
                    'text': '',
                    'error': str(e)
                }
        
        return analysis_results
    
    def cleanup(self):
        """Clean up resources"""
        if self.ocr_reader:
            del self.ocr_reader
        self._clear_memory()

def main():
    import sys
    import os
    import glob
    
    if len(sys.argv) < 2:
        print("Usage: python gpu_optimized_content_detector.py <image_dir> [options]")
        print("Options:")
        print("  --mode lightweight    : Fast Tesseract OCR (default)")
        print("  --mode cpu_accurate   : EasyOCR on CPU")
        print("  --mode gpu_optimized  : EasyOCR on GPU")
        print("  --batch_size N        : Process N images at a time (default: 4)")
        print("  --gpu                 : Enable GPU if available")
        sys.exit(1)
    
    image_dir = sys.argv[1]
    
    # Parse arguments
    ocr_mode = "lightweight"
    use_gpu = False
    batch_size = 4
    
    for i, arg in enumerate(sys.argv[2:], 2):
        if arg == "--mode" and i + 1 < len(sys.argv):
            ocr_mode = sys.argv[i + 1]
        elif arg == "--batch_size" and i + 1 < len(sys.argv):
            batch_size = int(sys.argv[i + 1])
        elif arg == "--gpu":
            use_gpu = True
    
    # Find image files
    image_extensions = ['*.png', '*.jpg', '*.jpeg', '*.tiff', '*.bmp']
    image_paths = []
    for ext in image_extensions:
        image_paths.extend(glob.glob(os.path.join(image_dir, ext)))
    image_paths.sort()
    
    if not image_paths:
        print(f"No images found in {image_dir}")
        sys.exit(1)
    
    print(f"Found {len(image_paths)} images")
    print(f"OCR Mode: {ocr_mode}")
    print(f"GPU: {'Enabled' if use_gpu else 'Disabled'}")
    print(f"Batch size: {batch_size}")
    
    # Initialize detector
    detector = ContentDetector(
        ocr_mode=ocr_mode,
        use_gpu=use_gpu,
        batch_size=batch_size
    )
    
    # Analyze images
    try:
        results = detector.analyze_images(image_paths)
        
        # Save results
        output_file = 'content_analysis.json'
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\nAnalysis complete:")
        print(f"- Pages with tables: {len(results['table_pages'])}")
        print(f"- Pages with formulas: {len(results['formula_pages'])}")
        print(f"- Pages with text: {len(results['text_pages'])}")
        print(f"- Results saved to: {output_file}")
        
    finally:
        detector.cleanup()

if __name__ == "__main__":
    main()
