# gpu_optimized_main.py
import os
import sys
import argparse
import json
import gc
from pdf_converter import PDFToImageConverter
from content_detector import ContentDetector
from table_extractor import TableExtractor
from formula_processor import FormulaProcessor

try:
    from vector_store_manager import VectorStoreManager
    VECTOR_STORE_AVAILABLE = True
except ImportError as e:
    print(f"Warning: GPU optimized vector store not available: {e}")
    try:
        from vector_store_manager import VectorStoreManager
        VECTOR_STORE_AVAILABLE = True
        print("Using standard vector store manager")
    except ImportError:
        VECTOR_STORE_AVAILABLE = False
        print("No vector store available")

from rag_pipeline import RAGPipeline

class PDFRAGOrchestrator:
    def __init__(self, output_dir="pdf_processing", db_path="rag_database.db", 
                 gpu_mode="auto", batch_size=4, ocr_mode="lightweight"):
        self.output_dir = output_dir
        self.db_path = db_path
        self.image_dir = os.path.join(output_dir, "images")
        self.analysis_file = os.path.join(output_dir, "content_analysis.json")
        
        # GPU optimization settings
        self.gpu_mode = gpu_mode
        self.batch_size = batch_size
        self.ocr_mode = ocr_mode
        self.use_gpu = self._determine_gpu_usage()
        
        # Create directories
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(self.image_dir, exist_ok=True)
        
        print(f"GPU Mode: {self.gpu_mode}")
        print(f"Using GPU: {self.use_gpu}")
        print(f"OCR Mode: {self.ocr_mode}")
        print(f"Batch Size: {self.batch_size}")
    
    def _determine_gpu_usage(self):
        """Determine if GPU should be used based on availability and mode"""
        if self.gpu_mode == "disable":
            return False
        
        try:
            import torch
            gpu_available = torch.cuda.is_available()
            
            if self.gpu_mode == "auto":
                if gpu_available:
                    # Check GPU memory
                    gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
                    print(f"GPU detected: {torch.cuda.get_device_name(0)} ({gpu_memory:.1f}GB)")
                    
                    # Use GPU only if we have sufficient memory
                    if gpu_memory >= 4.0:  # At least 4GB
                        return True
                    else:
                        print("GPU memory insufficient for optimization, using CPU")
                        return False
                else:
                    print("No GPU detected, using CPU")
                    return False
            
            elif self.gpu_mode == "force":
                if gpu_available:
                    return True
                else:
                    print("GPU forced but not available, falling back to CPU")
                    return False
            
        except ImportError:
            print("PyTorch not available, using CPU")
            return False
        
        return False
    
    def _clear_memory(self):
        """Clear memory and GPU cache"""
        gc.collect()
        if self.use_gpu:
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except ImportError:
                pass
    
    def process_pdf(self, pdf_path: str, dpi: int = 300):
        """
        Complete PDF processing pipeline with GPU optimization
        """
        print("=" * 60)
        print("GPU-OPTIMIZED PDF RAG PROCESSING PIPELINE")
        print("=" * 60)
        
        try:
            # Step 1: Convert PDF to images
            print("\n1. Converting PDF to images...")
            converter = PDFToImageConverter(self.image_dir)
            image_paths = converter.convert_pdf_to_images(pdf_path, dpi)
            
            if not image_paths:
                print("Error: No images were created from PDF")
                return False
            
            print(f"✓ Created {len(image_paths)} page images")
            self._clear_memory()
            
            # Step 2: Analyze content types with GPU optimization
            print(f"\n2. Analyzing content types using {self.ocr_mode} mode...")
            detector = ContentDetector(
                ocr_mode=self.ocr_mode,
                use_gpu=self.use_gpu,
                batch_size=self.batch_size
            )
            
            try:
                analysis_results = detector.analyze_images(image_paths)
                
                # Save analysis results
                with open(self.analysis_file, 'w') as f:
                    json.dump(analysis_results, f, indent=2)
                
                print(f"✓ Analysis complete:")
                print(f"  - Pages with tables: {len(analysis_results['table_pages'])}")
                print(f"  - Pages with formulas: {len(analysis_results['formula_pages'])}")
                print(f"  - Pages with text: {len(analysis_results['text_pages'])}")
                
            finally:
                detector.cleanup()
                self._clear_memory()
            
            # Step 3: Extract and store tables
            if analysis_results['table_pages']:
                print(f"\n3. Extracting tables from {len(analysis_results['table_pages'])} pages...")
                table_extractor = TableExtractor(self.db_path)
                table_extractor.process_table_pages(analysis_results['table_pages'])
                print(f"✓ Processed tables")
                self._clear_memory()
            else:
                print("\n3. No tables found to extract")
            
            # Step 4: Process formulas
            if analysis_results['formula_pages']:
                print(f"\n4. Processing formulas from {len(analysis_results['formula_pages'])} pages...")
                formula_processor = FormulaProcessor(self.db_path)
                formula_processor.process_formula_pages(analysis_results['formula_pages'])
                print(f"✓ Processed formulas")
                self._clear_memory()
            else:
                print("\n4. No formulas found to process")
            
            # Step 5: Store text in vector database with GPU optimization
            if analysis_results['text_pages'] and VECTOR_STORE_AVAILABLE:
                print(f"\n5. Storing text in vector database...")
                
                # Choose appropriate embedding model based on GPU availability
                if self.use_gpu and 'VectorStoreManager' in globals():
                    vector_manager = VectorStoreManager(
                        db_path=self.db_path,
                        use_gpu=True,
                        batch_size=max(4, self.batch_size // 2),  # Smaller batch for embeddings
                        embedding_model="gpu_optimized"
                    )
                elif 'VectorStoreManager' in globals():
                    vector_manager = VectorStoreManager(
                        db_path=self.db_path,
                        use_gpu=False,
                        batch_size=self.batch_size * 2,  # Larger batch for CPU
                        embedding_model="lightweight" if self.ocr_mode == "lightweight" else "cpu_optimized"
                    )
                else:
                    # Fallback to original vector manager
                    vector_manager = VectorStoreManager(db_path=self.db_path)
                
                try:
                    vector_manager.store_text_content(analysis_results['text_pages'])
                    print(f"✓ Stored text from {len(analysis_results['text_pages'])} pages")
                finally:
                    if hasattr(vector_manager, 'cleanup'):
                        vector_manager.cleanup()
                    self._clear_memory()
            
            elif analysis_results['text_pages']:
                print("\n5. Vector store not available, skipping text storage")
            else:
                print("\n5. No text content found to store")
            
            print("\n" + "=" * 60)
            print("PDF PROCESSING COMPLETE!")
            print("=" * 60)
            print(f"Database: {self.db_path}")
            print(f"Analysis: {self.analysis_file}")
            print(f"Images: {self.image_dir}")
            
            return True
            
        except Exception as e:
            print(f"\nError during processing: {str(e)}")
            return False
        finally:
            self._clear_memory()
    
    def start_query_interface(self):
        """
        Start interactive query interface
        """
        print("\n" + "=" * 60)
        print("STARTING RAG QUERY INTERFACE")
        print("=" * 60)
        
        try:
            rag = RAGPipeline(db_path=self.db_path)
            
            print("\nRAG system ready! You can now ask questions about the processed PDF.")
            # print("\nOptimization Status:")
            # print(f"- GPU Mode: {self.gpu_mode}")
            # print(f"- OCR Mode: {self.ocr_mode}")
            print(f"- Batch Size: {self.batch_size}")
            
            # print("\nExample queries:")
            # print("- 'Show me tables with financial data'")
            # print("- 'Find formulas related to velocity'")
            # print("- 'Explain the concept mentioned on page 5'")
            # print("- 'What mathematical equations are in this document?'")
            # print("\nType 'quit' to exit.")
            
            while True:
                print("\n" + "-" * 40)
                query = input("Enter your query: ").strip()
                
                if query.lower() in ['quit', 'exit', 'q']:
                    break
                
                if not query:
                    continue
                
                try:
                    response = rag.process_query(query)
                    print(f"\nResponse:\n{response}")
                except Exception as e:
                    print(f"Error processing query: {str(e)}")
                finally:
                    self._clear_memory()
            
            print("\nQuery session ended.")
            
        except Exception as e:
            print(f"Error starting RAG interface: {str(e)}")
        finally:
            self._clear_memory()

def main():
    parser = argparse.ArgumentParser(description='GPU-Optimized PDF RAG Processing Pipeline')
    parser.add_argument('pdf_path', help='Path to PDF file to process')
    parser.add_argument('--output_dir', default='pdf_processing', help='Output directory')
    parser.add_argument('--db_path', default='rag_database.db', help='SQLite database path')
    parser.add_argument('--dpi', type=int, default=300, help='DPI for PDF to image conversion')
    parser.add_argument('--no_query', action='store_true', help='Skip query interface after processing')
    
    # GPU optimization arguments
    parser.add_argument('--gpu_mode', choices=['auto', 'force', 'disable'], default='auto',
                       help='GPU usage mode: auto (use if available), force (require GPU), disable (CPU only)')
    parser.add_argument('--batch_size', type=int, default=4,
                       help='Batch size for processing (lower values use less memory)')
    parser.add_argument('--ocr_mode', choices=['lightweight', 'cpu_accurate', 'gpu_optimized'], 
                       default='lightweight',
                       help='OCR mode: lightweight (Tesseract), cpu_accurate (EasyOCR CPU), gpu_optimized (EasyOCR GPU)')
    
    # Memory optimization arguments
    parser.add_argument('--low_memory', action='store_true',
                       help='Use low memory settings (smaller batches, lightweight models)')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.pdf_path):
        print(f"Error: PDF file not found: {args.pdf_path}")
        sys.exit(1)
    
    # Adjust settings for low memory mode
    if args.low_memory:
        args.batch_size = min(args.batch_size, 2)
        if args.ocr_mode == 'gpu_optimized':
            print("Low memory mode: switching to lightweight OCR")
            args.ocr_mode = 'lightweight'
        if args.gpu_mode == 'auto':
            args.gpu_mode = 'disable'
            print("Low memory mode: disabling GPU")
    
    print("GPU-Optimized PDF RAG Pipeline")
    print(f"PDF: {args.pdf_path}")
    print(f"GPU Mode: {args.gpu_mode}")
    print(f"OCR Mode: {args.ocr_mode}")
    print(f"Batch Size: {args.batch_size}")
    print(f"Low Memory: {args.low_memory}")
    
    # Initialize orchestrator
    orchestrator = PDFRAGOrchestrator(
        output_dir=args.output_dir,
        db_path=args.db_path,
        gpu_mode=args.gpu_mode,
        batch_size=args.batch_size,
        ocr_mode=args.ocr_mode
    )
    
    # Process PDF
    success = orchestrator.process_pdf(args.pdf_path, args.dpi)
    
    if success and not args.no_query:
        # Start query interface
        orchestrator.start_query_interface()
    elif success:
        print(f"\nProcessing complete. To start queries later, run:")
        print(f"python rag_pipeline.py")
    
    # Final cleanup
    orchestrator._clear_memory()

if __name__ == "__main__":
    main()
