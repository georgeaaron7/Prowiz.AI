import os
import pdf2image
from PIL import Image
import argparse

class PDFToImageConverter:
    def __init__(self, output_dir="pdf_images"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    def convert_pdf_to_images(self, pdf_path, dpi=300):
        """
        Convert PDF to images and return list of image paths
        """
        try:
            # Convert PDF to images
            images = pdf2image.convert_from_path(pdf_path, dpi=dpi)
            image_paths = []
            
            pdf_name = os.path.splitext(os.path.basename(pdf_path))[0]
            
            for i, image in enumerate(images):
                # Save each page as an image
                image_path = os.path.join(self.output_dir, f"{pdf_name}_page_{i+1}.png")
                image.save(image_path, "PNG")
                image_paths.append(image_path)
                print(f"Saved page {i+1} to {image_path}")
            
            return image_paths
            
        except Exception as e:
            print(f"Error converting PDF: {str(e)}")
            return []

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Convert PDF to images')
    parser.add_argument('pdf_path', help='Path to PDF file')
    parser.add_argument('--output_dir', default='pdf_images', help='Output directory for images')
    parser.add_argument('--dpi', type=int, default=300, help='DPI for image conversion')
    
    args = parser.parse_args()
    
    converter = PDFToImageConverter(args.output_dir)
    image_paths = converter.convert_pdf_to_images(args.pdf_path, args.dpi)
    
    print(f"Converted {len(image_paths)} pages to images")
