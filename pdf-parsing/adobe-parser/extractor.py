import logging
import os
import json
import zipfile
from datetime import datetime

from adobe.pdfservices.operation.auth.service_principal_credentials import ServicePrincipalCredentials
from adobe.pdfservices.operation.exception.exceptions import ServiceApiException, ServiceUsageException, SdkException
from adobe.pdfservices.operation.io.cloud_asset import CloudAsset
from adobe.pdfservices.operation.io.stream_asset import StreamAsset
from adobe.pdfservices.operation.pdf_services import PDFServices
from adobe.pdfservices.operation.pdf_services_media_type import PDFServicesMediaType
from adobe.pdfservices.operation.pdfjobs.jobs.extract_pdf_job import ExtractPDFJob
from adobe.pdfservices.operation.pdfjobs.params.extract_pdf.extract_element_type import ExtractElementType
from adobe.pdfservices.operation.pdfjobs.params.extract_pdf.extract_pdf_params import ExtractPDFParams
from adobe.pdfservices.operation.pdfjobs.result.extract_pdf_result import ExtractPDFResult

# Initialize the logger
logging.basicConfig(level=logging.INFO)

class PDFDataExtractor:
    def __init__(self, pdf_file_path):
        self.pdf_file_path = pdf_file_path
        
    def extract_data(self):
        try:
            # Read the PDF file
            with open(self.pdf_file_path, 'rb') as file:
                input_stream = file.read()

            # Setup credentials
            credentials = ServicePrincipalCredentials(
                client_id=os.getenv('PDF_SERVICES_CLIENT_ID'),
                client_secret=os.getenv('PDF_SERVICES_CLIENT_SECRET')
            )

            # Create PDF Services instance
            pdf_services = PDFServices(credentials=credentials)

            # Upload the PDF
            input_asset = pdf_services.upload(input_stream=input_stream, mime_type=PDFServicesMediaType.PDF)

            # Configure extraction parameters
            extract_pdf_params = ExtractPDFParams(
                elements_to_extract=[
                    ExtractElementType.TEXT, 
                    ExtractElementType.TABLES
                ],
                add_char_info=True  # Include character positioning
            )

            # Create and submit the extraction job
            extract_pdf_job = ExtractPDFJob(input_asset=input_asset, extract_pdf_params=extract_pdf_params)
            location = pdf_services.submit(extract_pdf_job)
            pdf_services_response = pdf_services.get_job_result(location, ExtractPDFResult)

            # Download the results
            result_asset: CloudAsset = pdf_services_response.get_result().get_resource()
            stream_asset: StreamAsset = pdf_services.get_content(result_asset)

            # Save the ZIP file
            output_file_path = self.create_output_file_path()
            with open(output_file_path, "wb") as file:
                file.write(stream_asset.get_input_stream())

            print(f"Extraction completed! Results saved to: {output_file_path}")
            
            # Extract and process the JSON data
            self.process_extracted_data(output_file_path)
            
            return output_file_path

        except (ServiceApiException, ServiceUsageException, SdkException) as e:
            logging.exception(f'Exception encountered while executing operation: {e}')
            return None

    def process_extracted_data(self, zip_file_path):
        """Extract and process the JSON data from the ZIP file"""
        try:
            with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
                # List all files in the ZIP
                file_list = zip_ref.namelist()
                print(f"Files in ZIP: {file_list}")
                
                # Find the JSON file (usually named structuredData.json)
                json_file = None
                for file_name in file_list:
                    if file_name.endswith('.json'):
                        json_file = file_name
                        break
                
                if json_file:
                    # Extract and read the JSON data
                    with zip_ref.open(json_file) as json_data:
                        data = json.loads(json_data.read().decode('utf-8'))
                        
                        # Save processed data to separate files
                        self.save_extracted_text(data)
                        self.save_extracted_tables(data)
                        
                        print("Data processing completed!")
                else:
                    print("No JSON file found in the extracted ZIP")
                    
        except Exception as e:
            print(f"Error processing extracted data: {e}")

    def save_extracted_text(self, data):
        """Save extracted text to a file"""
        try:
            output_dir = os.path.dirname(self.create_output_file_path())
            text_file_path = os.path.join(output_dir, "extracted_text.txt")
            
            with open(text_file_path, 'w', encoding='utf-8') as f:
                if 'elements' in data:
                    for element in data['elements']:
                        if element.get('type') == 'text':
                            f.write(element.get('text', '') + '\n')
            
            print(f"Text extracted to: {text_file_path}")
            
        except Exception as e:
            print(f"Error saving text: {e}")

    def save_extracted_tables(self, data):
        """Save extracted tables to CSV files"""
        try:
            output_dir = os.path.dirname(self.create_output_file_path())
            table_count = 0
            
            if 'elements' in data:
                for element in data['elements']:
                    if element.get('type') == 'table':
                        table_count += 1
                        csv_file_path = os.path.join(output_dir, f"table_{table_count}.csv")
                        
                        # Convert table data to CSV
                        self.table_to_csv(element, csv_file_path)
                        print(f"Table {table_count} saved to: {csv_file_path}")
            
            if table_count == 0:
                print("No tables found in the PDF")
                
        except Exception as e:
            print(f"Error saving tables: {e}")

    def table_to_csv(self, table_element, csv_file_path):
        """Convert table element to CSV format"""
        import csv
        
        try:
            with open(csv_file_path, 'w', newline='', encoding='utf-8') as csvfile:
                writer = csv.writer(csvfile)
                
                # Process table rows
                if 'cells' in table_element:
                    # Group cells by row
                    rows = {}
                    for cell in table_element['cells']:
                        row_idx = cell.get('rowIndex', 0)
                        col_idx = cell.get('colIndex', 0)
                        text = cell.get('text', '')
                        
                        if row_idx not in rows:
                            rows[row_idx] = {}
                        rows[row_idx][col_idx] = text
                    
                    # Write rows to CSV
                    for row_idx in sorted(rows.keys()):
                        row_data = []
                        max_col = max(rows[row_idx].keys()) if rows[row_idx] else 0
                        for col_idx in range(max_col + 1):
                            row_data.append(rows[row_idx].get(col_idx, ''))
                        writer.writerow(row_data)
                        
        except Exception as e:
            print(f"Error converting table to CSV: {e}")

    @staticmethod
    def create_output_file_path() -> str:
        now = datetime.now()
        time_stamp = now.strftime("%Y-%m-%dT%H-%M-%S")
        os.makedirs("output/PDFDataExtraction", exist_ok=True)
        return f"output/PDFDataExtraction/extract_{time_stamp}.zip"


# Usage example
if __name__ == "__main__":
    # Replace with your PDF file path
    pdf_file_path = "resources/API RP 520 Sizing, Selection and Installation of Pressure-Relieving Devices in Refineries - Part-2 (2011).pdf"
    
    # Check if PDF file exists
    if not os.path.exists(pdf_file_path):
        print(f"PDF file not found: {pdf_file_path}")
        # print("Please place your PDF file in the src/resources/ directory")
    else:
        extractor = PDFDataExtractor(pdf_file_path)
        result = extractor.extract_data()
        
        if result:
            print(f"Extraction successful! Check the output directory for results.")
        else:
            print("Extraction failed. Please check the logs for errors.")
