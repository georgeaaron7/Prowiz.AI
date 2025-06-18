import json
import logging
from pathlib import Path
from parser_engine import ParserEngine
from evaluation_metrics import EvaluationMetrics

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class PDFProcessor:
    def __init__(self, data_dir="data", output_dir="output"):
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.engine = ParserEngine()
        self.evaluator = EvaluationMetrics()

        self.data_dir.mkdir(exist_ok=True)
        self.output_dir.mkdir(exist_ok=True)

    def process_all(self):
        pdf_files = list(self.data_dir.glob("*.pdf"))
        if not pdf_files:
            logger.warning("No PDF files found in data directory.")
            return

        for pdf in pdf_files:
            try:
                logger.info(f"Processing: {pdf.name}")
                result = self.engine.parse_pdf(str(pdf))

                # Save parsing result
                result_file = self.output_dir / f"{pdf.stem}_parsed.json"
                with open(result_file, "w", encoding="utf-8") as f:
                    json.dump(result, f, indent=2, ensure_ascii=False)
                logger.info(f"Saved parsed output to {result_file}")

                # Evaluate and save metrics
                metrics = self.evaluator.evaluate(result)
                metrics_file = self.output_dir / f"{pdf.stem}_metrics.json"
                with open(metrics_file, "w", encoding="utf-8") as f:
                    json.dump(metrics, f, indent=2, ensure_ascii=False)
                logger.info(f"Saved evaluation metrics to {metrics_file}")

            except Exception as e:
                logger.error(f"Error processing {pdf.name}: {e}")

if __name__ == "__main__":
    processor = PDFProcessor()
    processor.process_all()
