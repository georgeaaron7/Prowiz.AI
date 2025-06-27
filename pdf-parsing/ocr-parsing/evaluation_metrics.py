import numpy as np
from typing import Dict, List

class EvaluationMetrics:
    def __init__(self):
        pass

    def evaluate(self, parsed_data: Dict) -> Dict[str, float]:
        metrics = {}
        text_blocks = parsed_data.get("text_blocks", [])
        tables = parsed_data.get("tables", [])
        links = parsed_data.get("linked_content", [])

        metrics["text_block_count"] = len(text_blocks)
        if text_blocks:
            confidences = [tb.get("confidence", 0) for tb in text_blocks if "confidence" in tb]
            metrics["avg_text_confidence"] = float(np.mean(confidences)) if confidences else 0.0
            text_lengths = [len(tb.get("text", "")) for tb in text_blocks]
            metrics["avg_text_length"] = float(np.mean(text_lengths)) if text_lengths else 0.0

        metrics["table_count"] = len(tables)
        if tables:
            accuracies = [tbl.get("accuracy", 0.0) for tbl in tables if "accuracy" in tbl]
            metrics["avg_table_accuracy"] = float(np.mean(accuracies)) if accuracies else 0.0
            sizes = [tbl.get("shape", [0, 0])[0] * tbl.get("shape", [0, 0])[1] for tbl in tables]
            metrics["avg_table_size"] = float(np.mean(sizes)) if sizes else 0.0

        metrics["linked_reference_count"] = len(links)
        if links:
            link_confidences = [link.get("confidence", 0) for link in links]
            metrics["avg_link_confidence"] = float(np.mean(link_confidences)) if link_confidences else 0.0

        total_pages = self._estimate_total_pages(parsed_data)
        metrics["text_blocks_per_page"] = len(text_blocks) / total_pages if total_pages > 0 else 0
        metrics["tables_per_page"] = len(tables) / total_pages if total_pages > 0 else 0
        metrics["linked_refs_per_page"] = len(links) / total_pages if total_pages > 0 else 0

        return metrics

    def _estimate_total_pages(self, parsed_data: Dict) -> int:
        max_page = 0
        for block in parsed_data.get("text_blocks", []):
            max_page = max(max_page, block.get("page", 0))
        for table in parsed_data.get("tables", []):
            max_page = max(max_page, table.get("page", 0))
        return max_page
