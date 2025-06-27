import json
from tabulate import tabulate
from typing import Dict, List

def render_parsed_data(parsed_data: Dict):
    print(f"\nFile: {parsed_data.get('file_path', 'Unknown')}")
    print("=" * 80)

    text_blocks = parsed_data.get("text_blocks", [])
    print("\nTEXT BLOCKS:")
    for i, block in enumerate(text_blocks, 1):
        print(f"[{block.get('page', '?')}] {block.get('type', 'Text')}: {block.get('text', '').strip()}")

    tables = parsed_data.get("tables", [])
    print("\nTABLES:")
    for i, table in enumerate(tables, 1):
        print(f"\nTable {i} on page {table.get('page', '?')} (via {table.get('extraction_method')}):")
        data = table.get("data", [])
        if data:
            print(tabulate(data, headers="keys", tablefmt="grid"))
        else:
            print("  [Empty or malformed table]")


    links = parsed_data.get("linked_content", [])
    print("\nTEXT-TABLE REFERENCES:")
    for link in links:
        ref = link.get("reference_text", "[unknown ref]")
        ref_page = link.get("reference_page", "?")
        table_id = link.get("table_id", "?")
        table_page = link.get("table_page", "?")
        print(f"'{ref}' (on page {ref_page}) refers to {table_id} (on page {table_page})")

if __name__ == "__main__":
    with open("<json file here>", "r", encoding="utf-8") as f:
        parsed_data = json.load(f)

    render_parsed_data(parsed_data)
