import json
from tabulate import tabulate
from typing import Dict, List

def render_parsed_data(parsed_data: Dict):
    print(f"\n📄 File: {parsed_data.get('file_path', 'Unknown')}")
    print("=" * 80)

    # --- TEXT BLOCKS ---
    text_blocks = parsed_data.get("text_blocks", [])
    print("\n📝 TEXT BLOCKS:")
    for i, block in enumerate(text_blocks, 1):
        print(f"[{block.get('page', '?')}] {block.get('type', 'Text')}: {block.get('text', '').strip()}")

    # --- TABLES ---
    tables = parsed_data.get("tables", [])
    print("\n📊 TABLES:")
    # inside the TABLES section
    for i, table in enumerate(tables, 1):
        print(f"\nTable {i} on page {table.get('page', '?')} (via {table.get('extraction_method')}):")
        data = table.get("data", [])
        if data:
            print(tabulate(data, headers="keys", tablefmt="grid"))
        else:
            print("  [Empty or malformed table]")


    # --- TEXT-TABLE REFERENCES ---
    links = parsed_data.get("linked_content", [])
    print("\n🔗 TEXT-TABLE REFERENCES:")
    for link in links:
        ref = link.get("reference_text", "[unknown ref]")
        ref_page = link.get("reference_page", "?")
        table_id = link.get("table_id", "?")
        table_page = link.get("table_page", "?")
        print(f"'{ref}' (on page {ref_page}) refers to {table_id} (on page {table_page})")

# Example usage
if __name__ == "__main__":
    with open("output/API 5L Specification for Line Pipe (2018)-33_parsed.json", "r", encoding="utf-8") as f:
        parsed_data = json.load(f)

    render_parsed_data(parsed_data)
