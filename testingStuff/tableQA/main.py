import pymupdf
import pandas as pd
import json
import os
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

def extractTableInfo(file_path, saved_pathFile=None):
    doc = pymupdf.open(file_path)
    all_records = []

    for page_num in range(len(doc)):
        page = doc[page_num]
        tabs = page.find_tables()

        if tabs.tables:
            for tab in tabs.tables:
                df = tab.to_pandas()
                records = json.loads(df.to_json(orient='records'))
                all_records.extend(records)

    if saved_pathFile:
        with open(saved_pathFile, 'w') as f:
            json.dump(all_records, f, indent=2)
    
    return all_records

file_path = "testingStuff/tableQA/API 5L Specification for Line Pipe (2018)-27.pdf"
saved_pathFile = "testingStuff/tableQA/tableData.json"
data = extractTableInfo(file_path, saved_pathFile)


json_path = os.path.join(os.path.dirname(__file__), "tableData.json")
with open(json_path, "r") as f:
    table_rows = json.load(f)

texts = []
for row in table_rows:
    # build a single‐line description of each row
    entries = []
    for col, val in row.items():
        if val is not None:
            # sanitize newlines in long cells
            cell = str(val).replace("\n", " ")
            entries.append(f"{col}: {cell}")
    texts.append(" | ".join(entries))

docs = [
    Document(page_content=txt, metadata={"row_index": i})
    for i, txt in enumerate(texts)
]

# Now chunk exactly like before
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
chunked_docs = splitter.split_documents(docs)

# Finally rebuild texts for embedding
texts = [doc.page_content for doc in chunked_docs]
print(f"[INFO] After chunking: {len(texts)} chunks")

print(texts)