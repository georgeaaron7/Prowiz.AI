somewhat working pipeline
how it works:
  first it converts all the pdf pages into images and works on ocr
  detects tables (only structured), formulas and text
  stores the formulas and tables in an sqlite database
  text in qdrant vector db 
  and the retreiveing is done by going through the sql db and the vector db
