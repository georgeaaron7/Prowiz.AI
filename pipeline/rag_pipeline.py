# rag_pipeline.py
import sqlite3
import json
from typing import List, Dict, Any, Optional
from vector_store_manager import VectorStoreManager
import re
from sentence_transformers import SentenceTransformer
from transformers import pipeline

# Load a small local model for text generation (e.g., distilgpt2)
local_llm = pipeline("text-generation", model="distilgpt2")

class RAGPipeline:
    def __init__(self, db_path="rag_database.db", qdrant_host="localhost", qdrant_port=6333):
        self.db_path = db_path
        self.vector_manager = VectorStoreManager(
        qdrant_host=qdrant_host,
        qdrant_port=qdrant_port,
        db_path=db_path,
        use_gpu=True,                 
        embedding_model="gpu_optimized"  
        )   

        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Initialize OpenAI or your preferred LLM
        # openai.api_key = os.getenv("OPENAI_API_KEY")  # Uncomment and set your API key
    
    def enhance_query(self, query: str) -> Dict[str, Any]:
        """
        Analyze and enhance the user query to understand what types of content are needed
        """
        query_lower = query.lower()
        
        # Detect query intent
        intent_indicators = {
            'table': ['table', 'data', 'values', 'statistics', 'numbers', 'chart', 'column', 'row'],
            'formula': ['formula', 'equation', 'calculate', 'computation', 'math', 'derive', 'solve'],
            'text': ['explain', 'describe', 'what is', 'definition', 'concept', 'theory', 'information']
        }
        
        detected_intents = []
        for intent, keywords in intent_indicators.items():
            if any(keyword in query_lower for keyword in keywords):
                detected_intents.append(intent)
        
        # If no specific intent detected, include all types
        if not detected_intents:
            detected_intents = ['table', 'formula', 'text']
        
        # Extract potential variable names or specific terms
        variables = re.findall(r'\b[a-zA-Z][a-zA-Z0-9]*\b', query)
        numbers = re.findall(r'\d+\.?\d*', query)
        
        enhanced_query = {
            'original_query': query,
            'intents': detected_intents,
            'variables': variables,
            'numbers': numbers,
            'keywords': query.split()
        }
        
        return enhanced_query
    
    def search_tables(self, query: str, limit: int = 3) -> List[Dict]:
        """
        Search for relevant tables in the database
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Search tables by description and columns
            cursor.execute('''
            SELECT id, page_number, source_file, table_data, description, columns
            FROM tables
            WHERE description LIKE ? OR columns LIKE ?
            ORDER BY page_number
            LIMIT ?
            ''', (f'%{query}%', f'%{query}%', limit))
            
            results = []
            for row in cursor.fetchall():
                table_data = json.loads(row[3]) if row[3] else []
                columns = json.loads(row[5]) if row[5] else []
                
                results.append({
                    'id': row[0],
                    'page_number': row[1],
                    'source_file': row[2],
                    'table_data': table_data,
                    'description': row[4],
                    'columns': columns,
                    'type': 'table'
                })
            
            conn.close()
            return results
            
        except Exception as e:
            print(f"Error searching tables: {str(e)}")
            return []
    
    def search_formulas(self, query: str, limit: int = 3) -> List[Dict]:
        """
        Search for relevant formulas in the database
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Search formulas by original text, variables, and description
            cursor.execute('''
            SELECT id, page_number, source_file, original_formula, parsed_formula, 
                   formula_type, variables, description, executable_code
            FROM formulas
            WHERE original_formula LIKE ? OR description LIKE ? OR variables LIKE ?
            ORDER BY page_number
            LIMIT ?
            ''', (f'%{query}%', f'%{query}%', f'%{query}%', limit))
            
            results = []
            for row in cursor.fetchall():
                variables = json.loads(row[6]) if row[6] else []
                
                results.append({
                    'id': row[0],
                    'page_number': row[1],
                    'source_file': row[2],
                    'original_formula': row[3],
                    'parsed_formula': row[4],
                    'formula_type': row[5],
                    'variables': variables,
                    'description': row[7],
                    'executable_code': row[8],
                    'type': 'formula'
                })
            
            conn.close()
            return results
            
        except Exception as e:
            print(f"Error searching formulas: {str(e)}")
            return []
    
    def search_text(self, query: str, limit: int = 5) -> List[Dict]:
        """
        Search for relevant text content using vector similarity
        """
        return self.vector_manager.search_similar_text(query, limit)
    
    def execute_formula(self, formula_code: str, variables: Dict[str, float]) -> Optional[float]:
        """
        Execute a formula with given variable values
        """
        try:
            # Create a safe execution environment
            safe_dict = {
                '__builtins__': {},
                'math': __import__('math'),
                'numpy': __import__('numpy'),
                'np': __import__('numpy'),
            }
            safe_dict.update(variables)
            
            # Execute the formula code
            exec(formula_code, safe_dict)
            
            # Call the calculate function
            if 'calculate' in safe_dict:
                if variables:
                    return safe_dict['calculate'](**variables)
                else:
                    return safe_dict['calculate']()
            
            return None
            
        except Exception as e:
            print(f"Error executing formula: {str(e)}")
            return None
    
    def generate_response(self, query: str, context: Dict[str, List]) -> str:
        """
        Generate a comprehensive response using retrieved context
        """
        # Build context string
        context_parts = []
        
        # Add table information
        if context.get('tables'):
            context_parts.append("RELEVANT TABLES:")
            for table in context['tables']:
                context_parts.append(f"Page {table['page_number']}: {table['description']}")
                if table['table_data'] and len(table['table_data']) > 0:
                    # Show first few rows as sample
                    sample_rows = table['table_data'][:3]
                    for i, row in enumerate(sample_rows):
                        context_parts.append(f"Row {i+1}: {', '.join(row)}")
        
        # Add formula information
        if context.get('formulas'):
            context_parts.append("\nRELEVANT FORMULAS:")
            for formula in context['formulas']:
                context_parts.append(f"Page {formula['page_number']}: {formula['description']}")
                context_parts.append(f"Formula: {formula['original_formula']}")
                if formula['variables']:
                    context_parts.append(f"Variables: {', '.join(formula['variables'])}")
        
        # Add text information
        if context.get('text'):
            context_parts.append("\nRELEVANT TEXT CONTENT:")
            for text in context['text']:
                context_parts.append(f"Page {text['page_number']} (Score: {text['score']:.3f}): {text['text_content'][:200]}...")
        
        context_str = "\n".join(context_parts)
        
        # Simple response generation (you can replace with LLM API call)
        response_parts = [f"Based on the analyzed PDF content, here's the information relevant to your query: '{query}'"]
        
        if context.get('tables'):
            response_parts.append(f"\nFound {len(context['tables'])} relevant table(s) with data that may answer your question.")
        
        if context.get('formulas'):
            response_parts.append(f"\nFound {len(context['formulas'])} relevant formula(s) that may be applicable.")
        
        if context.get('text'):
            response_parts.append(f"\nFound {len(context['text'])} relevant text passage(s) with related information.")
        
        response_parts.append(f"\n--- DETAILED CONTEXT ---\n{context_str}")
        
        return "\n".join(response_parts)
    
    def process_query(self, query: str) -> str:
        """
        Main pipeline to process user query and generate response
        """
        print(f"Processing query: {query}")
        
        # Enhance query understanding
        enhanced_query = self.enhance_query(query)
        print(f"Detected intents: {enhanced_query['intents']}")
        
        # Search different content types based on intent
        context = {}
        
        if 'table' in enhanced_query['intents']:
            context['tables'] = self.search_tables(query)
            print(f"Found {len(context['tables'])} relevant tables")
        
        if 'formula' in enhanced_query['intents']:
            context['formulas'] = self.search_formulas(query)
            print(f"Found {len(context['formulas'])} relevant formulas")
        
        if 'text' in enhanced_query['intents']:
            context['text'] = self.search_text(query)
            print(f"Found {len(context['text'])} relevant text chunks")
        
        # Generate comprehensive response
        response = self.generate_response(query, context)
        
        return response

if __name__ == "__main__":
    # Initialize RAG pipeline
    rag = RAGPipeline()
    
    # Interactive query loop
    print("RAG Pipeline initialized. Type 'quit' to exit.")
    print("Ask questions about the processed PDF content.")
    
    while True:
        query = input("\nEnter your query: ").strip()
        
        if query.lower() in ['quit', 'exit', 'q']:
            break
        
        if not query:
            continue
        
        try:
            response = rag.process_query(query)
            print(f"\nResponse:\n{response}")
        except Exception as e:
            print(f"Error processing query: {str(e)}")
    
    print("RAG Pipeline session ended.")