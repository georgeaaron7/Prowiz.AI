import os
import asyncio
import logging
from typing import List, Dict, Any, Optional
import numpy as np
from pathlib import Path
import hashlib
import json

# Open source alternatives
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch
import requests
from llama_cloud_services import LlamaParse
import weaviate
from weaviate.classes.init import Auth
from weaviate.classes.config import Property, DataType
from weaviate.classes.query import MetadataQuery

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class OpenSourceLLM:
    """Open source LLM wrapper supporting multiple backends"""
    
    def __init__(self, model_type="huggingface", model_name="microsoft/DialoGPT-medium", **kwargs):
        self.model_type = model_type
        self.model_name = model_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {self.device}")
        
        if model_type == "huggingface":
            self._init_huggingface(**kwargs)
        elif model_type == "ollama":
            self._init_ollama(**kwargs)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
    
    def _init_huggingface(self, **kwargs):
        """Initialize Hugging Face model with CUDA support"""
        # For larger models, you might want to use 8-bit or 4-bit quantization
        load_in_8bit = kwargs.get('load_in_8bit', False)
        load_in_4bit = kwargs.get('load_in_4bit', False)
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        
        model_kwargs = {
            'torch_dtype': torch.float16 if self.device == "cuda" else torch.float32,
            **kwargs
        }
        
        if self.device == "cuda":
            model_kwargs['device_map'] = "auto"
        
        if load_in_8bit or load_in_4bit:
            model_kwargs['load_in_8bit'] = load_in_8bit
            model_kwargs['load_in_4bit'] = load_in_4bit
        
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            **model_kwargs
        )
        
        # Add padding token if it doesn't exist
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        device_id = 0 if self.device == "cuda" else -1
        
        self.generator = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            # device=device_id,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
        )
    
    def _init_ollama(self, base_url="http://localhost:11434", **kwargs):
        """Initialize Ollama client"""
        self.base_url = base_url
        
        # Check if Ollama is running
        try:
            response = requests.get(f"{self.base_url}/api/tags")
            if response.status_code != 200:
                raise ConnectionError("Ollama server not responding")
        except requests.RequestException:
            raise ConnectionError("Cannot connect to Ollama server")
    
    def complete(self, prompt: str, max_length: int = 2000, temperature: float = 0.7) -> str:
        """Generate completion for given prompt"""
        try:
            if self.model_type == "huggingface":
                return self._complete_huggingface(prompt, max_length, temperature=0.3)
            elif self.model_type == "ollama":
                return self._complete_ollama(prompt, temperature)
        except Exception as e:
            logger.error(f"Error in completion: {e}")
            return f"Error generating response: {str(e)}"
    
    def _complete_huggingface(self, prompt: str, max_length: int, temperature: float) -> str:
        """Generate completion using Hugging Face model"""
        # Format prompt for chat models
        formatted_prompt = f"You are a refinery expert. Answer the following question:\n\n{prompt}\n\nAnswer:"
        
        response = self.generator(
            formatted_prompt,
            max_new_tokens=max_length,
            temperature=temperature,
            num_return_sequences=1,
            pad_token_id=self.tokenizer.eos_token_id,
            do_sample=True,
            truncation=False
        )
        
        generated_text = response[0]['generated_text']
        # Extract only the assistant's response
        if "<|assistant|>" in generated_text:
            return generated_text.split("<|assistant|>")[-1].strip()
        return generated_text[len(formatted_prompt):].strip()
    
    def _complete_ollama(self, prompt: str, temperature: float) -> str:
        """Generate completion using Ollama"""
        response = requests.post(
            f"{self.base_url}/api/generate",
            json={
                "model": self.model_name,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": temperature}
            }
        )
        
        if response.status_code == 200:
            return response.json()['response']
        else:
            raise Exception(f"Ollama API error: {response.status_code}")

class OpenSourceRAGSystem:
    """Fully open source RAG system with CUDA support and document persistence"""
    
    def __init__(self, 
                 weaviate_url: str = "http://localhost:8080",
                 weaviate_api_key: Optional[str] = None,
                 embedding_model: str = "BAAI/bge-large-en-v1.5",
                 llm_type: str = "ollama",
                 llm_model: str = "llama3.2",
                 **llm_kwargs):
        """
        Initialize open source RAG system with CUDA support
        
        Args:
            weaviate_url: Weaviate instance URL
            weaviate_api_key: Weaviate API key (if using cloud)
            embedding_model: Hugging Face embedding model
            llm_type: Type of LLM ("huggingface" or "ollama")
            llm_model: LLM model name
            **llm_kwargs: Additional arguments for LLM initialization
        """
        # Load environment variables
        from dotenv import load_dotenv
        load_dotenv()
        
        self.llama_cloud_api_key = os.getenv("LLAMA_CLOUD_API_KEY")
        if not self.llama_cloud_api_key:
            raise ValueError("LLAMA_CLOUD_API_KEY environment variable is required")
        
        # Check CUDA availability
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {self.device}")
        
        if self.device == "cuda":
            logger.info(f"CUDA device: {torch.cuda.get_device_name()}")
            logger.info(f"CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        
        # Initialize embedding model with CUDA support
        logger.info(f"Loading embedding model: {embedding_model}")
        self.text_embedding_model = SentenceTransformer(embedding_model, device=self.device)
        self.embedding_dimension = self.text_embedding_model.get_sentence_embedding_dimension()
        
        # Initialize LLM
        logger.info(f"Loading LLM: {llm_type} - {llm_model}")
        self.llm = OpenSourceLLM(
            model_type=llm_type,
            model_name=llm_model,
            **llm_kwargs
        )
        
        # Initialize Weaviate client
        self.weaviate_client = self._initialize_weaviate(weaviate_url, weaviate_api_key)
        
        # Collection names
        self.text_collection_name = "DocumentTexts"
        self.image_collection_name = "DocumentImages"
        self.metadata_collection_name = "DocumentMetadata"
        
        # Initialize collections
        self._setup_collections()
        
        logger.info("Open Source RAG System initialized successfully")
    
    def _initialize_weaviate(self, url: str, api_key: Optional[str]):
        """Initialize Weaviate client"""
        try:
            if api_key:
                auth_config = Auth.api_key(api_key)
                client = weaviate.connect_to_weaviate_cloud(
                    cluster_url=url,
                    auth_credentials=auth_config
                )
            else:
                from urllib.parse import urlparse
                parsed = urlparse(url)
                host = parsed.hostname or "localhost"
                port = parsed.port or 8080
                client = weaviate.connect_to_local(host=host, port=port, skip_init_checks=True)
            
            if client.is_ready():
                logger.info("Successfully connected to Weaviate")
                return client
            else:
                raise Exception("Weaviate client not ready")
        except Exception as e:
            logger.error(f"Failed to connect to Weaviate: {e}")
            raise
    
    def _setup_collections(self):
        """Setup Weaviate collections"""
        try:
            # Text collection
            if not self.weaviate_client.collections.exists(self.text_collection_name):
                self.weaviate_client.collections.create(
                    name=self.text_collection_name,
                    properties=[
                        Property(name="content", data_type=DataType.TEXT),
                        Property(name="file_name", data_type=DataType.TEXT),
                        Property(name="page_number", data_type=DataType.INT),
                        Property(name="node_id", data_type=DataType.TEXT),
                        Property(name="file_hash", data_type=DataType.TEXT),
                    ],
                    vectorizer_config=weaviate.classes.config.Configure.Vectorizer.none(),
                )
                logger.info(f"Created text collection: {self.text_collection_name}")
            
            # Image collection
            if not self.weaviate_client.collections.exists(self.image_collection_name):
                self.weaviate_client.collections.create(
                    name=self.image_collection_name,
                    properties=[
                        Property(name="image_path", data_type=DataType.TEXT),
                        Property(name="file_name", data_type=DataType.TEXT),
                        Property(name="page_number", data_type=DataType.INT),
                        Property(name="node_id", data_type=DataType.TEXT),
                        Property(name="file_hash", data_type=DataType.TEXT),
                    ],
                    vectorizer_config=weaviate.classes.config.Configure.Vectorizer.none(),
                )
                logger.info(f"Created image collection: {self.image_collection_name}")
            
            # Metadata collection for tracking processed documents
            if not self.weaviate_client.collections.exists(self.metadata_collection_name):
                self.weaviate_client.collections.create(
                    name=self.metadata_collection_name,
                    properties=[
                        Property(name="file_name", data_type=DataType.TEXT),
                        Property(name="file_path", data_type=DataType.TEXT),
                        Property(name="file_hash", data_type=DataType.TEXT),
                        Property(name="processed_at", data_type=DataType.TEXT),
                        Property(name="text_nodes_count", data_type=DataType.INT),
                        Property(name="image_nodes_count", data_type=DataType.INT),
                    ],
                    vectorizer_config=weaviate.classes.config.Configure.Vectorizer.none(),
                )
                logger.info(f"Created metadata collection: {self.metadata_collection_name}")
        except Exception as e:
            logger.error(f"Error setting up collections: {e}")
            raise
    
    def _calculate_file_hash(self, file_path: str) -> str:
        """Calculate SHA256 hash of file"""
        hash_sha256 = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_sha256.update(chunk)
        return hash_sha256.hexdigest()
    
    def _is_document_processed(self, file_path: str) -> bool:
        """Check if document has already been processed"""
        try:
            file_hash = self._calculate_file_hash(file_path)
            metadata_collection = self.weaviate_client.collections.get(self.metadata_collection_name)
            
            response = metadata_collection.query.fetch_objects(
                where=weaviate.classes.query.Filter.by_property("file_hash").equal(file_hash),
                limit=1
            )
            
            if response.objects:
                logger.info(f"Document {file_path} already processed (hash: {file_hash[:8]}...)")
                return True
            
            return False
        except Exception as e:
            logger.error(f"Error checking document status: {e}")
            return False
    
    def _record_document_processing(self, file_path: str, text_nodes_count: int, image_nodes_count: int):
        """Record that document has been processed"""
        try:
            from datetime import datetime
            
            file_hash = self._calculate_file_hash(file_path)
            metadata_collection = self.weaviate_client.collections.get(self.metadata_collection_name)
            
            metadata_collection.data.insert(
                properties={
                    "file_name": Path(file_path).name,
                    "file_path": str(file_path),
                    "file_hash": file_hash,
                    "processed_at": datetime.now().isoformat(),
                    "text_nodes_count": text_nodes_count,
                    "image_nodes_count": image_nodes_count,
                }
            )
            
            logger.info(f"Recorded processing of {file_path} with {text_nodes_count} text nodes and {image_nodes_count} image nodes")
        except Exception as e:
            logger.error(f"Error recording document processing: {e}")
    
    def generate_embeddings(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings using Hugging Face model with CUDA support"""
        try:
            # Optimize batch size for CUDA
            batch_size = 64 if self.device == "cuda" else 32
            
            embeddings = self.text_embedding_model.encode(
                texts, 
                show_progress_bar=True,
                batch_size=batch_size,
                convert_to_numpy=True
            )
            return embeddings
        except Exception as e:
            logger.error(f"Error generating embeddings: {e}")
            raise
    
    async def parse_document(self, file_path: str, take_screenshot: bool = True):
        """Parse document using LlamaParse"""
        try:
            logger.info(f"Parsing document: {file_path}")
            
            parser = LlamaParse(
                api_key=self.llama_cloud_api_key,
                take_screenshot=take_screenshot,
                result_type="markdown"
            )
            
            result = await parser.aparse(file_path)
            markdown_nodes = await result.aget_markdown_nodes(split_by_page=True)
            
            image_nodes = []
            if take_screenshot:
                image_nodes = await result.aget_image_nodes(
                    include_screenshot_images=True,
                    include_object_images=False,
                    image_download_dir="./images"
                )
            
            logger.info(f"Parsed {len(markdown_nodes)} text nodes and {len(image_nodes)} image nodes")
            return markdown_nodes, image_nodes
        except Exception as e:
            logger.error(f"Error parsing document: {e}")
            raise
    
    def store_text_nodes(self, nodes, file_path: str, batch_size: int = 100):
        """Store text nodes in Weaviate"""
        try:
            logger.info(f"Storing {len(nodes)} text nodes")
            
            file_hash = self._calculate_file_hash(file_path)
            text_collection = self.weaviate_client.collections.get(self.text_collection_name)
            
            # Use larger batch size for CUDA
            if self.device == "cuda":
                batch_size = min(batch_size * 2, 200)
            
            for i in range(0, len(nodes), batch_size):
                batch_nodes = nodes[i:i + batch_size]
                batch_texts = [node.text for node in batch_nodes]
                
                embeddings = self.generate_embeddings(batch_texts)
                
                with text_collection.batch.dynamic() as batch:
                    for j, node in enumerate(batch_nodes):
                        batch.add_object(
                            properties={
                                "content": node.text,
                                "file_name": node.metadata.get("file_name", Path(file_path).name),
                                "page_number": node.metadata.get("page_number", 0),
                                "node_id": node.node_id,
                                "file_hash": file_hash,
                            },
                            vector=embeddings[j].tolist()
                        )
                
                logger.info(f"Stored batch {i//batch_size + 1}/{(len(nodes) + batch_size - 1)//batch_size}")
        except Exception as e:
            logger.error(f"Error storing text nodes: {e}")
            raise
    
    def similarity_search(self, query: str, limit: int = 5, file_hash: Optional[str] = None) -> List[Dict]:
        """Perform similarity search with optional file filtering"""
        try:
            query_embedding = self.generate_embeddings([query])[0]
            
            collection = self.weaviate_client.collections.get(self.text_collection_name)
            
            # Build where clause for file filtering
            where_clause = None
            if file_hash:
                where_clause = weaviate.classes.query.Filter.by_property("file_hash").equal(file_hash)
            
            response = collection.query.near_vector(
                near_vector=query_embedding.tolist(),
                limit=limit,
                # where=where_clause,
                return_metadata=MetadataQuery(distance=True)
            )
            
            results = []
            for obj in response.objects:
                results.append({
                    "content": obj.properties.get("content", ""),
                    "file_name": obj.properties.get("file_name", ""),
                    "page_number": obj.properties.get("page_number", 0),
                    "file_hash": obj.properties.get("file_hash", ""),
                    "distance": obj.metadata.distance if obj.metadata else None
                })
            
            return results
        except Exception as e:
            logger.error(f"Error in similarity search: {e}")
            raise
    
    def generate_response(self, query: str, context_results: List[Dict]) -> str:
        """Generate response using open source LLM"""
        try:
            prompt = ""
            for idx, result in enumerate(context_results):
                prompt += f"Context {idx+1} (Page {result['page_number']}): {result['content']}\n\n"
            prompt += f"Based on the above contexts, please answer the following question accurately and concisely:\n\nQuestion: {query}\n\nAnswer:"
            
            response = self.llm.complete(prompt, max_length=1000, temperature=0.7)
            return response
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return f"Error generating response: {str(e)}"
    
    async def process_and_store_document(self, file_path: str, take_screenshot: bool = True):
        """Process and store document with persistence checking"""
        try:
            # Check if document is already processed
            if self._is_document_processed(file_path):
                logger.info(f"Document {file_path} already processed. Skipping...")
                return
            
            # Process document
            markdown_nodes, image_nodes = await self.parse_document(file_path, take_screenshot)
            
            if markdown_nodes:
                self.store_text_nodes(markdown_nodes, file_path)
            
            # Record processing
            self._record_document_processing(file_path, len(markdown_nodes), len(image_nodes))
            
            logger.info("Document processing completed")
        except Exception as e:
            logger.error(f"Error processing document: {e}")
            raise
    
    def query(self, question: str, top_k: int = 5, file_hash: Optional[str] = None) -> Dict[str, Any]:
        """Query the RAG system"""
        try:
            results = self.similarity_search(question, limit=top_k, file_hash=file_hash)
            answer = self.generate_response(question, results)
            
            return {
                "answer": answer,
                "sources": results,
                "num_sources": len(results)
            }
        except Exception as e:
            logger.error(f"Error querying system: {e}")
            raise
    
    def get_processed_documents(self) -> List[Dict]:
        """Get list of processed documents"""
        try:
            metadata_collection = self.weaviate_client.collections.get(self.metadata_collection_name)
            response = metadata_collection.query.fetch_objects(limit=100)
            
            documents = []
            for obj in response.objects:
                documents.append({
                    "file_name": obj.properties.get("file_name", ""),
                    "file_path": obj.properties.get("file_path", ""),
                    "file_hash": obj.properties.get("file_hash", ""),
                    "processed_at": obj.properties.get("processed_at", ""),
                    "text_nodes_count": obj.properties.get("text_nodes_count", 0),
                    "image_nodes_count": obj.properties.get("image_nodes_count", 0),
                })
            
            return documents
        except Exception as e:
            logger.error(f"Error getting processed documents: {e}")
            return []
    
    def close(self):
        """Close connections"""
        if hasattr(self, 'weaviate_client'):
            self.weaviate_client.close()

# Example usage
async def main():
    # Choose your preferred models - optimized for CUDA
    rag_system = OpenSourceRAGSystem(
        embedding_model="BAAI/bge-large-en-v1.5",  # Better performance with CUDA
        llm_type="huggingface",
        llm_model="microsoft/DialoGPT-medium",  # You can use larger models with CUDA
        # For larger models, uncomment these:
        # load_in_8bit=True,  # Use 8-bit quantization to save memory
        # load_in_4bit=True,  # Use 4-bit quantization to save even more memory
    )
    
    try:
        # Show processed documents
        processed_docs = rag_system.get_processed_documents()
        if processed_docs:
            print("Already processed documents:")
            for doc in processed_docs:
                print(f"  - {doc['file_name']} ({doc['text_nodes_count']} text nodes)")
        
        # Process document (will skip if already processed)
        document_path = "pluh/api-520.pdf"
        if Path(document_path).exists():
            await rag_system.process_and_store_document(document_path)
            
            # Query: take input from user
            while True:
                question = input("Enter your question (or 'exit' to quit): ").strip()
                if question.lower() == "exit":
                    break
                if not question:
                    continue
                
                result = rag_system.query(question, top_k=3)
                print(f"\nQuestion: {question}")
                print(f"Answer: {result['answer']}")
                print(f"Sources: {result['num_sources']}")
                print("-" * 50)
        else:
            print(f"Document not found: {document_path}")
            
    finally:
        rag_system.close()

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
