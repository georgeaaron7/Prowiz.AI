from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
import sqlite3
import json
import uuid
from typing import List, Dict, Any
import numpy as np
import gc
import torch

# Multiple fallback embedding approaches
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False

# Lightweight alternatives
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.decomposition import TruncatedSVD
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

import pickle
import os

class VectorStoreManager:
    def __init__(self, qdrant_host="localhost", qdrant_port=6333, db_path="rag_database.db", 
                 use_gpu=False, batch_size=16, embedding_model="lightweight"):
        self.client = QdrantClient(host=qdrant_host, port=qdrant_port)
        self.db_path = db_path
        self.collection_name = "pdf_content"
        self.use_gpu = use_gpu and torch.cuda.is_available()
        self.batch_size = batch_size
        self.device = 'cuda' if self.use_gpu else 'cpu'
        
        # Initialize embedding model with GPU optimization
        self._init_embedding_model(embedding_model)
        
        self.init_vector_collection()
        self.init_database()
    
    def _init_embedding_model(self, model_type="lightweight"):
        """Initialize the most appropriate embedding model for available resources"""
        
        if model_type == "lightweight" or not SENTENCE_TRANSFORMERS_AVAILABLE:
            self._init_tfidf_embedding()
            return
            
        if model_type == "cpu_optimized":
            self._init_cpu_sentence_transformer()
        elif model_type == "gpu_optimized" and self.use_gpu:
            self._init_gpu_sentence_transformer()
        else:
            self._init_tfidf_embedding()
    
    def _init_tfidf_embedding(self):
        """Ultra-lightweight TF-IDF embedding (no GPU needed)"""
        self.embedding_method = "tfidf"
        self.embedding_dim = 256  # Smaller dimension for efficiency
        self.vectorizer = TfidfVectorizer(
            max_features=300,  # Reduced features
            stop_words='english',
            ngram_range=(1, 2),
            min_df=1,
            max_df=0.95
        )
        self.svd = TruncatedSVD(n_components=self.embedding_dim, random_state=42)
        self.is_fitted = False
        self.model_path = "lightweight_tfidf_model.pkl"
        print(f"Using lightweight TF-IDF embeddings (dim: {self.embedding_dim}, CPU-only)")
    
    def _init_cpu_sentence_transformer(self):
        """CPU-optimized sentence transformer"""
        try:
            self.model = SentenceTransformer('all-MiniLM-L6-v2', device='cpu')
            self.model.max_seq_length = 256  # Limit sequence length
            self.embedding_dim = 384
            self.embedding_method = "sentence_transformers_cpu"
            print("Using CPU-optimized SentenceTransformers")
        except Exception as e:
            print(f"Failed to load CPU SentenceTransformers: {e}")
            self._init_tfidf_embedding()
    
    def _init_gpu_sentence_transformer(self):
        """GPU-optimized sentence transformer with memory management"""
        try:
            # Use smaller model for GPU efficiency
            self.model = SentenceTransformer('all-MiniLM-L6-v2', device=self.device)
            self.model.max_seq_length = 128  # Shorter sequences for GPU memory
            self.embedding_dim = 384
            self.embedding_method = "sentence_transformers_gpu"
            
            # Enable mixed precision if available
            if hasattr(torch, 'cuda') and torch.cuda.is_available():
                try:
                    self.model.half()  # Use FP16 to save GPU memory
                    print("Using GPU-optimized SentenceTransformers with FP16")
                except:
                    print("Using GPU-optimized SentenceTransformers with FP32")
            
        except Exception as e:
            print(f"Failed to load GPU SentenceTransformers: {e}")
            self._init_cpu_sentence_transformer()
    
    def _clear_gpu_cache(self):
        """Clear GPU cache to prevent memory issues"""
        if self.use_gpu and torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()
    
    def encode(self, text: str) -> List[float]:
        """Memory-efficient text encoding"""
        if self.embedding_method == "tfidf":
            return self._encode_tfidf(text)
        elif "sentence_transformers" in self.embedding_method:
            return self._encode_sentence_transformer(text)
        else:
            return [0.0] * self.embedding_dim
    
    def _encode_tfidf(self, text: str) -> List[float]:
        """TF-IDF encoding (CPU-only, very lightweight)"""
        if not self.is_fitted:
            return [0.0] * self.embedding_dim
        
        try:
            tfidf_vec = self.vectorizer.transform([text])
            embedding = self.svd.transform(tfidf_vec)
            return embedding[0].tolist()
        except:
            return [0.0] * self.embedding_dim
    
    def _encode_sentence_transformer(self, text: str) -> List[float]:
        """Sentence transformer encoding with memory optimization"""
        try:
            # Truncate text to prevent memory issues
            if len(text) > 500:
                text = text[:500]
            
            # Encode with no gradient computation to save memory
            with torch.no_grad():
                if self.use_gpu:
                    embedding = self.model.encode(text, show_progress_bar=False, 
                                                convert_to_numpy=True, normalize_embeddings=True)
                else:
                    embedding = self.model.encode(text, show_progress_bar=False, 
                                                convert_to_tensor=False)
            
            return embedding.tolist() if hasattr(embedding, 'tolist') else embedding
            
        except Exception as e:
            print(f"Error encoding text: {e}")
            return [0.0] * self.embedding_dim
        finally:
            self._clear_gpu_cache()
    
    def encode_batch(self, texts: List[str]) -> List[List[float]]:
        """Batch encoding with memory management"""
        if self.embedding_method == "tfidf":
            return [self._encode_tfidf(text) for text in texts]
        
        embeddings = []
        # Process in smaller batches to prevent GPU memory overflow
        for i in range(0, len(texts), self.batch_size):
            batch_texts = texts[i:i + self.batch_size]
            
            # Truncate texts
            batch_texts = [text[:500] for text in batch_texts]
            
            try:
                with torch.no_grad():
                    if self.use_gpu:
                        batch_embeddings = self.model.encode(batch_texts, 
                                                           show_progress_bar=False,
                                                           convert_to_numpy=True,
                                                           normalize_embeddings=True)
                    else:
                        batch_embeddings = self.model.encode(batch_texts,
                                                           show_progress_bar=False,
                                                           convert_to_tensor=False)
                
                embeddings.extend(batch_embeddings.tolist() if hasattr(batch_embeddings, 'tolist') 
                                else batch_embeddings)
                
            except Exception as e:
                print(f"Error in batch encoding: {e}")
                # Fallback to individual encoding
                for text in batch_texts:
                    embeddings.append(self.encode(text))
            
            finally:
                self._clear_gpu_cache()
        
        return embeddings
    
    def init_database(self):
        """Initialize text storage table"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS tables (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            page_number INTEGER,
            source_file TEXT,
            table_data TEXT,
            description TEXT,
            columns TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        ''')

        cursor.execute('''
        CREATE TABLE IF NOT EXISTS formulas (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            page_number INTEGER,
            source_file TEXT,
            original_formula TEXT,
            parsed_formula TEXT,
            formula_type TEXT,
            variables TEXT,
            description TEXT,
            executable_code TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        ''')

        
        conn.commit()
        conn.close()
    
    def init_vector_collection(self):
        """Initialize or fix Qdrant collection with correct dimension"""
        try:
            collections = self.client.get_collections()
            collection_exists = any(c.name == self.collection_name for c in collections.collections)

            if not collection_exists:
                # Fresh create
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(size=self.embedding_dim, distance=Distance.COSINE)
                )
                print(f"Created Qdrant collection: {self.collection_name} (dim={self.embedding_dim})")

            else:
                # Check existing collection details
                existing = self.client.get_collection(self.collection_name)
                existing_dim = existing.config.params.vectors.size

                if existing_dim != self.embedding_dim:
                    print(
                        f"[VectorStoreManager] Dimension mismatch: "
                        f"Qdrant={existing_dim}, expected={self.embedding_dim}. "
                        f"Recreating collection..."
                    )
                    self.client.delete_collection(self.collection_name)
                    self.client.create_collection(
                        collection_name=self.collection_name,
                        vectors_config=VectorParams(size=self.embedding_dim, distance=Distance.COSINE)
                    )
                    print(f"Recreated Qdrant collection with dim={self.embedding_dim}")
                else:
                    print(f"Qdrant collection {self.collection_name} exists with correct dim={existing_dim}")

        except Exception as e:
            print(f"Error initializing Qdrant collection: {str(e)}")

    
    def chunk_text(self, text: str, chunk_size: int = 400, overlap: int = 40) -> List[str]:
        """Smaller chunks for better GPU memory management"""
        if len(text) <= chunk_size:
            return [text]
        
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + chunk_size
            
            if end < len(text):
                sentence_end = text.rfind('. ', start, end)
                if sentence_end > start:
                    end = sentence_end + 1
                else:
                    word_end = text.rfind(' ', start, end)
                    if word_end > start:
                        end = word_end
            
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            
            start = end - overlap
            
        return chunks
    
    def _load_or_create_tfidf_model(self, texts: List[str]):
        """Efficient TF-IDF model loading/creation"""
        if os.path.exists(self.model_path) and not self.is_fitted:
            try:
                with open(self.model_path, 'rb') as f:
                    model_data = pickle.load(f)
                    self.vectorizer = model_data['vectorizer']
                    self.svd = model_data['svd']
                    self.is_fitted = True
                    print("Loaded existing lightweight TF-IDF model")
                    return
            except Exception as e:
                print(f"Failed to load TF-IDF model: {e}")
        
        if not self.is_fitted and texts:
            print("Fitting TF-IDF model...")
            tfidf_matrix = self.vectorizer.fit_transform(texts)
            self.svd.fit(tfidf_matrix)
            self.is_fitted = True
            
            try:
                with open(self.model_path, 'wb') as f:
                    pickle.dump({
                        'vectorizer': self.vectorizer,
                        'svd': self.svd
                    }, f)
                print("Saved lightweight TF-IDF model")
            except Exception as e:
                print(f"Failed to save TF-IDF model: {e}")
    
    def store_text_content(self, text_pages: List[Dict]):
        """Memory-efficient content storage with batch processing"""
        points = []
        text_records = []
        all_texts = []
        
        print("Preparing text content...")
        # Collect all texts
        for page_info in text_pages:
            text_content = page_info['text']
            if text_content.strip():
                chunks = self.chunk_text(text_content)
                all_texts.extend(chunks)
        
        print(f"Processing {len(all_texts)} text chunks...")
        
        # Fit TF-IDF model if needed
        if self.embedding_method == "tfidf":
            self._load_or_create_tfidf_model(all_texts)
        
        # Process in batches to manage memory
        batch_texts = []
        batch_metadata = []
        
        for page_info in text_pages:
            page_number = page_info['page']
            image_path = page_info['path']
            text_content = page_info['text']
            
            if not text_content.strip():
                continue
            
            chunks = self.chunk_text(text_content)
            
            for i, chunk in enumerate(chunks):
                batch_texts.append(chunk)
                batch_metadata.append({
                    'page_number': page_number,
                    'source_file': image_path.split('/')[-1],
                    'chunk_index': i,
                    'text_content': chunk
                })
                
                # Process batch when it reaches batch_size
                if len(batch_texts) >= self.batch_size:
                    self._process_batch(batch_texts, batch_metadata, points, text_records)
                    batch_texts = []
                    batch_metadata = []
        
        # Process remaining batch
        if batch_texts:
            self._process_batch(batch_texts, batch_metadata, points, text_records)
        
        # Store in databases
        self._store_in_databases(points, text_records)
        
        # Final cleanup
        self._clear_gpu_cache()
    
    def _process_batch(self, texts, metadata, points, text_records):
        """Process a batch of texts efficiently"""
        try:
            # Get embeddings for batch
            if self.embedding_method == "tfidf":
                embeddings = [self.encode(text) for text in texts]
            else:
                embeddings = self.encode_batch(texts)
            
            # Create points and records
            for text, meta, embedding in zip(texts, metadata, embeddings):
                vector_id = str(uuid.uuid4())
                
                point = PointStruct(
                    id=vector_id,
                    vector=embedding,
                    payload={
                        'page_number': meta['page_number'],
                        'source_file': meta['source_file'],
                        'text_content': text,
                        'text_type': 'body_text',
                        'chunk_index': meta['chunk_index']
                    }
                )
                points.append(point)
                
                text_records.append((
                    vector_id, meta['page_number'], meta['source_file'],
                    text, 'body_text', meta['chunk_index']
                ))
                
        except Exception as e:
            print(f"Error processing batch: {e}")
    
    def _store_in_databases(self, points, text_records):
        """Store processed data in databases"""
        if points:
            try:
                self.client.upsert(
                    collection_name=self.collection_name,
                    points=points
                )
                print(f"Stored {len(points)} text chunks in Qdrant")
            except Exception as e:
                print(f"Error storing in Qdrant: {str(e)}")
        
        if text_records:
            try:
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                
                cursor.executemany('''
                INSERT INTO text_content (vector_id, page_number, source_file, text_content, text_type, chunk_index)
                VALUES (?, ?, ?, ?, ?, ?)
                ''', text_records)
                
                conn.commit()
                conn.close()
                print(f"Stored {len(text_records)} text records in SQLite")
            except Exception as e:
                print(f"Error storing in SQLite: {str(e)}")
    
    def search_similar_text(self, query: str, limit: int = 5) -> List[Dict]:
        """Memory-efficient similarity search"""
        try:
            query_embedding = self.encode(query)
            
            if all(x == 0.0 for x in query_embedding):
                print("Warning: Query embedding is zero vector")
                return self._fallback_text_search(query, limit)
            
            search_results = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_embedding,
                limit=limit,
                with_payload=True
            )
            
            results = []
            for result in search_results:
                results.append({
                    'id': result.id,
                    'score': result.score,
                    'page_number': result.payload['page_number'],
                    'source_file': result.payload['source_file'],
                    'text_content': result.payload['text_content'],
                    'text_type': result.payload['text_type'],
                    'chunk_index': result.payload['chunk_index']
                })
            
            return results
            
        except Exception as e:
            print(f"Error searching vectors: {str(e)}")
            return self._fallback_text_search(query, limit)
        finally:
            self._clear_gpu_cache()
    
    def _fallback_text_search(self, query: str, limit: int) -> List[Dict]:
        """Fallback text search using SQLite"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
            SELECT vector_id, page_number, source_file, text_content, text_type, chunk_index
            FROM text_content
            WHERE text_content LIKE ?
            ORDER BY page_number
            LIMIT ?
            ''', (f'%{query}%', limit))
            
            results = []
            for row in cursor.fetchall():
                results.append({
                    'id': row[0],
                    'score': 0.5,
                    'page_number': row[1],
                    'source_file': row[2],
                    'text_content': row[3],
                    'text_type': row[4],
                    'chunk_index': row[5]
                })
            
            conn.close()
            return results
            
        except Exception as e:
            print(f"Error in fallback search: {str(e)}")
            return []
    
    def cleanup(self):
        """Clean up resources"""
        self._clear_gpu_cache()
        if hasattr(self, 'model'):
            del self.model
        gc.collect()

# Usage example with different optimization levels
if __name__ == "__main__":
    import sys
    
    # Different modes for different hardware
    if "--gpu" in sys.argv:
        # GPU mode with memory optimization
        vector_manager = VectorStoreManager(
            use_gpu=True, 
            batch_size=8,  # Smaller batch for GPU memory
            embedding_model="gpu_optimized"
        )
    elif "--cpu" in sys.argv:
        # CPU-optimized mode
        vector_manager = VectorStoreManager(
            use_gpu=False,
            batch_size=32,  # Larger batch for CPU
            embedding_model="cpu_optimized"
        )
    else:
        # Lightweight mode (default)
        vector_manager = VectorStoreManager(
            use_gpu=False,
            batch_size=64,  # Even larger batch for lightweight mode
            embedding_model="lightweight"
        )
    
    if len(sys.argv) >= 2 and sys.argv[-1].endswith('.json'):
        with open(sys.argv[-1], 'r') as f:
            analysis_results = json.load(f)
        
        vector_manager.store_text_content(analysis_results['text_pages'])
        print("Vector storage complete!")
        
        # Cleanup
        vector_manager.cleanup()
    else:
        print("Usage: python gpu_optimized_vector_store_manager.py [--gpu|--cpu] <content_analysis.json>")
        print("  --gpu: Use GPU-optimized mode (requires CUDA)")
        print("  --cpu: Use CPU-optimized mode")
        print("  (default): Use lightweight TF-IDF mode")
