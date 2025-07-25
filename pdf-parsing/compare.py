from dotenv import load_dotenv
import os
import time
import pickle
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, List, Any
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import google.generativeai as genai

load_dotenv()

# Only need Gemini API key now since we're using HF models for both
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

class DeepSeekEmbeddings:
    """Using Hugging Face model instead of DeepSeek API"""
    def __init__(self):
        # Using a different HF model to simulate DeepSeek
        # You can replace this with any other embedding model from HuggingFace
        print("Loading DeepSeek (HF) model...")
        self.model = SentenceTransformer('all-mpnet-base-v2')
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        embeddings = self.model.encode(texts)
        return embeddings.tolist()

@dataclass
class EmbeddingResult:
    model_name: str
    embeddings: List[List[float]]
    processing_time: float
    embedding_dimension: int
    cost_estimate: float = 0.0

class EmbeddingComparator:
    def __init__(self):
        self.results: Dict[str, EmbeddingResult] = {}
    
    def test_embedding_model(self, model_name: str, embedding_model, texts: List[str]) -> EmbeddingResult:
        start_time = time.time()
        embeddings = embedding_model.embed_documents(texts)
        end_time = time.time()
        processing_time = end_time - start_time
        
        result = EmbeddingResult(
            model_name=model_name,
            embeddings=embeddings,
            processing_time=processing_time,
            embedding_dimension=len(embeddings[0]) if embeddings else 0
        )
        
        self.results[model_name] = result
        return result

def setup_embedding_models():
    """Setup all embedding models"""
    print("Loading Hugging Face model...")
    hf_model = SentenceTransformer('all-MiniLM-L6-v2')
    
    print("Setting up DeepSeek (HF) model...")
    deepseek_model = DeepSeekEmbeddings()
    
    # Only setup Gemini if API key is available
    gemini_model = None
    if GEMINI_API_KEY:
        print("Configuring Gemini...")
        genai.configure(api_key=GEMINI_API_KEY)
        
        class GeminiEmbeddings:
            def embed_documents(self, texts: List[str]) -> List[List[float]]:
                embeddings = []
                for text in texts:
                    try:
                        result = genai.embed_content(
                            model="models/embedding-001",
                            content=text
                        )
                        embeddings.append(result['embedding'])
                    except Exception as e:
                        print(f"Gemini API error: {e}")
                        # Return zero vector as fallback
                        embeddings.append([0.0] * 768)
                return embeddings
        
        gemini_model = GeminiEmbeddings()
    else:
        print("No Gemini API key found, skipping Gemini model...")
    
    models = {
        'huggingface': hf_model,
        'deepseek_hf': deepseek_model,
    }
    
    if gemini_model:
        models['gemini'] = gemini_model
    
    return models

def run_comprehensive_comparison(llamaparse_results: List[str]):
    """Run comparison across all available embedding models"""
    comparator = EmbeddingComparator()
    models = setup_embedding_models()
    
    for model_name, model in models.items():
        print(f"Testing {model_name}...")
        
        try:
            if model_name == 'huggingface':
                start_time = time.time()
                embeddings = model.encode(llamaparse_results)
                end_time = time.time()
                
                result = EmbeddingResult(
                    model_name=model_name,
                    embeddings=embeddings.tolist(),
                    processing_time=end_time - start_time,
                    embedding_dimension=len(embeddings[0])
                )
            else:
                result = comparator.test_embedding_model(model_name, model, llamaparse_results)
            
            comparator.results[model_name] = result
            print(f"  Completed in {result.processing_time:.2f}s")
            
        except Exception as e:
            print(f"  Error with {model_name}: {e}")
            continue
    
    return comparator

def analyze_results(comparator: EmbeddingComparator):
    """Analyze and compare the results"""
    if not comparator.results:
        print("No results to analyze!")
        return
        
    results_df = pd.DataFrame([
        {
            'Model': name,
            'Processing Time (s)': result.processing_time,
            'Embedding Dimension': result.embedding_dimension,
            'Avg Embedding Magnitude': np.mean([np.linalg.norm(emb) for emb in result.embeddings])
        }
        for name, result in comparator.results.items()
    ])
    
    print("\nPerformance Comparison:")
    print(results_df.to_string(index=False))
    
    print("\nSemantic Similarity Analysis:")
    models = list(comparator.results.keys())
    for i, model1 in enumerate(models):
        for j, model2 in enumerate(models[i+1:], i+1):
            emb1 = np.array(comparator.results[model1].embeddings)
            emb2 = np.array(comparator.results[model2].embeddings)
            if emb1.shape[1] != emb2.shape[1]:
                print(f"Skipping {model1} vs {model2}: incompatible dimensions ({emb1.shape[1]} vs {emb2.shape[1]})")
                continue
            similarities = [cosine_similarity([e1], [e2])[0][0] 
                            for e1, e2 in zip(emb1, emb2)]
            avg_similarity = np.mean(similarities)

            print(f"{model1} vs {model2}: {avg_similarity:.4f}")

def quality_assessment(llamaparse_results: List[str], comparator: EmbeddingComparator):
    def clustering_quality_test(embeddings, texts):
        similarities = cosine_similarity(embeddings)
        return np.mean(similarities)
    
    def retrieval_test(embeddings, texts, query_idx=0):
        query_embedding = embeddings[query_idx]
        similarities = cosine_similarity([query_embedding], embeddings)[0]
        # Return top-k most similar (excluding the query itself)
        top_indices = np.argsort(similarities)[::-1][1:6]  # top 5
        return top_indices, similarities[top_indices]
    
    quality_results = {}
    for model_name, result in comparator.results.items():
        embeddings = np.array(result.embeddings)
        
        cluster_score = clustering_quality_test(embeddings, llamaparse_results)
        top_indices, top_scores = retrieval_test(embeddings, llamaparse_results)
        
        quality_results[model_name] = {
            'clustering_score': cluster_score,
            'retrieval_scores': top_scores.tolist(),
            'avg_retrieval_score': np.mean(top_scores)
        }
    
    return quality_results

def main():
    print("Starting Embedding Models Comparison...")
    llamaparse_results = [
        "Machine learning is a subset of artificial intelligence that focuses on algorithms.",
        "Deep learning uses neural networks with multiple layers to learn complex patterns.",
        "Natural language processing enables computers to understand and generate human language.",
        "Computer vision allows machines to interpret and analyze visual information.",
        "Reinforcement learning involves agents learning through interaction with an environment.",
        "Data preprocessing is crucial for improving model performance and accuracy.",
        "Feature engineering involves selecting and transforming variables for machine learning models.",
        "Cross-validation helps assess how well a model generalizes to unseen data."
    ]
    
    print(f"Testing with {len(llamaparse_results)} text chunks...")
    comparator = run_comprehensive_comparison(llamaparse_results)
    
    if comparator.results:
        analyze_results(comparator)
        quality_results = quality_assessment(llamaparse_results, comparator)
        
        print("\nQuality Assessment:")
        for model, metrics in quality_results.items():
            print(f"{model}:")
            print(f"  Clustering Score: {metrics['clustering_score']:.4f}")
            print(f"  Avg Retrieval Score: {metrics['avg_retrieval_score']:.4f}")
        
        print("\nSaving results...")
        os.makedirs('results', exist_ok=True)
        with open('results/embedding_comparison_results.pkl', 'wb') as f:
            pickle.dump({
                'comparator': comparator,
                'quality_results': quality_results
            }, f)
        
        print("Comparison completed successfully!")
    else:
        print("No models were successfully tested!")

if __name__ == "__main__":
    main()