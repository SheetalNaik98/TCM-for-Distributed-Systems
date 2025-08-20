"""Vector store implementation for similarity search"""

from typing import Any, Dict, List, Optional
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import os


class VectorStore:
    """Simple vector store with TF-IDF and optional embeddings"""
    
    def __init__(self, use_embeddings: bool = False):
        self.entries = []
        self.vectors = []
        self.use_embeddings = use_embeddings and self._check_embeddings_available()
        
        if self.use_embeddings:
            self.model = SentenceTransformer('all-MiniLM-L6-v2')
        else:
            self.vectorizer = TfidfVectorizer(max_features=768)
            self.fitted = False
            
    def _check_embeddings_available(self) -> bool:
        """Check if we can use embeddings (requires model download)"""
        try:
            # Try to load the model
            _ = SentenceTransformer('all-MiniLM-L6-v2')
            return True
        except:
            return False
            
    def add(self, entry: Any) -> str:
        """Add entry to store and return ID"""
        self.entries.append(entry)
        
        # Generate vector
        if self.use_embeddings:
            vector = self.model.encode(entry.content)
        else:
            # For TF-IDF, we need to refit when adding
            self.fitted = False
            vector = None  # Will be computed on search
            
        self.vectors.append(vector)
        entry.vector = vector
        
        return entry.id
    
    def search(self, query: str, k: int = 5) -> List[Dict]:
        """Search for similar entries"""
        if not self.entries:
            return []
            
        if self.use_embeddings:
            query_vector = self.model.encode(query)
            
            # Compute similarities
            similarities = []
            for i, vector in enumerate(self.vectors):
                if vector is not None:
                    sim = cosine_similarity([query_vector], [vector])[0][0]
                    similarities.append((i, sim))
                    
        else:
            # TF-IDF search
            if not self.fitted and len(self.entries) > 0:
                # Fit vectorizer on all entries
                texts = [e.content for e in self.entries]
                self.vectorizer.fit(texts)
                self.fitted = True
                
                # Compute all vectors
                all_vectors = self.vectorizer.transform(texts)
                for i, vec in enumerate(all_vectors):
                    self.vectors[i] = vec
                    
            # Transform query
            query_vector = self.vectorizer.transform([query])
            
            # Compute similarities
            similarities = []
            for i, vector in enumerate(self.vectors):
                if vector is not None:
                    sim = cosine_similarity(query_vector, vector)[0][0]
                    similarities.append((i, sim))
                    
        # Sort by similarity
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        # Return top k
        results = []
        for idx, sim in similarities[:k]:
            entry = self.entries[idx]
            results.append({
                "id": entry.id,
                "content": entry.content,
                "topic": entry.topic,
                "owner": entry.owner,
                "metadata": entry.metadata,
                "similarity": sim
            })
            
        return results
    
    def search_by_id(self, entry_id: str) -> List[Dict]:
        """Search for specific entry by ID"""
        for entry in self.entries:
            if entry.id == entry_id:
                return [{
                    "id": entry.id,
                    "content": entry.content,
                    "topic": entry.topic,
                    "owner": entry.owner,
                    "metadata": entry.metadata
                }]
        return []
