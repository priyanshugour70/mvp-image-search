"""
Search Engine Module using FAISS

This module provides fast similarity search capabilities using Facebook's FAISS
library for efficient nearest neighbor search in high-dimensional spaces.
"""

import logging
import os
import pickle
from typing import List, Dict, Tuple, Optional
import numpy as np
import faiss
import config

logger = logging.getLogger(__name__)


class SearchEngine:
    """
    FAISS-based search engine for fast similarity search
    """
    
    def __init__(self, 
                 embedding_dim: int = None,
                 use_gpu: bool = None):
        """
        Initialize the search engine
        
        Args:
            embedding_dim: Dimension of embeddings
            use_gpu: Whether to use GPU acceleration
        """
        self.embedding_dim = embedding_dim or config.CLIP_EMBEDDING_DIM
        self.use_gpu = use_gpu if use_gpu is not None else config.FAISS_USE_GPU
        
        self.index = None
        self.image_paths = []
        self.embeddings = None
        self.metadata = []  # Store SKU and product metadata
        
        logger.info(f"Initialized SearchEngine (dim={self.embedding_dim}, gpu={self.use_gpu})")
    
    def build_index(self, 
                   embeddings: np.ndarray, 
                   image_paths: List[str],
                   metadata: List[Dict] = None):
        """
        Build FAISS index from embeddings
        
        Args:
            embeddings: Array of shape (N, embedding_dim)
            image_paths: List of image paths corresponding to embeddings
            metadata: Optional list of metadata dicts (SKU info) for each image
        """
        if len(embeddings) != len(image_paths):
            raise ValueError("Number of embeddings must match number of image paths")
        
        logger.info(f"Building FAISS index for {len(embeddings)} images")
        
        # Normalize embeddings (important for cosine similarity)
        embeddings = embeddings.astype('float32')
        faiss.normalize_L2(embeddings)
        
        # Create index (using Inner Product for normalized vectors = cosine similarity)
        self.index = faiss.IndexFlatIP(self.embedding_dim)
        
        # Use GPU if requested and available
        if self.use_gpu:
            try:
                res = faiss.StandardGpuResources()
                self.index = faiss.index_cpu_to_gpu(res, 0, self.index)
                logger.info("Using GPU for FAISS index")
            except Exception as e:
                logger.warning(f"GPU not available, using CPU: {e}")
        
        # Add embeddings to index
        self.index.add(embeddings)
        
        # Store metadata
        self.image_paths = image_paths
        self.embeddings = embeddings
        self.metadata = metadata if metadata is not None else [{}] * len(image_paths)
        
        logger.info(f"FAISS index built successfully with {self.index.ntotal} vectors")
    
    def search(self, 
              query_embedding: np.ndarray, 
              top_k: int = None,
              threshold: float = None) -> List[Dict]:
        """
        Search for similar images
        
        Args:
            query_embedding: Query embedding vector
            top_k: Number of results to return
            threshold: Minimum similarity threshold
            
        Returns:
            List of result dictionaries with keys:
            - image_path: path to the image
            - similarity: similarity score (0-1)
            - rank: result rank (0-indexed)
            - sku_id: SKU identifier
            - product_name: Product name
            - category: Product category
            - price: Product price
        """
        if self.index is None or self.index.ntotal == 0:
            logger.warning("Index is empty or not built")
            return []
        
        top_k = top_k or config.DEFAULT_TOP_K
        threshold = threshold or config.SIMILARITY_THRESHOLD
        
        # Ensure query is correct shape and normalized
        query_embedding = query_embedding.astype('float32').reshape(1, -1)
        faiss.normalize_L2(query_embedding)
        
        # Search
        top_k_actual = min(top_k, self.index.ntotal)
        similarities, indices = self.index.search(query_embedding, top_k_actual)
        
        # Format results
        results = []
        for rank, (idx, similarity) in enumerate(zip(indices[0], similarities[0])):
            if similarity >= threshold:
                result = {
                    'image_path': self.image_paths[idx],
                    'similarity': float(similarity),
                    'rank': rank
                }
                
                # Add metadata (SKU info) if available
                if idx < len(self.metadata) and self.metadata[idx]:
                    result.update(self.metadata[idx])
                
                results.append(result)
        
        logger.info(f"Found {len(results)} results above threshold {threshold}")
        return results
    
    def save_index(self, save_dir: str = None):
        """
        Save FAISS index and metadata to disk
        
        Args:
            save_dir: Directory to save index (default: config.EMBEDDINGS_DIR)
        """
        save_dir = save_dir or config.EMBEDDINGS_DIR
        os.makedirs(save_dir, exist_ok=True)
        
        index_path = os.path.join(save_dir, 'faiss.index')
        metadata_path = os.path.join(save_dir, 'metadata.pkl')
        embeddings_path = os.path.join(save_dir, 'embeddings.npy')
        
        try:
            # Save FAISS index
            if self.use_gpu:
                # Move to CPU before saving
                cpu_index = faiss.index_gpu_to_cpu(self.index)
                faiss.write_index(cpu_index, index_path)
            else:
                faiss.write_index(self.index, index_path)
            
            # Save metadata
            metadata_dict = {
                'image_paths': self.image_paths,
                'embedding_dim': self.embedding_dim,
                'metadata': self.metadata
            }
            with open(metadata_path, 'wb') as f:
                pickle.dump(metadata_dict, f)
            
            # Save embeddings
            np.save(embeddings_path, self.embeddings)
            
            logger.info(f"Index saved to {save_dir}")
            
        except Exception as e:
            logger.error(f"Error saving index: {e}")
            raise
    
    def load_index(self, load_dir: str = None) -> bool:
        """
        Load FAISS index and metadata from disk
        
        Args:
            load_dir: Directory to load from (default: config.EMBEDDINGS_DIR)
            
        Returns:
            True if successful, False otherwise
        """
        load_dir = load_dir or config.EMBEDDINGS_DIR
        
        index_path = os.path.join(load_dir, 'faiss.index')
        metadata_path = os.path.join(load_dir, 'metadata.pkl')
        embeddings_path = os.path.join(load_dir, 'embeddings.npy')
        
        # Check if files exist
        if not all(os.path.exists(p) for p in [index_path, metadata_path, embeddings_path]):
            logger.warning(f"Index files not found in {load_dir}")
            return False
        
        try:
            # Load FAISS index
            self.index = faiss.read_index(index_path)
            
            # Use GPU if requested
            if self.use_gpu:
                try:
                    res = faiss.StandardGpuResources()
                    self.index = faiss.index_cpu_to_gpu(res, 0, self.index)
                    logger.info("Using GPU for FAISS index")
                except Exception as e:
                    logger.warning(f"GPU not available, using CPU: {e}")
            
            # Load metadata
            with open(metadata_path, 'rb') as f:
                metadata_dict = pickle.load(f)
            
            self.image_paths = metadata_dict['image_paths']
            self.embedding_dim = metadata_dict['embedding_dim']
            self.metadata = metadata_dict.get('metadata', [{}] * len(self.image_paths))
            
            # Load embeddings
            self.embeddings = np.load(embeddings_path)
            
            logger.info(f"Index loaded from {load_dir} ({self.index.ntotal} vectors)")
            return True
            
        except Exception as e:
            logger.error(f"Error loading index: {e}")
            return False
    
    def add_images(self, 
                  embeddings: np.ndarray, 
                  image_paths: List[str],
                  metadata: List[Dict] = None):
        """
        Add new images to existing index
        
        Args:
            embeddings: New embeddings to add
            image_paths: Corresponding image paths
            metadata: Optional metadata for new images
        """
        if self.index is None:
            # Build new index if none exists
            self.build_index(embeddings, image_paths, metadata)
            return
        
        if len(embeddings) != len(image_paths):
            raise ValueError("Number of embeddings must match number of image paths")
        
        # Normalize and add
        embeddings = embeddings.astype('float32')
        faiss.normalize_L2(embeddings)
        
        self.index.add(embeddings)
        self.image_paths.extend(image_paths)
        
        # Update metadata
        new_metadata = metadata if metadata is not None else [{}] * len(image_paths)
        self.metadata.extend(new_metadata)
        
        # Update embeddings array
        if self.embeddings is not None:
            self.embeddings = np.vstack([self.embeddings, embeddings])
        else:
            self.embeddings = embeddings
        
        logger.info(f"Added {len(embeddings)} images. Total: {self.index.ntotal}")
    
    def get_index_size(self) -> int:
        """
        Get the number of vectors in the index
        
        Returns:
            Number of indexed vectors
        """
        return self.index.ntotal if self.index else 0
    
    def clear_index(self):
        """
        Clear the index and reset
        """
        self.index = None
        self.image_paths = []
        self.embeddings = None
        self.metadata = []
        logger.info("Index cleared")
    
    def remove_images(self, indices: List[int]):
        """
        Remove images from index by indices
        Note: FAISS doesn't support efficient deletion, so we rebuild the index
        
        Args:
            indices: List of indices to remove
        """
        if self.index is None or self.index.ntotal == 0:
            return
        
        # Create mask for indices to keep
        mask = np.ones(len(self.image_paths), dtype=bool)
        mask[indices] = False
        
        # Filter embeddings and paths
        new_embeddings = self.embeddings[mask]
        new_paths = [p for i, p in enumerate(self.image_paths) if mask[i]]
        
        # Rebuild index
        self.build_index(new_embeddings, new_paths)
        
        logger.info(f"Removed {len(indices)} images from index")


# Convenience function for quick index creation
def create_index_from_paths(image_paths: List[str], 
                           feature_extractor) -> SearchEngine:
    """
    Create a search index from image paths
    
    Args:
        image_paths: List of image paths
        feature_extractor: FeatureExtractor instance
        
    Returns:
        SearchEngine with built index
    """
    from feature_extractor import FeatureExtractor
    
    if not isinstance(feature_extractor, FeatureExtractor):
        feature_extractor = FeatureExtractor()
    
    logger.info(f"Extracting features for {len(image_paths)} images")
    embeddings = feature_extractor.extract_from_path(image_paths)
    
    search_engine = SearchEngine()
    search_engine.build_index(embeddings, image_paths)
    
    return search_engine

