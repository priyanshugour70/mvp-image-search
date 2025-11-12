"""
Feature Extraction Module using CLIP

This module extracts robust, viewpoint-invariant embeddings from images
using OpenAI's CLIP model.
"""

import logging
from typing import List, Union
import numpy as np
from PIL import Image
import torch
from transformers import CLIPProcessor, CLIPModel
import config

logger = logging.getLogger(__name__)


class FeatureExtractor:
    """
    CLIP-based feature extractor for creating image embeddings
    """
    
    def __init__(self, model_name: str = None):
        """
        Initialize the feature extractor
        
        Args:
            model_name: CLIP model name from Hugging Face
        """
        self.model_name = model_name or config.CLIP_MODEL_NAME
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        logger.info(f"Loading CLIP model: {self.model_name}")
        logger.info(f"Using device: {self.device}")
        
        try:
            self.model = CLIPModel.from_pretrained(self.model_name).to(self.device)
            self.processor = CLIPProcessor.from_pretrained(self.model_name)
            self.model.eval()  # Set to evaluation mode
            logger.info("CLIP model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading CLIP model: {e}")
            raise
    
    def extract(self, image: Union[Image.Image, List[Image.Image]]) -> np.ndarray:
        """
        Extract feature embeddings from image(s)
        
        Args:
            image: Single PIL Image or list of PIL Images
            
        Returns:
            Numpy array of shape (embedding_dim,) for single image
            or (num_images, embedding_dim) for multiple images
        """
        # Handle single image vs batch
        is_single = not isinstance(image, list)
        images = [image] if is_single else image
        
        try:
            # Preprocess images
            inputs = self.processor(
                images=images,
                return_tensors="pt",
                padding=True
            ).to(self.device)
            
            # Extract features
            with torch.no_grad():
                image_features = self.model.get_image_features(**inputs)
                
                # Normalize embeddings (important for cosine similarity)
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            
            # Convert to numpy
            embeddings = image_features.cpu().numpy()
            
            # Return single embedding if single image
            return embeddings[0] if is_single else embeddings
            
        except Exception as e:
            logger.error(f"Error extracting features: {e}")
            raise
    
    def extract_from_path(self, image_path: Union[str, List[str]]) -> np.ndarray:
        """
        Extract features from image file path(s)
        
        Args:
            image_path: Single image path or list of paths
            
        Returns:
            Feature embedding(s)
        """
        from utils import load_image
        
        # Handle single path vs batch
        is_single = isinstance(image_path, str)
        paths = [image_path] if is_single else image_path
        
        # Load images
        images = [load_image(path) for path in paths]
        
        return self.extract(images)
    
    def extract_batch(self, 
                     images: List[Image.Image], 
                     batch_size: int = None) -> np.ndarray:
        """
        Extract features from a large batch of images with batching
        
        Args:
            images: List of PIL Images
            batch_size: Batch size for processing
            
        Returns:
            Numpy array of shape (num_images, embedding_dim)
        """
        batch_size = batch_size or config.BATCH_SIZE
        all_embeddings = []
        
        for i in range(0, len(images), batch_size):
            batch = images[i:i + batch_size]
            logger.info(f"Processing batch {i//batch_size + 1}/{(len(images)-1)//batch_size + 1}")
            embeddings = self.extract(batch)
            all_embeddings.append(embeddings)
        
        return np.vstack(all_embeddings)
    
    def compute_similarity(self, 
                          embedding1: np.ndarray, 
                          embedding2: np.ndarray) -> float:
        """
        Compute cosine similarity between two embeddings
        
        Args:
            embedding1: First embedding vector
            embedding2: Second embedding vector
            
        Returns:
            Similarity score (0-1, higher is more similar)
        """
        # Embeddings are already normalized, so dot product = cosine similarity
        similarity = np.dot(embedding1, embedding2)
        return float(similarity)
    
    def compute_similarity_matrix(self, 
                                 embeddings1: np.ndarray, 
                                 embeddings2: np.ndarray) -> np.ndarray:
        """
        Compute pairwise similarity matrix between two sets of embeddings
        
        Args:
            embeddings1: Array of shape (N, embedding_dim)
            embeddings2: Array of shape (M, embedding_dim)
            
        Returns:
            Similarity matrix of shape (N, M)
        """
        # Matrix multiplication (embeddings are normalized)
        similarity_matrix = np.dot(embeddings1, embeddings2.T)
        return similarity_matrix
    
    def find_most_similar(self, 
                         query_embedding: np.ndarray, 
                         database_embeddings: np.ndarray,
                         top_k: int = 5) -> tuple:
        """
        Find most similar embeddings from database
        
        Args:
            query_embedding: Query embedding vector
            database_embeddings: Database embeddings (N, embedding_dim)
            top_k: Number of top results to return
            
        Returns:
            Tuple of (indices, similarities)
        """
        # Compute similarities
        similarities = np.dot(database_embeddings, query_embedding)
        
        # Get top-k indices
        top_k = min(top_k, len(similarities))
        top_indices = np.argsort(similarities)[::-1][:top_k]
        top_similarities = similarities[top_indices]
        
        return top_indices, top_similarities
    
    def get_embedding_dimension(self) -> int:
        """
        Get the dimensionality of the embeddings
        
        Returns:
            Embedding dimension
        """
        return config.CLIP_EMBEDDING_DIM


# Convenience function for quick feature extraction
def extract_features(image_path: str) -> np.ndarray:
    """
    Quick function to extract features from an image
    
    Args:
        image_path: Path to image file
        
    Returns:
        Feature embedding vector
    """
    extractor = FeatureExtractor()
    return extractor.extract_from_path(image_path)

