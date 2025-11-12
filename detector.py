"""
Object Detection Module using YOLOv8

This module provides object detection capabilities to localize objects
in images before feature extraction and matching.
"""

import logging
from typing import List, Dict, Tuple, Optional
import numpy as np
from PIL import Image
from ultralytics import YOLO
import config
import utils

logger = logging.getLogger(__name__)


class ObjectDetector:
    """
    YOLOv8-based object detector for localizing objects in images
    """
    
    def __init__(self, model_name: str = None, confidence: float = None):
        """
        Initialize the object detector
        
        Args:
            model_name: YOLOv8 model name (e.g., 'yolov8n.pt')
            confidence: Detection confidence threshold
        """
        self.model_name = model_name or config.YOLO_MODEL
        self.confidence = confidence or config.YOLO_CONFIDENCE
        
        logger.info(f"Loading YOLO model: {self.model_name}")
        try:
            self.model = YOLO(self.model_name)
            logger.info("YOLO model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading YOLO model: {e}")
            raise
    
    def detect(self, 
               image: Image.Image, 
               classes: Optional[List[int]] = None,
               confidence: Optional[float] = None) -> List[Dict]:
        """
        Detect objects in an image
        
        Args:
            image: PIL Image object
            classes: Optional list of class IDs to detect (None = all classes)
            confidence: Override default confidence threshold
            
        Returns:
            List of detection dictionaries with keys:
            - bbox: (x1, y1, x2, y2) bounding box coordinates
            - confidence: detection confidence score
            - class_id: COCO class ID
            - class_name: human-readable class name
            - cropped_image: PIL Image of cropped object region
        """
        conf_threshold = confidence or self.confidence
        
        try:
            # Run inference
            results = self.model.predict(
                image,
                conf=conf_threshold,
                iou=config.YOLO_IOU,
                classes=classes,
                verbose=False
            )
            
            detections = []
            
            if len(results) == 0:
                logger.warning("No detections found")
                return detections
            
            result = results[0]
            
            # Process each detection
            for box in result.boxes:
                # Extract bounding box coordinates
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                bbox = (int(x1), int(y1), int(x2), int(y2))
                
                # Add padding to bbox for better context
                img_width, img_height = image.size
                padded_bbox = utils.pad_bbox(bbox, padding=0.1, 
                                            img_width=img_width, 
                                            img_height=img_height)
                
                # Crop the object region
                cropped_image = utils.crop_image(image, padded_bbox)
                
                # Extract metadata
                class_id = int(box.cls[0].cpu().numpy())
                confidence_score = float(box.conf[0].cpu().numpy())
                class_name = config.COCO_CLASSES[class_id] if class_id < len(config.COCO_CLASSES) else f"class_{class_id}"
                
                detection = {
                    'bbox': bbox,
                    'padded_bbox': padded_bbox,
                    'confidence': confidence_score,
                    'class_id': class_id,
                    'class_name': class_name,
                    'cropped_image': cropped_image
                }
                
                detections.append(detection)
            
            logger.info(f"Found {len(detections)} objects")
            return detections
            
        except Exception as e:
            logger.error(f"Error during object detection: {e}")
            raise
    
    def detect_from_path(self, image_path: str, **kwargs) -> List[Dict]:
        """
        Detect objects from an image file path
        
        Args:
            image_path: Path to image file
            **kwargs: Additional arguments passed to detect()
            
        Returns:
            List of detection dictionaries
        """
        image = utils.load_image(image_path, max_size=config.MAX_IMAGE_DIMENSION)
        return self.detect(image, **kwargs)
    
    def get_best_detection(self, 
                          detections: List[Dict], 
                          preferred_classes: Optional[List[int]] = None) -> Optional[Dict]:
        """
        Get the best detection based on confidence score
        
        Args:
            detections: List of detection dictionaries
            preferred_classes: Optional list of preferred class IDs
            
        Returns:
            Best detection dictionary or None
        """
        if not detections:
            return None
        
        # Filter by preferred classes if specified
        if preferred_classes:
            filtered = [d for d in detections if d['class_id'] in preferred_classes]
            if filtered:
                detections = filtered
        
        # Return detection with highest confidence
        return max(detections, key=lambda x: x['confidence'])
    
    def visualize_detections(self, 
                           image: Image.Image, 
                           detections: List[Dict]) -> Image.Image:
        """
        Draw bounding boxes on the image (for debugging/visualization)
        
        Args:
            image: Original PIL Image
            detections: List of detection dictionaries
            
        Returns:
            Image with bounding boxes drawn
        """
        from PIL import ImageDraw, ImageFont
        
        # Create a copy to draw on
        img_copy = image.copy()
        draw = ImageDraw.Draw(img_copy)
        
        for det in detections:
            bbox = det['bbox']
            class_name = det['class_name']
            confidence = det['confidence']
            
            # Draw bounding box
            draw.rectangle(bbox, outline='red', width=3)
            
            # Draw label
            label = f"{class_name}: {confidence:.2f}"
            draw.text((bbox[0], bbox[1] - 10), label, fill='red')
        
        return img_copy
    
    def batch_detect(self, images: List[Image.Image], **kwargs) -> List[List[Dict]]:
        """
        Detect objects in multiple images (batch processing)
        
        Args:
            images: List of PIL Image objects
            **kwargs: Additional arguments passed to detect()
            
        Returns:
            List of detection lists (one per image)
        """
        all_detections = []
        
        for i, image in enumerate(images):
            logger.info(f"Processing image {i+1}/{len(images)}")
            detections = self.detect(image, **kwargs)
            all_detections.append(detections)
        
        return all_detections


# Convenience function for quick detection
def detect_objects(image_path: str, **kwargs) -> List[Dict]:
    """
    Quick function to detect objects from an image path
    
    Args:
        image_path: Path to image file
        **kwargs: Additional arguments for detection
        
    Returns:
        List of detection dictionaries
    """
    detector = ObjectDetector()
    return detector.detect_from_path(image_path, **kwargs)

