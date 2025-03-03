from fastapi import FastAPI, UploadFile, File, HTTPException, Form, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from typing import Dict, Any, List, Optional
import json
import cv2
import numpy as np
from pydantic import BaseModel
import base64
import io
from PIL import Image

from app.image_processing.core import ImageProcessor
from app.image_processing.utils import (
    encode_image_to_base64, decode_base64_to_image,
    format_metrics, get_available_operations,
    optimize_for_large_images, restore_original_size
)

app = FastAPI(title="Image Enhancement API", 
              description="API for enhancing images using spatial domain techniques")

# Disable CORS. Do not remove this for full-stack development.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Pydantic models for request/response validation
class EnhancementRequest(BaseModel):
    image: str  # Base64 encoded image
    operation: str
    parameters: Optional[Dict[str, Any]] = None
    
class EnhancementResponse(BaseModel):
    original_image: str  # Base64 encoded original image
    enhanced_image: str  # Base64 encoded enhanced image
    metrics: Dict[str, str]  # Formatted metrics

class OperationInfo(BaseModel):
    id: str
    name: str
    description: str
    parameters: List[Dict[str, Any]]

@app.get("/healthz")
async def healthz():
    return {"status": "ok"}

@app.get("/api/operations", response_model=List[OperationInfo])
async def get_operations():
    """Get list of available image enhancement operations."""
    return get_available_operations()

@app.post("/api/enhance", response_model=EnhancementResponse)
async def enhance_image(request: EnhancementRequest):
    """
    Enhance image using the specified operation and parameters.
    
    Args:
        request: Enhancement request containing base64 encoded image, operation, and parameters
        
    Returns:
        Enhanced image and metrics
    """
    try:
        # Decode base64 image
        image = decode_base64_to_image(request.image)
        
        # Optimize for large images
        original_shape = image.shape[:2]
        optimized_image, scale_factor = optimize_for_large_images(image)
        
        # Process image
        processed, metrics = ImageProcessor.process_image(
            optimized_image, 
            request.operation, 
            request.parameters
        )
        
        # Restore original size if needed
        if scale_factor < 1.0:
            processed = restore_original_size(processed, original_shape)
        
        # Format metrics for display
        formatted_metrics = format_metrics(metrics)
        
        # Encode images to base64
        original_base64 = encode_image_to_base64(image)
        processed_base64 = encode_image_to_base64(processed)
        
        return {
            "original_image": original_base64,
            "enhanced_image": processed_base64,
            "metrics": formatted_metrics
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/api/upload", response_model=Dict[str, str])
async def upload_image(file: UploadFile = File(...)):
    """
    Upload an image file and return it as base64.
    
    Args:
        file: Uploaded image file
        
    Returns:
        Base64 encoded image
    """
    try:
        # Read file content
        content = await file.read()
        
        # Load image
        image = ImageProcessor.load_image(content)
        
        # Encode image to base64
        base64_image = encode_image_to_base64(image)
        
        return {"image": base64_image}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/api/batch-enhance")
async def batch_enhance(
    image: str = Body(...),  # Base64 encoded image
    operations: List[Dict[str, Any]] = Body(...)  # List of operations to apply
):
    """
    Apply multiple enhancement operations to an image in sequence.
    
    Args:
        image: Base64 encoded image
        operations: List of operations to apply, each with operation name and parameters
        
    Returns:
        List of enhanced images and metrics
    """
    try:
        # Decode base64 image
        current_image = decode_base64_to_image(image)
        original_image = current_image.copy()
        
        results = []
        
        # Apply each operation in sequence
        for op in operations:
            operation = op.get("operation")
            parameters = op.get("parameters", {})
            
            # Process image
            processed, metrics = ImageProcessor.process_image(
                current_image, 
                operation, 
                parameters
            )
            
            # Update current image for next operation
            current_image = processed.copy()
            
            # Format metrics for display
            formatted_metrics = format_metrics(metrics)
            
            # Encode processed image to base64
            processed_base64 = encode_image_to_base64(processed)
            
            results.append({
                "operation": operation,
                "enhanced_image": processed_base64,
                "metrics": formatted_metrics
            })
        
        # Encode original image to base64
        original_base64 = encode_image_to_base64(original_image)
        
        return {
            "original_image": original_base64,
            "results": results
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
