#!/usr/bin/env python3
"""
Script to download test images from open source datasets for the Image Enhancement System.
"""

import os
import sys
import urllib.request
import ssl
import shutil
from pathlib import Path

# Add the src directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.helpers import create_directory_if_not_exists

# Define the output directory
OUTPUT_DIR = os.path.join("data", "input", "samples")

# Create the output directory if it doesn't exist
create_directory_if_not_exists(OUTPUT_DIR)

# Define image sources
# Format: (url, filename, description)
IMAGE_SOURCES = [
    # Standard test images
    (
        "https://sipi.usc.edu/database/misc/4.1.01.tiff",
        "lena.tiff",
        "Standard test image (Lena) - Good for general testing"
    ),
    (
        "https://sipi.usc.edu/database/misc/4.1.02.tiff",
        "mandrill.tiff",
        "Mandrill/Baboon - High frequency details, good for edge detection and sharpening"
    ),
    (
        "https://sipi.usc.edu/database/misc/4.1.03.tiff",
        "peppers.tiff",
        "Peppers - Good for color processing and contrast adjustment"
    ),
    (
        "https://sipi.usc.edu/database/misc/4.1.05.tiff",
        "house.tiff",
        "House - Good for edge detection and structural details"
    ),
    (
        "https://sipi.usc.edu/database/misc/4.1.06.tiff",
        "jellybeans.tiff",
        "Jelly Beans - Good for color processing and pseudocolor tests"
    ),
    (
        "https://sipi.usc.edu/database/misc/4.1.07.tiff",
        "airplane.tiff",
        "Airplane - Good for edge detection and noise reduction"
    ),
    
    # Noisy images
    (
        "https://homepages.cae.wisc.edu/~ece533/images/zelda.png",
        "zelda.png",
        "Zelda - Good for noise reduction testing"
    ),
    (
        "https://homepages.cae.wisc.edu/~ece533/images/boat.png",
        "boat.png",
        "Boat - Good for noise reduction and edge detection"
    ),
    
    # Low contrast images
    (
        "https://homepages.cae.wisc.edu/~ece533/images/monarch.png",
        "monarch.png",
        "Monarch - Good for contrast adjustment and color processing"
    ),
    (
        "https://homepages.cae.wisc.edu/~ece533/images/pool.png",
        "pool.png",
        "Pool - Good for contrast enhancement and edge detection"
    ),
    
    # High resolution images for interpolation/magnification
    (
        "https://sipi.usc.edu/database/misc/4.2.03.tiff",
        "resolution_chart.tiff",
        "Resolution Chart - Good for interpolation and magnification testing"
    ),
    
    # Grayscale images
    (
        "https://sipi.usc.edu/database/misc/5.1.09.tiff",
        "moon.tiff",
        "Moon Surface - Good for grayscale processing and pseudocolor"
    ),
    (
        "https://sipi.usc.edu/database/misc/5.1.10.tiff",
        "aerial.tiff",
        "Aerial - Good for grayscale processing and edge enhancement"
    ),
    
    # Medical images (good for pseudocolor processing)
    (
        "https://sipi.usc.edu/database/misc/7.1.01.tiff",
        "mri.tiff",
        "MRI - Good for pseudocolor processing and medical image enhancement"
    ),
    (
        "https://sipi.usc.edu/database/misc/7.1.03.tiff",
        "xray.tiff",
        "X-ray - Good for pseudocolor processing and contrast enhancement"
    ),
    
    # Additional images from Berkeley Segmentation Dataset
    (
        "https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/bsds/BSDS300/images/test/100080.jpg",
        "starfish.jpg",
        "Starfish - Good for edge detection and color processing"
    ),
    (
        "https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/bsds/BSDS300/images/test/101087.jpg",
        "butterfly.jpg",
        "Butterfly - Good for detail enhancement and color processing"
    ),
    (
        "https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/bsds/BSDS300/images/test/108082.jpg",
        "surfer.jpg",
        "Surfer - Good for contrast adjustment and edge detection"
    )
]

def download_image(url, output_path, description):
    """
    Download an image from a URL.
    
    Args:
        url: URL of the image
        output_path: Path to save the image
        description: Description of the image
    
    Returns:
        True if successful, False otherwise
    """
    try:
        # Create SSL context to handle HTTPS
        ssl_context = ssl.create_default_context()
        ssl_context.check_hostname = False
        ssl_context.verify_mode = ssl.CERT_NONE
        
        # Download the image
        with urllib.request.urlopen(url, context=ssl_context) as response:
            with open(output_path, 'wb') as out_file:
                shutil.copyfileobj(response, out_file)
        
        print(f"Downloaded: {os.path.basename(output_path)} - {description}")
        return True
    except Exception as e:
        print(f"Error downloading {url}: {str(e)}")
        return False

def main():
    """Main function to download test images."""
    print(f"Downloading test images to: {os.path.abspath(OUTPUT_DIR)}")
    
    # Create a metadata file
    metadata_path = os.path.join(OUTPUT_DIR, "metadata.txt")
    with open(metadata_path, 'w') as metadata_file:
        metadata_file.write("Test Images for Image Enhancement System\n")
        metadata_file.write("==========================================\n\n")
        
        # Download each image
        success_count = 0
        for url, filename, description in IMAGE_SOURCES:
            output_path = os.path.join(OUTPUT_DIR, filename)
            
            # Download the image
            if download_image(url, output_path, description):
                success_count += 1
                
                # Add to metadata
                metadata_file.write(f"Filename: {filename}\n")
                metadata_file.write(f"Source: {url}\n")
                metadata_file.write(f"Description: {description}\n")
                metadata_file.write("\n")
        
        # Write summary
        metadata_file.write(f"\nSummary: Downloaded {success_count} of {len(IMAGE_SOURCES)} images.\n")
    
    print(f"\nDownloaded {success_count} of {len(IMAGE_SOURCES)} images.")
    print(f"Metadata saved to: {metadata_path}")

if __name__ == "__main__":
    main()
