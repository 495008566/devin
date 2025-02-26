# Image Enhancement System

A spatial domain image enhancement system using Python and OpenCV.

## Features

- Grayscale conversion and contrast adjustment
- Noise reduction using mean filter
- Edge enhancement and sharpening using Sobel operator
- Image filtering
- Interpolation and magnification
- Pseudocolor processing
- Before/after comparison view
- Image statistics and metrics
- MySQL database for storing metadata and results

## Project Structure

```
image_enhancement_system/
├── data/
│   ├── input/       # Input images
│   └── output/      # Enhanced images
├── docs/            # Documentation
├── src/             # Source code
│   ├── core/        # Core image processing functions
│   ├── database/    # Database operations
│   ├── ui/          # User interface
│   └── utils/       # Utility functions
└── tests/           # Test cases
```

## Requirements

See `requirements.txt` for a list of dependencies.

## Setup and Installation

1. Clone the repository
2. Install dependencies: `pip install -r requirements.txt`
3. Configure MySQL database (see docs/database_setup.md)
4. Run the application: `python src/main.py`

## License

MIT
