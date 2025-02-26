# MNIST Handwritten Digit Recognition with CNN

This project implements a Convolutional Neural Network (CNN) for recognizing handwritten digits from the MNIST dataset.

## Project Structure

```
mnist/
├── data/                # Directory for storing the MNIST dataset
├── models/              # Directory for saving trained models
├── notebooks/           # Jupyter notebooks for exploration and visualization
├── results/             # Directory for storing results and visualizations
├── src/                 # Source code
│   ├── data_loader.py   # Data loading and preprocessing
│   ├── model.py         # CNN model definition
│   ├── train.py         # Training script
│   ├── evaluate.py      # Evaluation script
│   └── visualize.py     # Visualization utilities
└── requirements.txt     # Project dependencies
```

## Setup

1. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

2. Run the training script:
   ```
   python src/train.py
   ```

3. Evaluate the model:
   ```
   python src/evaluate.py
   ```

4. Generate visualizations:
   ```
   python src/visualize.py
   ```

## Results

The CNN model achieves high accuracy on the MNIST test set. Detailed results and visualizations can be found in the `results/` directory.
