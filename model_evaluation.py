# ============================================================
# Team Member: Omar
# Component: Model Evaluation & Metrics Analytics
# ============================================================
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf

def evaluate_model(model_path, test_data_dir):
    """
    Omar: This function will compute the exact Accuracy, Precision, Recall, and F1 Score
    for the Hybrid Vision Transformer across all 23 DermNet classes.
    """
    print(f"Loading final weights from: {model_path}")
    # model = tf.keras.models.load_model(model_path)
    
    print("Evaluating over Test Set...")
    # Add your data generator prediction logic here!
    
    print("Generating Confusion Matrix...")
    # Plot the matrix using seaborn heatmap 
    
    # Plot the ROC Curves using sklearn
    
if __name__ == "__main__":
    # evaluate_model('../models/final_hybrid_transformer.keras', '../data/test')
    pass