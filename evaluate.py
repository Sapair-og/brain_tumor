import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import load_img, img_to_array          #type: ignore
from config import IMG_SIZE, final_model_path, SAVE_WEIGHTS

def evaluate_model(model,testgenrator):
    """Evaluate model on test data and save if configured"""
    test_loss, test_accuracy = model.evaluate(testgenrator)
    print(f"\nTest Accuracy: {test_accuracy*100:.2f}%")
    
    # Save model weights if configured
    if SAVE_WEIGHTS:
        try:
            print(f"Saving model to {final_model_path}...")
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(final_model_path), exist_ok=True)
            model.save(final_model_path)
            if os.path.exists(final_model_path):
                print(f"Model saved successfully. File size: {os.path.getsize(final_model_path) / (1024*1024):.2f} MB")
            else:
                print(f"Warning: Model file not found after save attempt.")
        except Exception as e:
            print(f"Error saving model: {e}")
    else:
        print("Model saving is disabled in configuration.")
    
    return test_loss, test_accuracy