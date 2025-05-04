import tensorflow as tf
import torch
import os
import argparse
from data_loader import get_data_generators, check_data_balance
from model_builder import build_model
from callbacks import get_callbacks
from train import train_model_phase1, fine_tune_model
from evaluate import evaluate_model
from visualize import plot_training_curves, visualize_predictions, generate_confusion_matrix, visualize_class_activation_maps, plot_sample_images_per_class
from config import USE_SAVED_WEIGHTS, INITIAL_EPOCHS, FINE_TUNE_EPOCHS, final_model_path, data_dir, CLASSES

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Brain Tumor Classification Model')
    parser.add_argument('--use-saved-weights', action='store_true', 
                        help='Use saved weights instead of training from scratch')
    parser.add_argument('--no-save-weights', action='store_true',
                        help='Do not save weights after training')
    parser.add_argument('--skip-training', action='store_true',
                        help='Skip training completely, only evaluate model')
    parser.add_argument('--test-image', type=str,
                        help='Path to an image to test after loading model')
    parser.add_argument('--visualize-only', action='store_true',
                        help='Only visualize sample images from the dataset')
    parser.add_argument('--check-balance', action='store_true',
                        help='Check class balance in the dataset splits')
    args = parser.parse_args()
    return args

def prompt_for_image_test():
    """Ask user if they want to test on an image"""
    while True:
        response = input("\nDo you want to test the model on an image? (yes/no): ").strip().lower()
        if response in ['yes', 'y']:
            image_path = input("Enter the path to the brain MRI image: ").strip()
            if os.path.exists(image_path):
                return image_path
            else:
                print(f"Error: File '{image_path}' not found. Please try again.")
        elif response in ['no', 'n']:
            return None
        else:
            print("Please answer with 'yes' or 'no'.")

def display_prediction(image_path):
    """Display prediction for a single image"""
    from tensorflow.keras.preprocessing.image import load_img, img_to_array  #type:ignore
    from tensorflow.keras.models import load_model  #type: ignore
    import numpy as np
    import matplotlib.pyplot as plt
    
    try:
        # Load the model
        if not os.path.exists(final_model_path):
            print(f"Error: Model file not found at {final_model_path}")
            return
        
        model = load_model(final_model_path)
        
        # Load and preprocess the image
        img = load_img(image_path, target_size=(224, 224))
        img_array = img_to_array(img)
        img_array = img_array / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        # Make prediction
        prediction = model.predict(img_array)[0]
        predicted_class_idx = np.argmax(prediction)
        predicted_class = CLASSES[predicted_class_idx]
        confidence = prediction[predicted_class_idx] * 100
        
        # Display results
        plt.figure(figsize=(8, 6))
        plt.imshow(img)
        plt.title(f"Prediction: {predicted_class.upper()}\nConfidence: {confidence:.2f}%")
        plt.axis('off')
        plt.show()
        
        print(f"\nPrediction Results:")
        print(f"Class: {predicted_class.upper()}")
        print(f"Confidence: {confidence:.2f}%")
        
        # Display all class probabilities
        print("\nClass Probabilities:")
        for i, class_name in enumerate(CLASSES):
            print(f"{class_name.upper()}: {prediction[i]*100:.2f}%")
            
    except Exception as e:
        print(f"Error making prediction: {e}")

def main():
    
    args = parse_arguments()
    
    
    model_exists = os.path.exists(final_model_path)
    
    
    use_saved_weights = True if model_exists else False
    
    
    if args.use_saved_weights:
        use_saved_weights = True
    save_weights = not args.no_save_weights
    
    
    if args.check_balance:
        print("\n=== Checking Dataset Balance ===")
        check_data_balance()
        return
        
    
    if args.visualize_only:
        print("\n=== Visualizing Sample Images ===")
        plot_sample_images_per_class(data_dir)
        return
    
    
    if use_saved_weights and not model_exists:
        print(f"Warning: Model not found at {final_model_path}. Training new model.")
        use_saved_weights = False
    
    if not use_saved_weights and not args.skip_training:
        print("\n=== Training New Brain Tumor Classification Model ===")
        # Load data
        traingenrator, valgenrator, testgenrator = get_data_generators()

        # Build model 
        model, base_model = build_model()
        
        # Update model to match the 4-class classification task
        # Find and replace the final Dense layer with 4 outputs
        found_final_layer = False
        for i in range(len(model.layers)-1, -1, -1):
            if isinstance(model.layers[i], tf.keras.layers.Dense) and model.layers[i].units == 1:
                # Remove the old final layer
                model.pop()
                # Add new final layer for 4 classes
                model.add(tf.keras.layers.Dense(len(CLASSES), activation='softmax'))
                found_final_layer = True
                break
        
        if not found_final_layer:
            print("Warning: Could not find final Dense layer to replace. Model may not be compatible.")
            # Add the layer anyway
            model.add(tf.keras.layers.Dense(len(CLASSES), activation='softmax'))
        
        # Print model summary
        model.summary()

        # Callbacks
        callbacks_list = get_callbacks()

        print("Training model from scratch...")
        # Train model
        history_1 = train_model_phase1(model, traingenrator, valgenrator, callbacks_list, INITIAL_EPOCHS)
        history_2 = fine_tune_model(model, base_model, traingenrator, valgenrator, callbacks_list, INITIAL_EPOCHS, FINE_TUNE_EPOCHS)
        
        # Combine histories
        history = {}
        history['accuracy'] = history_1.history['accuracy'] + history_2.history['accuracy']
        history['val_accuracy'] = history_1.history['val_accuracy'] + history_2.history['val_accuracy']
        history['loss'] = history_1.history['loss'] + history_2.history['loss']
        history['val_loss'] = history_1.history['val_loss'] + history_2.history['val_loss']
        
        # Visualize
        plot_training_curves(history, INITIAL_EPOCHS)
        
        # Evaluate model
        test_loss, test_accuracy = evaluate_model(model, testgenrator)
        print(f"Final evaluation - Loss: {test_loss:.4f}, Accuracy: {test_accuracy:.4f}")
        
        # Generate and display confusion matrix
        print("\n=== Generating Confusion Matrix ===")
        generate_confusion_matrix(model, testgenrator)
        
        # Confirm model was saved
        if save_weights and os.path.exists(final_model_path):
            print(f"\n✓ Model weights successfully saved to: {final_model_path}")
        elif save_weights:
            print(f"\n✗ Failed to save model weights to: {final_model_path}")
    elif args.skip_training:
        if not model_exists:
            print(f"Error: Cannot skip training because model not found at {final_model_path}")
            return
        print(f"\n=== Skipping Training (--skip-training flag used) ===")
    else:
        print(f"\n=== Using Saved Model ===")
        print(f"Loading model from: {final_model_path}")
    
    # Test on image if provided via command line
    if args.test_image:
        if os.path.exists(args.test_image):
            print(f"\n=== Testing Model on Provided Image ===")
            print(f"Image path: {args.test_image}")
            display_prediction(args.test_image)
        else:
            print(f"\nError: Image file '{args.test_image}' not found.")
    # Optionally prompt for image testing
    else:
        print("\n=== Image Testing ===")
        image_path = prompt_for_image_test()
        if image_path:
            print(f"Testing model on image: {image_path}")
            display_prediction(image_path)
        else:
            print("No image testing requested.")

if __name__ == "__main__":
    
    print("\n" + "="*60)
    print("Brain Tumor Classification Model")
    print("Classes: " + ", ".join([c.upper() for c in CLASSES]))
    print("="*60 + "\n")
    
    main()