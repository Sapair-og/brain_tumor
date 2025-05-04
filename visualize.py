import matplotlib.pyplot as plt
import os
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array #type:ignore
from tensorflow.keras.models import load_model  #type: ignore
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

from config import IMG_SIZE, final_model_path, CLASSES

def plot_training_curves(history, initial_epochs):
    """Plot the training and validation accuracy/loss curves"""
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(history['accuracy'], label='Train Accuracy')
    plt.plot(history['val_accuracy'], label='Val Accuracy')
    plt.plot([initial_epochs-1, initial_epochs-1], plt.ylim(), 'k--')
    plt.legend()
    plt.title('Accuracy Curves')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.annotate('Fine-tuning', xy=(initial_epochs, 0.5))

    plt.subplot(1, 2, 2)
    plt.plot(history['loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Val Loss')
    plt.plot([initial_epochs-1, initial_epochs-1], plt.ylim(), 'k--')
    plt.legend()
    plt.title('Loss Curves')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')

    plt.tight_layout()
    plt.savefig('training_curves.png')
    plt.show()

def visualize_predictions(model, images, true_labels=None):
    """
    Visualize model predictions on a set of images
    
    Args:
        model: Trained keras model
        images: List of image paths
        true_labels: Optional list of true numeric labels (0-3 representing the four classes)
    """
    num_images = len(images)
    rows = (num_images + 2) // 3  # Calculate rows needed
    
    plt.figure(figsize=(15, 5 * rows))
    
    for i, img_path in enumerate(images):
        # Load and preprocess image
        img = load_img(img_path, target_size=IMG_SIZE)
        img_array = img_to_array(img)
        img_array = img_array / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        # Predict
        prediction = model.predict(img_array)[0]
        predicted_class_idx = np.argmax(prediction)
        predicted_class = CLASSES[predicted_class_idx]
        confidence = prediction[predicted_class_idx]
        
        # Plot
        plt.subplot(rows, 3, i+1)
        plt.imshow(img)
        
        title = f"Pred: {predicted_class.upper()} ({confidence:.2f})"
        if true_labels is not None:
            true_class = CLASSES[true_labels[i]]
            title = f"True: {true_class.upper()}\n{title}"
            
        plt.title(title)
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('prediction_visualization.png')
    plt.show()

def generate_confusion_matrix(model, test_generator):
    """
    Generate and visualize confusion matrix for the model
    
    Args:
        model: Trained model
        test_generator: Test data generator
    """
    # Reset the generator to ensure we start from the beginning
    test_generator.reset()
    
    # Get predictions
    steps = len(test_generator)
    predictions = model.predict(test_generator, steps=steps, verbose=1)
    predicted_classes = np.argmax(predictions, axis=1)
    
    # Get true labels (need to get enough samples)
    true_classes = test_generator.classes[:len(predicted_classes)]
    
    # Compute confusion matrix
    cm = confusion_matrix(true_classes, predicted_classes)
    
    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=[c.upper() for c in CLASSES],
                yticklabels=[c.upper() for c in CLASSES])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.savefig('confusion_matrix.png')
    plt.show()
    
    # Print classification report
    print(classification_report(true_classes, predicted_classes, target_names=[c.upper() for c in CLASSES]))
    
    return cm

def visualize_class_activation_maps(model, img_paths, layer_name='block5_conv3'):
    """
    Visualize class activation maps (CAM) to show which regions of the image most influenced the prediction
    
    Args:
        model: Trained model
        img_paths: List of image paths to visualize
        layer_name: Name of the convolutional layer to use for generating activation maps
    """
    import tensorflow as tf
    from tensorflow.keras.models import Model  #type:ignore
    
    # Create a model that will return the activations of the specified layer
    try:
        # Find the layer by name
        cam_layer = None
        for layer in model.layers:
            if layer_name in layer.name:
                cam_layer = layer
                break
        
        if cam_layer is None:
            print(f"Layer '{layer_name}' not found. Available layers:")
            for layer in model.layers:
                if 'conv' in layer.name:
                    print(f"  - {layer.name}")
            return
            
        # Create the activation model
        activation_model = Model(
            inputs=model.input,
            outputs=[cam_layer.output, model.output]
        )
    except Exception as e:
        print(f"Error creating activation model: {e}")
        return
    
    num_images = len(img_paths)
    rows = num_images
    
    plt.figure(figsize=(12, 5 * rows))
    
    for i, img_path in enumerate(img_paths):
        # Load and preprocess image
        img = load_img(img_path, target_size=IMG_SIZE)
        img_array = img_to_array(img)
        img_array = img_array / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        # Get activations and predictions
        try:
            activations, predictions = activation_model.predict(img_array)
            predicted_class_idx = np.argmax(predictions[0])
            predicted_class = CLASSES[predicted_class_idx]
            
            # Create heatmap
            activations = activations[0]
            heatmap = np.mean(activations, axis=-1)
            heatmap = np.maximum(heatmap, 0)
            heatmap /= np.max(heatmap)
            
            # Original image
            plt.subplot(rows, 3, i*3 + 1)
            plt.imshow(img)
            plt.title(f"Original\nPredicted: {predicted_class.upper()}")
            plt.axis('off')
            
            # Heatmap
            plt.subplot(rows, 3, i*3 + 2)
            plt.imshow(heatmap, cmap='viridis')
            plt.title('Activation Map')
            plt.axis('off')
            
            # Heatmap overlaid on original image
            plt.subplot(rows, 3, i*3 + 3)
            img_display = np.array(img)
            heatmap_resized = np.uint8(255 * heatmap)
            heatmap_resized = np.resize(heatmap_resized, (IMG_SIZE[0], IMG_SIZE[1]))
            heatmap_colored = plt.cm.viridis(heatmap_resized)[:,:,:3]
            superimposed_img = heatmap_colored * 0.4 + img_display / 255.0 * 0.6
            plt.imshow(superimposed_img)
            plt.title('Activation Overlay')
            plt.axis('off')
        except Exception as e:
            print(f"Error processing image {img_path}: {e}")
            plt.subplot(rows, 3, i*3 + 1)
            plt.imshow(img)
            plt.title("Error generating CAM")
            plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('activation_maps.png')
    plt.show()

def plot_sample_images_per_class(data_dir):
    """
    Plot sample images from each class in the dataset
    
    Args:
        data_dir: Directory containing class subdirectories
    """
    plt.figure(figsize=(16, 4 * len(CLASSES)))
    
    for i, class_name in enumerate(CLASSES):
        class_dir = os.path.join(data_dir, class_name)
        image_files = os.listdir(class_dir)
        
        # Get 5 random images from this class
        sample_images = np.random.choice(image_files, min(5, len(image_files)), replace=False)
        
        for j, img_file in enumerate(sample_images):
            img_path = os.path.join(class_dir, img_file)
            img = plt.imread(img_path)
            
            plt.subplot(len(CLASSES), 5, i*5 + j + 1)
            plt.imshow(img)
            plt.title(f"{class_name.upper()}")
            plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('sample_images.png')
    plt.show()

def visualize_feature_maps(model, img_path, layer_names=None):
    """
    Visualize feature maps from different layers of the model for a given image
    
    Args:
        model: Trained model
        img_path: Path to the image to visualize
        layer_names: List of layer names to visualize. If None, will use a default selection.
    """
    from tensorflow.keras.models import Model  #type:ignore
    
    # Default to visualizing early, middle and late convolutional layers if no names provided
    if layer_names is None:
        # Find convolutional layers
        conv_layers = []
        for layer in model.layers:
            if 'conv' in layer.name and 'block' in layer.name:
                conv_layers.append(layer.name)
        
        # Sample from early, middle and late conv layers
        if len(conv_layers) >= 3:
            layer_names = [
                conv_layers[0],  # Early
                conv_layers[len(conv_layers)//2],  # Middle
                conv_layers[-1]  # Late
            ]
        else:
            layer_names = conv_layers
    
    # Load image
    img = load_img(img_path, target_size=IMG_SIZE)
    img_array = img_to_array(img)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    # Make prediction
    prediction = model.predict(img_array)[0]
    predicted_class_idx = np.argmax(prediction)
    predicted_class = CLASSES[predicted_class_idx]
    
    # Create visualization for each requested layer
    for layer_name in layer_names:
        try:
            # Get the output of the layer
            activation_model = Model(inputs=model.input, outputs=model.get_layer(layer_name).output)
            activation = activation_model.predict(img_array)
            
            # Plot the feature maps
            features = activation[0]
            num_features = min(64, features.shape[-1])  # Show at most 64 feature maps
            size = int(np.ceil(np.sqrt(num_features)))
            
            plt.figure(figsize=(size*2, size*2))
            plt.suptitle(f"Feature Maps for Layer: {layer_name} (Predicted: {predicted_class.upper()})")
            
            for i in range(num_features):
                plt.subplot(size, size, i+1)
                plt.imshow(features[:, :, i], cmap='viridis')
                plt.axis('off')
            
            plt.tight_layout(rect=[0, 0, 1, 0.95])  # Make room for the suptitle
            plt.savefig(f'feature_maps_{layer_name}.png')
            plt.show()
        except Exception as e:
            print(f"Error visualizing layer {layer_name}: {e}")
