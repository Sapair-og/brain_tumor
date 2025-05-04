import os
from tensorflow.keras.models import Sequential, load_model  #type:ignore
from tensorflow.keras.layers import GlobalAveragePooling2D, BatchNormalization, Dense, Dropout #type:ignore
from tensorflow.keras.applications import ResNet50V2 #type:ignore
from config import IMG_SIZE, final_model_path, USE_SAVED_WEIGHTS,CLASSES


def build_model():
    """Build and return the model architecture"""

    if USE_SAVED_WEIGHTS and os.path.exists(final_model_path):
          print(f"Loading saved model from {final_model_path}")
          try:
                model =load_model(final_model_path)

                base_model =None
                for layer in model.layers:
                    if hasattr(layer,'layers') and len(layer.layers) > 0:
                         base_model =layer
                         break
                if base_model is None:
                    print("Warning: Base model not found in loaded model. Using default ResNet50V2.") 
                    base_model =ResNet50V2(
                         weights='imagenet',
                         include_top=False,
                         input_shape=(IMG_SIZE[0],IMG_SIZE[1],3)
                    )
                print("Model loaded successfully.")
                return model,base_model
          except Exception as e:
               print(f"Error loading model:{e}")
               print("Building new model instead.")

            

    base_model = ResNet50V2(
         weights='imagenet',
         include_top=False,
         input_shape=(IMG_SIZE[0],IMG_SIZE[1],3)
    )
    base_model.trainable=False

    model = Sequential([
         base_model,
         GlobalAveragePooling2D(),
         BatchNormalization(),
         Dense(512,activation='relu'),
         BatchNormalization(momentum=0.9),
         Dropout(0.4),
         Dense(128,activation='relu'),
         BatchNormalization(momentum=0.9),
         Dropout(0.3),
         Dense(len(CLASSES), activation='softmax')
    ])
    

    return model,base_model