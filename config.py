import os

data_dir = r'brain_tumor_project\brain_tumor_data'


TRAIN_SPLIT = 0.7
VAL_SPLIT = 0.15
TEST_SPLIT = 0.15

# Image size
IMG_SIZE = (224, 224)

# Model checkpoint paths
checkpoint_dir = os.path.join(os.path.dirname(__file__), "checkpoints")
os.makedirs(checkpoint_dir, exist_ok=True)
checkpoint_path = os.path.join(checkpoint_dir, "brain_tumor_model_checkpoint.keras")
final_model_path = os.path.join(checkpoint_dir, "brain_tumor_detection_final.keras")

USE_SAVED_WEIGHTS = False  
SAVE_WEIGHTS = True  


temp_dir = os.path.join(os.path.dirname(data_dir), "temp_splits")
temp_train_dir = os.path.join(temp_dir, "train")
temp_val_dir = os.path.join(temp_dir, "val")
temp_test_dir = os.path.join(temp_dir, "test")


CLASSES = ["glioma", "meningioma", "notumor", "pituitary"]

# Training parameters
INITIAL_EPOCHS = 10
FINE_TUNE_EPOCHS = 15