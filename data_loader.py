from tensorflow.keras.preprocessing.image import ImageDataGenerator #type: ignore
import os
import shutil
import random
from sklearn.model_selection import train_test_split
from tensorflow.keras.applications.resnet_v2 import preprocess_input  #type: ignore


from config import data_dir, IMG_SIZE, TRAIN_SPLIT, VAL_SPLIT, TEST_SPLIT, CLASSES
from config import temp_dir, temp_train_dir, temp_val_dir, temp_test_dir

def prepare_split_directories():
    """Create train/val/test splits from the combined data directory"""
    
    for dir_path in [temp_train_dir, temp_val_dir, temp_test_dir]:
        for class_name in CLASSES:
            os.makedirs(os.path.join(dir_path, class_name), exist_ok=True)
    
    
    class_files = {}
    
    
    for class_name in CLASSES:
        class_dir = os.path.join(data_dir, class_name)
        class_files[class_name] = [f for f in os.listdir(class_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    
    for class_name in CLASSES:
        print(f"Sample {class_name} files: {class_files[class_name][:3]}")
    
    # Dictionary to store split files
    splits = {
        'train': {},
        'val': {},
        'test': {}
    }
    
    # Split the files for each class
    for class_name in CLASSES:
        train_files, temp_files = train_test_split(
            class_files[class_name], 
            test_size=(VAL_SPLIT + TEST_SPLIT), 
            random_state=42
        )
        
        val_files, test_files = train_test_split(
            temp_files, 
            test_size=TEST_SPLIT/(VAL_SPLIT + TEST_SPLIT), 
            random_state=42
        )
        
        splits['train'][class_name] = train_files
        splits['val'][class_name] = val_files
        splits['test'][class_name] = test_files
    
    
    print(f"\nDataset distribution:")
    for class_name in CLASSES:
        total = len(class_files[class_name])
        train_count = len(splits['train'][class_name])
        val_count = len(splits['val'][class_name])
        test_count = len(splits['test'][class_name])
        
        print(f"{class_name.upper()} - Total: {total}, Train: {train_count}, Val: {val_count}, Test: {test_count}")
    
    # Copy files to temporary directories
    for split_name, split_data in splits.items():
        if split_name == 'train':
            dest_dir = temp_train_dir
        elif split_name == 'val':
            dest_dir = temp_val_dir
        else:  # test
            dest_dir = temp_test_dir
            
        for class_name, file_list in split_data.items():
            for filename in file_list:
                src_path = os.path.join(data_dir, class_name, filename)
                dest_path = os.path.join(dest_dir, class_name, filename)
                shutil.copy2(src_path, dest_path)
    
    print(f"Data successfully split into {temp_train_dir}, {temp_val_dir}, and {temp_test_dir}")
    return temp_train_dir, temp_val_dir, temp_test_dir

def get_data_generators():
    
    train_dir, val_dir, test_dir = prepare_split_directories()
    
    
    traindata_gen = ImageDataGenerator(
        preprocessing_function=preprocess_input,
        rotation_range=20,
        zoom_range=0.15,
        shear_range=0.15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        brightness_range=[0.85, 1.15],
        horizontal_flip=True,
        fill_mode='constant',
        cval=0
    )
    

    testdata_gen = ImageDataGenerator(preprocessing_function=preprocess_input)
    valdata_gen = ImageDataGenerator(preprocessing_function=preprocess_input)
    
    
    print("Creating training data generator...")
    traingenrator = traindata_gen.flow_from_directory(
        train_dir,
        target_size=IMG_SIZE,
        batch_size=32,
        class_mode='categorical',  
        shuffle=True,
        classes=CLASSES
    )
    
    print("Creating validation data generator...")
    valgenrator = valdata_gen.flow_from_directory(
        val_dir,
        target_size=IMG_SIZE,
        batch_size=32,
        class_mode='categorical',  
        classes=CLASSES
    )
    
    print("Creating test data generator...")
    testgenrator = testdata_gen.flow_from_directory(
        test_dir,
        target_size=IMG_SIZE,
        batch_size=32,
        class_mode='categorical', 
        shuffle=False,
        classes=CLASSES
    )
    
    # Print class indices for verification
    print(f"Class indices mapping - Training: {traingenrator.class_indices}")
    print(f"Class indices mapping - Validation: {valgenrator.class_indices}")
    print(f"Class indices mapping - Test: {testgenrator.class_indices}")
    
    # Verify the first few images and labels
    print("Checking a few training samples:")
    for i in range(3):
        batch_x, batch_y = next(traingenrator)
        print(f"Batch {i+1}: Images shape={batch_x.shape}, Labels shape={batch_y.shape}")
        for j in range(3):  
            label_idx = batch_y[j].argmax()  
            label_name = CLASSES[label_idx]
            print(f"  Sample {j+1}: Label={label_name} (index={label_idx})")
    
    return traingenrator, valgenrator, testgenrator

def check_data_balance():
    """
    Check the balance of classes in the dataset splits
    Returns a summary of the class distribution
    """
    # Get directories
    train_dir, val_dir, test_dir = prepare_split_directories()
    
    # Initialize a dictionary to store the counts
    counts = {
        "train": {},
        "val": {},
        "test": {}
    }
    
    
    for class_name in CLASSES:
        counts["train"][class_name] = len(os.listdir(os.path.join(train_dir, class_name)))
        counts["val"][class_name] = len(os.listdir(os.path.join(val_dir, class_name)))
        counts["test"][class_name] = len(os.listdir(os.path.join(test_dir, class_name)))
    
    # Print summary
    print("\n----- Dataset Balance Check -----")
    for split in ["train", "val", "test"]:
        print(f"{split.capitalize()} set:")
        for class_name in CLASSES:
            print(f"  {class_name.upper()}: {counts[split][class_name]}")
        
        # Calculate the percentage of each class
        total = sum(counts[split].values())
        if total > 0:
            percentages = {class_name: (count/total)*100 for class_name, count in counts[split].items()}
            print(f"  Distribution percentages:")
            for class_name in CLASSES:
                print(f"    {class_name.upper()}: {percentages[class_name]:.2f}%")
    
    return counts

if __name__ == "__main__":
    
    balance_info = check_data_balance()
    
    # Create generators and print class indices
    train_gen, val_gen, test_gen = get_data_generators()
    print(f"\nClass indices mapping: {train_gen.class_indices}")
    
    # This confirms the mapping of class indices
    print("This mapping is critical for correct predictions")