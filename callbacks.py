from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint #type: ignore
from config import checkpoint_path



def get_callbacks():
    """
    Get the list of callbacks for model training
    
    Returns:
        list: List of Keras callbacks
    """
    early_stopping =EarlyStopping(
        monitor = 'val_loss',
        patience=3,
        restore_best_weights = True,
        verbose=1,
        baseline=None
    )

    reduce_lr = ReduceLROnPlateau(
        monitor = 'val_loss',
        factor=0.2,
        patience=2,
        min_lr=0.00001,
        verbose=1
    )


    checkpoint = ModelCheckpoint(
        checkpoint_path,
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1,
        save_weights_only=False
    )

    return [early_stopping,reduce_lr,checkpoint]