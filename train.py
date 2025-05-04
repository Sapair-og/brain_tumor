import tensorflow as tf
from rich.progress import Progress

def train_model_phase1(model, traingenrator, valgenrator, callbacks_list, initial_epochs):
    print("Phase 1: Training only top layers...")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss='categorical_crossentropy',  # Changed from binary to categorical for multi-class
        metrics=['accuracy']
    )

    history = {'loss': [], 'accuracy': [], 'val_loss': [], 'val_accuracy': []}
    
    for callback in callbacks_list:
        if hasattr(callback, 'set_model'):
            callback.set_model(model)
        if hasattr(callback, 'on_train_begin'):
            callback.on_train_begin()

    for epoch in range(initial_epochs):
        print(f"\nEpoch {epoch+1}/{initial_epochs}")
        
        for callback in callbacks_list:
            if hasattr(callback, 'on_epoch_begin'):
                callback.on_epoch_begin(epoch)
                
        with Progress() as progress:
            task = progress.add_task("[green]Training...", total=len(traingenrator))
            epoch_loss = 0
            epoch_accuracy = 0
            for batch_x, batch_y in traingenrator:
                metrics = model.train_on_batch(batch_x, batch_y)
                epoch_loss += metrics[0]
                epoch_accuracy += metrics[1]
                progress.update(task, advance=1)
                if progress.finished:
                    break

        history['loss'].append(epoch_loss / len(traingenrator))
        history['accuracy'].append(epoch_accuracy / len(traingenrator))

        val_loss, val_accuracy = model.evaluate(valgenrator, verbose=0)
        history['val_loss'].append(val_loss)
        history['val_accuracy'].append(val_accuracy)

        print(f"Loss: {history['loss'][-1]:.4f} - Accuracy: {history['accuracy'][-1]:.4f} - Val Loss: {val_loss:.4f} - Val Accuracy: {val_accuracy:.4f}")

        logs = {
            'loss': history['loss'][-1], 
            'accuracy': history['accuracy'][-1], 
            'val_loss': val_loss, 
            'val_accuracy': val_accuracy
        }
        
        should_stop = False
        for callback in callbacks_list:
            if hasattr(callback, 'on_epoch_end'):
                callback.on_epoch_end(epoch, logs)
                if hasattr(callback, 'stopped_epoch') and callback.stopped_epoch > 0:
                    should_stop = True
        
        if should_stop:
            break

    for callback in callbacks_list:
        if hasattr(callback, 'on_train_end'):
            callback.on_train_end()

    return type('History', (object,), {'history': history})

def fine_tune_model(model, base_model, traingenrator, valgenrator, callbacks_list, initial_epochs, fine_tune_epochs):
    print("Phase 2: Fine-tuning...")
    base_model.trainable = True
    for layer in base_model.layers[:int(len(base_model.layers) * 0.7)]:
        layer.trainable = False

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=5e-5),
        loss='categorical_crossentropy',  # Changed from binary to categorical for multi-class
        metrics=['accuracy']
    )

    total_epochs = initial_epochs + fine_tune_epochs
    history = {'loss': [], 'accuracy': [], 'val_loss': [], 'val_accuracy': []}
    
    for callback in callbacks_list:
        if hasattr(callback, 'set_model'):
            callback.set_model(model)
        if hasattr(callback, 'on_train_begin'):
            callback.on_train_begin()

    for epoch in range(initial_epochs, total_epochs):
        print(f"\nEpoch {epoch+1}/{total_epochs}")
        
        for callback in callbacks_list:
            if hasattr(callback, 'on_epoch_begin'):
                callback.on_epoch_begin(epoch)
                
        with Progress() as progress:
            task = progress.add_task("[cyan]Fine-tuning...", total=len(traingenrator))
            epoch_loss = 0
            epoch_accuracy = 0
            for batch_x, batch_y in traingenrator:
                metrics = model.train_on_batch(batch_x, batch_y)
                epoch_loss += metrics[0]
                epoch_accuracy += metrics[1]
                progress.update(task, advance=1)
                if progress.finished:
                    break

        history['loss'].append(epoch_loss / len(traingenrator))
        history['accuracy'].append(epoch_accuracy / len(traingenrator))

        val_loss, val_accuracy = model.evaluate(valgenrator, verbose=0)
        history['val_loss'].append(val_loss)
        history['val_accuracy'].append(val_accuracy)

        print(f"Loss: {history['loss'][-1]:.4f} - Accuracy: {history['accuracy'][-1]:.4f} - Val Loss: {val_loss:.4f} - Val Accuracy: {val_accuracy:.4f}")

        logs = {
            'loss': history['loss'][-1], 
            'accuracy': history['accuracy'][-1], 
            'val_loss': val_loss, 
            'val_accuracy': val_accuracy
        }
        
        should_stop = False
        for callback in callbacks_list:
            if hasattr(callback, 'on_epoch_end'):
                callback.on_epoch_end(epoch, logs)
                if hasattr(callback, 'stopped_epoch') and callback.stopped_epoch > 0:
                    should_stop = True
        
        if should_stop:
            break
            
    for callback in callbacks_list:
        if hasattr(callback, 'on_train_end'):
            callback.on_train_end()

    return type('History', (object,), {'history': history})