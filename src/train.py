#This trains and saves the best model; returns the history
from tensorflow.keras.callbacks import ModelCheckpoint

def train_with_checkpoint(model, train_ds, val_ds, epochs, save_path):
    checkpoint = ModelCheckpoint(
        save_path,
        monitor="val_accuracy",
        save_best_only=True,
        mode="max",
        verbose=1,
    )

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=[checkpoint],
    )
    return history
