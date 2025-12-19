#Loads the data from data directory and finds train, test, and validation sets

train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    data_dir / "train",
    image_size=(224, 224),
    batch_size=32,
)

val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    data_dir / "val",
    image_size=(224, 224),
    batch_size=32,
)

test_ds = tf.keras.preprocessing.image_dataset_from_directory(
    data_dir / "test",
    image_size=(224, 224),
    batch_size=32,
)
