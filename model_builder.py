import os
import yaml
import tensorflow as tf
from tensorflow.keras import layers, Sequential
from data_loader import get_data_generators, load_config


def build_model(config):
    """
    Build the CNN model based on the configuration.
    """
    img_size = config['loader']['img_size']
    augmentation_config = config['augmentation']

    # Data augmentation
    data_augmentation = Sequential()
    if augmentation_config['flip_horizontal']:
        data_augmentation.add(layers.RandomFlip("horizontal", input_shape=(img_size, img_size, 3)))
    if augmentation_config['rotation_range'] > 0:
        data_augmentation.add(layers.RandomRotation(augmentation_config['rotation_range']))
    if augmentation_config['zoom_range'] > 0:
        data_augmentation.add(layers.RandomZoom(augmentation_config['zoom_range']))

    # Build CNN model
    model = Sequential([
        data_augmentation,
        layers.Rescaling(1./255),
        layers.Conv2D(16, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(32, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(64, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Dropout(0.2),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(len(config['classification']['flower_names']))
    ])

    model.compile(
        optimizer='adam',
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy']
    )

    return model




def train_and_save_model(config):
    """
    Train the model and save it to the specified path.
    """
    # Load datasets
    train_ds, val_ds = get_data_generators(config)

    # Build model
    model = build_model(config)

    # Train the model
    print("\n--- Starting model training ---")
    history = model.fit(train_ds, epochs=15, validation_data=val_ds)

    # Save the model
    model_path = config['model']['saved_model_path']
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    model.save(model_path)
    print(f"\nModel saved at: {model_path}")

    return history


if __name__ == "__main__":
    # Load configuration
    config = load_config()

    # Train and save the model
    history = train_and_save_model(config)
