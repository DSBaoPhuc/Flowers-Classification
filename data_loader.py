import yaml
import tensorflow as tf

def load_config():
    with open("config.yml", "r") as file:
        return yaml.safe_load(file)

def get_data_generators(config):
    """
    Load dataset and split into training and validation sets.
    """
    img_size = config['loader']['img_size']
    batch_size = config['loader']['batch_size']
    validation_split = config['loader']['validation_split']
    data_dir = config['loader']['traindata_dir']

    # Create training dataset
    train_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        seed=123,
        validation_split=validation_split,
        subset="training",
        batch_size=batch_size,
        image_size=(img_size, img_size)
    )
    
    # Create validation dataset
    val_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        seed=123,
        validation_split=validation_split,
        subset="validation",
        batch_size=batch_size,
        image_size=(img_size, img_size)
    )

    # Optimize dataset loading
    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

    return train_ds, val_ds
