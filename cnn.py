import tensorflow as tf
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, Rescaling
from tensorflow.python.keras import layers

class FlowerClassifier:
  def __init__(self, img_size=150, batch_size=32):
        self.img_size = img_size
        self.batch_size = batch_size
        self.model = None
        self.train_ds = None
        self.val_ds = None

  def architecture(self):
        # Define the CNN architecture
        self.model = Sequential([
            self.data_augmentation,
            Rescaling(1./255),
            Conv2D(16, 3, padding='same', activation='relu'),
            MaxPooling2D(),
            Conv2D(32, 3, padding='same', activation='relu'),
            MaxPooling2D(),
            Conv2D(64, 3, padding='same', activation='relu'),
            MaxPooling2D(),
            Dropout(0.2),
            Flatten(),
            Dense(128, activation='relu'),
            Dense(5)  # Assuming 5 flower classes
        ])
        self.model.compile(
            optimizer='adam',
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=['accuracy']
        )
  def train(self, epochs=15):
        # Train the model
        if self.model is None:
            raise ValueError("Model architecture not defined. Call architecture() first.")
        history = self.model.fit(self.train_ds, epochs=epochs, validation_data=self.val_ds)
        return history

  def finetuned(self):
        # Perform fine-tuning if needed
        base_model = tf.keras.applications.MobileNetV2(input_shape=(self.img_size, self.img_size, 3),
                                                        include_top=False,
                                                        weights='imagenet')
        base_model.trainable = False

        model = Sequential([
            self.data_augmentation,
            base_model,
            layers.GlobalAveragePooling2D(),
            Dense(128, activation='relu'),
            Dense(5)
        ])
        model.compile(
            optimizer='adam',
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=['accuracy']
        )

        history = model.fit(self.train_ds, epochs=15, validation_data=self.val_ds)
        return history
  

if __name__ == '__main__':
     print('hello')