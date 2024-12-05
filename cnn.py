import os
import yaml
import time
import tensorflow as tf
import logging
import os
from tensorflow.python.keras import Sequential
from tensorflow.python.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from math import ceil
from preprocessing_data import PreprocessingData
from tensorflow.python.keras.optimizer_v2 import adam


config_path = r"E:\\Github_destop\\Flowers-Classification\\config.yml"
if not os.path.exists(config_path):
    raise FileNotFoundError(f"Tệp cấu hình không tồn tại: {config_path}")
with open(config_path, 'r', encoding='utf-8') as config_file:
    config = yaml.safe_load(config_file)


def train(exp_name="flower_experiment", model_name=None, train_ds=None, val_ds=None):
    """
    Hàm huấn luyện mô hình với cấu hình đã cho và lưu kết quả.

    Args:
        exp_name (str): Tên của experiment để lưu log.
        model_name (str): Tên mô hình hoặc cấu hình mô hình.
        train_ds (tf.data.Dataset): Dữ liệu huấn luyện.
        val_ds (tf.data.Dataset): Dữ liệu kiểm tra.
        config (dict): Cấu hình huấn luyện bao gồm các tham số như batch_size, epochs, ...
    """
    # Kiểm tra nếu chưa có dataset hoặc model
    start_time = time.time()
    # if data_pretrains is None:
    #     data_pretrains = config['data_pretrains']['data_pretrains']
    
    # Mở file log để ghi lại quá trình huấn luyện
    log_folder = config['train']['log_dir']
    # os.makedirs(log_folder, exist_ok=True)
    log_file = os.path.join(log_folder, f"{exp_name}.log")
    logging.basicConfig(
        filename=log_file,
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )
    
    logging.info(f"Starting training experiment: {exp_name}")
    logging.info(f"Model: {model_name}")
    logging.info(f"Training dataset size: {len(train_ds)}")
    logging.info(f"Validation dataset size: {len(val_ds)}")
    
    # Các tham số cấu hình mặc định nếu không có config
    input_shape = config['train']['input_shape']
    num_classes = config['train']['num_classes']
    epochs = config['train']['epochs']
    dropout_rate = config['train']['dropout_rate']
    learning_rate = config['train']['learning_rate'] 
    batch_size = config['train']['batch_size']



    # Định nghĩa mô hình nếu chưa có
    if model_name is None:
        model_name = "flower_classifier_model"
    
    model = Sequential([
        Conv2D(16, 3, padding='same', activation='relu'),
        MaxPooling2D(),
        Conv2D(32, 3, padding='same', activation='relu'),
        MaxPooling2D(),
        Conv2D(64, 3, padding='same', activation='relu'),
        MaxPooling2D(),
        Dropout(dropout_rate),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(num_classes)  # Số lớp phân loại
    ])
    
    model.compile(
        optimizer=adam.Adam(learning_rate=learning_rate),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy']
    )
    
    # Huấn luyện mô hình
    logging.info(f"Training model for {epochs} epochs...")
    pretrained_model = model.fit(
        x = train_ds,
        validation_data=val_ds,
        epochs=epochs
    )
    
    # Sau khi huấn luyện, lưu mô hình
    output_dir = config['train']['model_dir']
    # os.makedirs(output_dir, exist_ok=True)
    model_path = os.path.join(output_dir, f"{exp_name}_model.h5")
    pretrained_model.save(model_path)
    end_time = time.time()
    total_time = end_time - start_time
    logging.info("Time taken to run in seconds: :"+ str(total_time))
    logging.info(f"Model saved to {model_path}.")
    
    logging.info("Training completed.")
    return 'Training completed!!!'

#   def finetuned(self):
#         # Perform fine-tuning if needed
#         base_model = tf.keras.applications.MobileNetV2(input_shape=(self.img_size, self.img_size, 3),
#                                                         include_top=False,
#                                                         weights='imagenet')
#         base_model.trainable = False

#         model = Sequential([
#             self.data_augmentation,
#             base_model,
#             layers.GlobalAveragePooling2D(),
#             Dense(128, activation='relu'),
#             Dense(5)
#         ])
#         model.compile(
#             optimizer='adam',
#             loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
#             metrics=['accuracy']
#         )

#         history = model.fit(self.train_ds, epochs=15, validation_data=self.val_ds)
#         return history
  

if __name__ == '__main__':
    # Create an instance of PreprocessingData class
    preprocessing = PreprocessingData(config)

    # Test load_data method
    print("Loading data...")
    train_ds, val_ds = preprocessing.load_data(config)

    # Test split_data method
    print("Splitting data...")
    train_ds, val_ds = preprocessing.split_data()

    print("Rescaling data without augmentation...")
    rescaled_train_ds, rescaled_val_ds = preprocessing.rescale_data(train_ds, val_ds, apply_augmentation=False)

    train(exp_name="flower_experiment", model_name=None, train_ds=rescaled_train_ds, val_ds=rescaled_val_ds)
    print('Done!!!!!!')

