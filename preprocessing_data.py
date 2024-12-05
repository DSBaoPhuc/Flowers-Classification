import os
import yaml
import random
import tensorflow as tf
from tensorflow.python.keras.models import Sequential
import numpy as np
# from tensorflow.python.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense
from tensorflow.python.keras import layers
# from tensorflow.python.keras.layers import RandomRotation
# from tensorflow.keras.layers import Rescaling
# from tensorflow.keras.layers import RandomRotation, RandomFlip, RandomZoom, RandomTranslation, RandomBrightness, Cropping2D

# from tensorflow.python.keras.optimizers import RMSprop
# from tensorflow.keras.preprocessing import image_dataset_from_directory
# import configparser
# from configparser import ConfigParser, ExtendedInterpolation

config_path = r"E:\\Github_destop\\Flowers-Classification\\config.yml"
if not os.path.exists(config_path):
    raise FileNotFoundError(f"Tệp cấu hình không tồn tại: {config_path}")
with open(config_path, 'r', encoding='utf-8') as config_file:
    config = yaml.safe_load(config_file)


class PreprocessingData:
    def __init__(self, config):
        # config_path = r"E:\\Github_destop\\Flowers-Classification\\config.yml"

        # if not os.path.exists(config_path):
        #     raise FileNotFoundError(f"Tệp cấu hình không tồn tại: {config_path}")

        # # Đọc file YAML
        # with open(config_path, 'r', encoding='utf-8') as config_file:
        #     config = yaml.safe_load(config_file)

        # Lấy thông tin từ file YAML
        self.base_dir = config['default']['base_dir']
        self.traindata_dir = os.path.join(self.base_dir, config['loader']['traindata_dir'].strip('/'))
        self.img_size = config['loader']['img_size']
        self.batch_size = config['loader']['batch_size']

        # Lấy các tham số từ phần 'augmentation' trong config
        self.flip_horizontal = config['augmentation']['flip_horizontal']
        self.rotation_range = config['augmentation']['rotation_range']
        self.zoom_range = config['augmentation']['zoom_range']
        self.translation_range = config['augmentation']['translation_range']
        self.brightness_range = config['augmentation']['brightness_range']
        self.crop_percentage = config['augmentation']['crop_percentage']

        #lấy thống tin cho phần data_pretrains
        self.data_pretrains = os.path.join(self.base_dir, config['data_pretrains']['data_pretrains'].strip('/'))

        # Kiểm tra đường dẫn
        if not os.path.exists(self.base_dir):
            print("0000000000000000")
            raise ValueError(f"Base directory {self.base_dir} không tồn tại.")
        if not os.path.exists(self.traindata_dir):
            print("0000000000000000")
            raise ValueError(f"Training data directory {self.traindata_dir} không tồn tại.")

        # In thông tin cấu hình để kiểm tra
        print(f"Base directory: {self.base_dir}")
        print(f"Training data directory: {self.traindata_dir}")
        print(f"Image size: {self.img_size}")
        print(f"Batch size: {self.batch_size}")

    def data_augmentation(self, augmentation_types=None):
        """
        Tạo một pipeline data augmentation với các phép biến đổi ngẫu nhiên từ danh sách các phép biến đổi được chỉ định.

        Args:
            augmentation_types (list): Danh sách các phép biến đổi muốn áp dụng. Ví dụ: ['rotate', 'flip']

        Returns:
            Sequential: Đối tượng Sequential chứa các lớp biến đổi dữ liệu.
        """
        if augmentation_types is None:
            augmentation_types = ['rotate', 'flip', 'zoom', 'translation', 'brightness', 'crop','all']
        
        data_augmentation = Sequential()

        # Chọn ngẫu nhiên các phép biến đổi cần áp dụng
        if 'rotate' in augmentation_types:
            data_augmentation.add(layers.RandomRotation(self.rotation_range))
        
        if 'flip' in augmentation_types:
            if self.flip_horizontal:
                data_augmentation.add(layers.Flip("horizontal", input_shape=(self.img_size, self.img_size, 3)))

        if 'zoom' in augmentation_types:
            data_augmentation.add(layers.Zoom(self.zoom_range))

        if 'translation' in augmentation_types:
            data_augmentation.add(layers.Translation(height_factor=self.translation_range, width_factor=self.translation_range))

        if 'brightness' in augmentation_types:
            data_augmentation.add(layers.Brightness(factor=self.brightness_range))

        if 'crop' in augmentation_types:
            data_augmentation.add(layers.Cropping2D(height=int(self.img_size * self.crop_percentage), width=int(self.img_size * self.crop_percentage)))
        if 'all' in augmentation_types:
            if self.flip_horizontal:
                data_augmentation.add(layers.Flip("horizontal", input_shape=(self.img_size, self.img_size, 3)))
            data_augmentation.add(layers.Rotation(self.rotation_range))
            data_augmentation.add(layers.Zoom(self.zoom_range))
            data_augmentation.add(layers.Translation(height_factor=self.translation_range, width_factor=self.translation_range))
            data_augmentation.add(layers.Brightness(factor=self.brightness_range))
            data_augmentation.add(layers.Crop(height=int(self.img_size * self.crop_percentage), width=int(self.img_size * self.crop_percentage)))

        return data_augmentation

    def randomize_augmentation(self, augmentation_types=None):
        """
        Lựa chọn ngẫu nhiên các phép biến đổi từ danh sách và áp dụng chúng.

        Args:
            augmentation_types (list): Danh sách các phép biến đổi muốn áp dụng. Ví dụ: ['rotate', 'flip']

        Returns:
            Sequential: Đối tượng Sequential chứa các lớp biến đổi dữ liệu.
        """
        if augmentation_types is None:
            augmentation_types = ['rotate', 'flip', 'zoom', 'translation', 'brightness', 'crop']

        # Chọn một số phép biến đổi ngẫu nhiên từ danh sách
        selected_augmentations = random.sample(augmentation_types, k=random.randint(1, len(augmentation_types)))

        # Áp dụng augmentation
        return self.data_augmentation(selected_augmentations)

    def load_data(self,config,data_dir = None):
        """
        Tải dữ liệu huấn luyện và kiểm tra từ thư mục.

        Args:
            data_dir (str): Đường dẫn đến thư mục chứa dữ liệu.

        Returns:
            tuple: Trả về tập huấn luyện và kiểm tra.
        """
        # Kiểm tra số lượng dữ liệu trong tập datasets
        if data_dir is None:
            data_dir = self.traindata_dir
        count = 0
        dirs = os.listdir(data_dir)  # Lấy danh sách các thư mục con

        for dir in dirs:
            dir_path = os.path.join(data_dir, dir)  # Tạo đường dẫn đầy đủ đến thư mục con
            if os.path.isdir(dir_path):  # Kiểm tra xem có phải là thư mục không
                files = os.listdir(dir_path)  # Lấy danh sách các file trong thư mục
                print(f"{dir} folder has {len(files)} images")
                count += len(files)

        print(f"Images folder has {count} images")
        print("data folder : " + self.traindata_dir)

        # Chia dữ liệu thành tập huấn luyện và kiểm tra
        train_ds = tf.keras.preprocessing.image_dataset_from_directory(
            data_dir,
            validation_split=config['loader']['validation_split'],
            subset="training",
            seed=123,
            image_size=(self.img_size, self.img_size),
            batch_size=self.batch_size
        )

        val_ds = tf.keras.preprocessing.image_dataset_from_directory(
            data_dir,
            validation_split=config['loader']['validation_split'],
            subset="validation",
            seed=123,
            image_size=(self.img_size, self.img_size),
            batch_size=self.batch_size
        )

        return train_ds, val_ds
    
    def split_data(self):
        """
        Chia dữ liệu thành tập huấn luyện và kiểm tra.
        
        Returns:
            tuple: Trả về tập huấn luyện và kiểm tra.
        """
        # Sử dụng load_data để lấy dữ liệu
        train_ds, val_ds = self.load_data(config,data_dir = None)

        # Tối ưu hóa hiệu suất đọc dữ liệu
        train_ds = train_ds.prefetch(buffer_size=tf.data.AUTOTUNE)
        val_ds = val_ds.prefetch(buffer_size=tf.data.AUTOTUNE)

        return train_ds, val_ds
    

    def rescale_data(self, train_ds, val_ds, apply_augmentation=False):
        """
        Tạo lớp chuẩn hóa dữ liệu (Rescaling) và áp dụng lên train_ds và val_ds.
        
        Args:
            train_ds (tf.data.Dataset): Tập huấn luyện.
            val_ds (tf.data.Dataset): Tập kiểm tra.
            apply_augmentation (bool): Nếu True, sẽ thực hiện augmentation trước khi chuẩn hóa.
        
        Returns:
            tuple: Trả về train_ds và val_ds sau khi đã chuẩn hóa.
        """
        def rescale_lambda(x):
            return x / 255.0
        
        # Nếu áp dụng augmentation, chúng ta cần thực hiện sau khi augmentation
        if apply_augmentation:
            # Áp dụng augmentation lên train_ds và val_ds trước khi rescaling
            train_ds = train_ds.map(lambda x, y: (self.data_augmentation()(x), y))
            val_ds = val_ds.map(lambda x, y: (self.data_augmentation()(x), y))

        # Áp dụng rescaling vào train_ds và val_ds
        train_ds = train_ds.map(lambda x, y: (rescale_lambda(x), y))
        val_ds = val_ds.map(lambda x, y: (rescale_lambda(x), y))

        return train_ds, val_ds
    
    def save_pretrain(self,dataset,output_file = None, prefix = str):
        """
        Lưu một tf.data.Dataset vào folder dưới dạng các file .npz.
    
        Args:
            dataset (tf.data.Dataset): Dataset cần lưu.
            folder_name (str): Đường dẫn tới folder lưu trữ.
            prefix (str): Tiền tố cho tên file (ví dụ: 'train' hoặc 'val').
        """
        if output_file is None:
            output_file = self.data_pretrains


        for i, (images, labels) in enumerate(dataset):
            file_path = os.path.join(output_file, f"{prefix}_batch_{i}.npz")
            np.savez(file_path, images=images.numpy(), labels=labels.numpy())
            print(f"Lưu batch {i} vào {file_path}")

# Kiểm tra nhanh module khi chạy trực tiếp file
if __name__ == "__main__":
    
    # Create an instance of PreprocessingData class
    preprocessing = PreprocessingData(config)

    # Test load_data method
    print("Loading data...")
    train_ds, val_ds = preprocessing.load_data(config)

    # Test split_data method
    print("Splitting data...")
    train_ds, val_ds = preprocessing.split_data()

    # # Test data augmentation
    # print("Applying data augmentation (flip, rotate)...")
    # augmented_train_ds = train_ds.map(lambda x, y: (preprocessing.data_augmentation(['rotate', 'flip'])(x), y))

    # Test rescaling data without augmentation
    print("Rescaling data without augmentation...")
    rescaled_train_ds, rescaled_val_ds = preprocessing.rescale_data(train_ds, val_ds, apply_augmentation=False)
    
    print("Save data after processing...")
    preprocessing.save_pretrain(train_ds,prefix = "train")
    preprocessing.save_pretrain(val_ds,prefix ="val")
    # # Test rescaling data with augmentation
    # print("Rescaling data with augmentation...")
    # rescaled_train_ds, rescaled_val_ds = preprocessing.rescale_data(train_ds, val_ds, apply_augmentation=True)
    
    print("All tests completed!")
