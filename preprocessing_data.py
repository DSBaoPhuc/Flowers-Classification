import os
import tensorflow as tf
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense
from tensorflow.python.keras import layers
# from tensorflow.python.keras.optimizers import RMSprop
# from tensorflow.keras.preprocessing import image_dataset_from_directory
import configparser

class PreprocessingData:
    def __init__(self, config_path='../config.ini'):
        simple_config = configparser.ConfigParser()
        simple_config.read(config_path)
        simple_config.sections()

        # Lấy thông tin từ phần 'default' và 'preprocess' trong file config
        self.base_dir = simple_config['default']['base_dir']
        self.traindata_dir = os.path.join(self.base_dir, simple_config['preprocess']['traindata_dir'].strip('/'))
        self.img_size = int(simple_config['preprocess']['img_size'])
        self.batch_size = int(simple_config['preprocess']['batch_size'])

        # Kiểm tra xem các đường dẫn có tồn tại hay không
        if not os.path.exists(self.base_dir):
            raise ValueError(f"Base directory {self.base_dir} không tồn tại.")
        if not os.path.exists(self.traindata_dir):
            raise ValueError(f"Training data directory {self.traindata_dir} không tồn tại.")

        print(f"Base directory: {self.base_dir}")
        print(f"Training data directory: {self.traindata_dir}")
        print(f"Image size: {self.img_size}")
        print(f"Batch size: {self.batch_size}")

    def data_augmentation(self):
        """
        Tạo một pipeline data augmentation với các phép biến đổi cơ bản.

        Returns:
            Sequential: Đối tượng Sequential chứa các lớp biến đổi dữ liệu.
        """
        data_augmentation = Sequential([
            layers.RandomFlip("horizontal", input_shape=(self.img_size, self.img_size, 3)),
            layers.RandomRotation(0.1),
            layers.RandomZoom(0.1)
        ])
        return data_augmentation

    def load_data(self, data_dir):
        """
        Tải dữ liệu huấn luyện và kiểm tra từ thư mục.

        Args:
            data_dir (str): Đường dẫn đến thư mục chứa dữ liệu.

        Returns:
            tuple: Trả về tập huấn luyện và kiểm tra.
        """
        # check số lượng dữ liệu trong tập datasets
        count = 0
        dirs = self.traindata_dir
        for dir in dirs:
            files = list(os.listdir(dirs + dir))
            print(dir + 'folder has' + str(len(files)) + 'Images')
            count += len(files)
        print('Images folder has ' + str(count) + ' Images') 


        train_ds = tf.keras.preprocessing.image_dataset_from_directory(
            data_dir,
            validation_split=0.2,
            subset="training",
            seed=123,
            image_size=(self.img_size, self.img_size),
            batch_size=self.batch_size
        )
        val_ds =  tf.keras.preprocessing.image_dataset_from_directory(
            data_dir,
            validation_split=0.2,
            subset="validation",
            seed=123,
            image_size=(self.img_size, self.img_size),
            batch_size=self.batch_size
        )

        # Tối ưu hóa hiệu suất đọc dữ liệu
        train_ds = train_ds.prefetch(buffer_size=tf.data.AUTOTUNE)
        val_ds = val_ds.prefetch(buffer_size=tf.data.AUTOTUNE)

        return train_ds, val_ds

    # def rescale_data(self):
    #     """
    #     Tạo lớp chuẩn hóa dữ liệu (Rescaling).
        
    #     Returns:
    #         Rescaling: Lớp Rescaling để chuẩn hóa giá trị pixel của ảnh.
    #     """
    #     return Rescaling(1./255)

# Kiểm tra nhanh module khi chạy trực tiếp file
if __name__ == "__main__":

    data = PreprocessingData()
    result = data.load_data()
    print(result)
