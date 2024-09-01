import numpy as np
import h5py #A library to interact with HDF5 files, which are often used to store large datasets.
import os
from tensorflow.keras.utils import Sequence

# Data loading and preprocessing 
class ImageMaskGenerator(Sequence):
    def __init__(self, image_files, batch_size, dataset_path):
        self.image_files = image_files
        self.batch_size = batch_size
        self.dataset_path = dataset_path

    def __len__(self):
        return int(np.ceil(len(self.image_files) / float(self.batch_size)))

    def __getitem__(self, idx):
        batch_x = self.image_files[idx * self.batch_size:(idx + 1) * self.batch_size]
        images = []
        masks = []
        for file_name in batch_x:
            img_path = os.path.join(self.dataset_path, file_name)
            if not os.path.isfile(img_path):
                print(f"Warning: {img_path} does not exist or is not a file.")
                continue
            
            try:
                with h5py.File(img_path, 'r') as file:
                    images.append(np.expand_dims(np.dot(np.array(file['image'])[...,:3], [0.2989, 0.5870, 0.1140]), -1))
                    masks.append(np.expand_dims(np.dot(np.array(file['mask'])[...,:3], [0.2989, 0.5870, 0.1140]), -1))
            except (OSError, KeyError) as e:
                print(f"Error opening {img_path}: {e}")
                continue

        return np.array(images), np.array(masks)