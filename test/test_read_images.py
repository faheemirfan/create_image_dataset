import sys
sys.path.append('../src')
import numpy as np
from pathlib import Path
from read_images import create_image_dataset

image_folder = 'resources/EuroSAT_RGB'

def test_init():
    dataset_obj = create_image_dataset(image_folder)
    assert dataset_obj.image_dataset==[]
    assert dataset_obj.image_labels==[]
    assert dataset_obj.images_folder==image_folder

def test_read_image():
    dataset_obj = create_image_dataset(image_folder)
    image_path = 'resources/EuroSAT_RGB/AnnualCrop/AnnualCrop_1.jpg'
    img_array = dataset_obj.read_image_and_convert_to_array(image_path)

    assert isinstance(img_array,np.ndarray)
    assert img_array.shape == (64,64,3)
    assert np.min(img_array)>=0.0
    assert np.max(img_array)<=1.0

def test_read_all_images_in_folder():
    dataset_obj = create_image_dataset(image_folder)
    images, labels = dataset_obj.read_all_images_in_folder(Path('resources/EuroSAT_RGB/AnnualCrop'),0)

    assert isinstance(images,list)
    assert isinstance(labels,list)
    assert len(images)==2
    assert len(labels)==2

    assert all(lab==0 for lab in labels)