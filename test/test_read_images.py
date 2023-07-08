import sys
sys.path.append('../src')
import pytest
import numpy as np
from pathlib import Path
from read_images import create_image_dataset
import os
image_folder = 'resources/EuroSAT_RGB'

@pytest.fixture
def dataset_obj():
    return create_image_dataset(os.path.abspath(image_folder))


def test_init(dataset_obj):
    assert dataset_obj.image_dataset==[]
    assert dataset_obj.image_labels==[]
    assert dataset_obj.images_folder==image_folder

def test_read_image_and_convert_to_array(dataset_obj):
    image_path = os.path.abspath('resources/EuroSAT_RGB/AnnualCrop/AnnualCrop_1.jpg')
    img_array = dataset_obj.read_image_and_convert_to_array(image_path)

    assert isinstance(img_array,np.ndarray)
    assert img_array.shape == (64,64,3)
    assert np.min(img_array)>=0.0
    assert np.max(img_array)<=1.0

def test_read_all_images_in_folder(dataset_obj):
    images, labels = dataset_obj.read_all_images_in_folder(Path('resources/EuroSAT_RGB/AnnualCrop'),0)

    assert isinstance(images,list)
    assert isinstance(labels,list)
    assert len(images)==2
    assert len(labels)==2

    assert all(lab==0 for lab in labels)

def test_read_create_dataset(dataset_obj):
    img_array, lab_array = dataset_obj.create_dataset()
    assert len(dataset_obj.image_dataset) == 20
    assert len(dataset_obj.image_labels) == 20
    assert isinstance(img_array,np.ndarray)
    assert isinstance(lab_array,np.ndarray)
    assert img_array.shape == (20,64,64,3)
    assert lab_array.shape == (20,)

def test_load_dataset(dataset_obj):
    dataset_obj.create_dataset = lambda:(np.random.random((20,64,64,3)),np.random.randint(0,10,(20,)))
    train_data, test_data, train_labels, test_labels = dataset_obj.load_dataset(testSize=0.2)

    assert train_data.shape == (16,64,64,3)
    assert test_data.shape == (4,64,64,3)
    assert train_labels.shape == (16,)
    assert test_labels.shape == (4,)


