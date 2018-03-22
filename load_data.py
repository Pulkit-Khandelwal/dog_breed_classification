"""
This module helps to load the data.
The data is organized as follows
/data/
     /images
     /annotation
     /lists
          /file_list.mat
          /train_list.mat
          /test_list.mat
Returns:
    Pandas Dataframes to load and process data.
    This DataFrame contains all the necessary information regarding the metadata
    of each image of the dataset. This serves as a nice representation to go over the data at hand.
    At run time we can create three such dataframes one each for full_dataset, train_dataset and
    test_dataset

    Here are the headers of the DataFrame:

"""

import scipy.io
import numpy as np
import pandas as pd
import xml.etree.ElementTree as ET
import pickle
from skimage import data, io
from skimage.transform import rescale, resize, downscale_local_mean
from skimage import data, io
from skimage.util import img_as_ubyte
import matplotlib.pyplot as plt


def load_matfile(file_path=None):
    """
    Extracts the mat files and returns three lists per mat file viz.
        lists of filenames, labels and annotation list

    Args:file_path: path of the file (string)
    Returns:
            filename_list
            labels_list
            annotation_list

    Example of how each entry in the list looks like:
            filename_list = ....... 'n02116738_9829.jpg'
            labels_list = .......   120
            annotation_list = ...... 'n02116738_9829'

    """

    filename_list = scipy.io.loadmat(file_path)['file_list']
    filename_list = [x[0][0] for x in filename_list]
    filename_list = [x.split('/') for x in filename_list]
    filename_list = [x[1] for x in filename_list]

    labels_list = scipy.io.loadmat(file_path)['labels']
    labels_list= [x[0] for x in labels_list]

    annotation_list = scipy.io.loadmat(file_path)['annotation_list']
    annotation_list = [x[0][0] for x in annotation_list]
    annotation_list = [x.split('/') for x in annotation_list]
    annotation_list = [x[1] for x in annotation_list]

    return filename_list, labels_list, annotation_list


def create_dataframe(filename_list=None, labels_list=None, annotation_list=None, annotation_path=None):
    """
    Takes the lists returned by the method load_matfile() for a given set of dataset.
    Lops through the data and spits out a dataframe with all the metadata intact.

    Args:
        filename_list
        labels_list
        annotation_list
        data_path(String): Location of the directory where the annotations are present viz.
                            /data
                                /annotation

    Returns:
            data_frame

    Example: An example of the first two rows for the test dataset

     folder  label          image_name      annotation width height depth  \
    02085620      1  n02085620_2650.jpg  n02085620_2650   500    333     3
    02085620      1  n02085620_4919.jpg  n02085620_4919   240    206     3

        name  xmin ymin xmax ymax
    Chihuahua  108   87  405  303
    Chihuahua   10   22  180  205

    """

    data_frame = pd.DataFrame(columns=['folder', 'label'])

    data_frame['image_name'] = filename_list
    data_frame['label'] = labels_list
    data_frame['annotation'] = annotation_list

    for index, file in data_frame.iterrows():
        "Creating a nice dataframe!"
        if index % 100 == 0:
            print(index)
        full_path = annotation_path + file['annotation']
        _parser(data_frame, full_path, index)

    return data_frame


def _parser(data_frame, full_path, index):
    """
    Internal method to parse the xml files
    Args:
        data_frame: dataframe to be populated
        full_path: path of the annotation xml file
        index: integer to loop through the dataframe
    Returns:
            data_frame (pandas dataframe)
    """
    tree = ET.parse(full_path)
    root = tree.getroot()
    for child in root:
        if child.tag == 'folder':
            data_frame.loc[index, 'folder'] = child.text

        if child.tag == 'size':
            for object_iter in child:
                data_frame.loc[index, object_iter.tag] = object_iter.text

        if child.tag == 'object':
            for object_iter in child:
                if object_iter.tag == 'name':
                    data_frame.loc[index, object_iter.tag] = object_iter.text
                if object_iter.tag == 'bndbox':
                    for it in object_iter:
                        data_frame.loc[index, it.tag] = it.text


def to_numpy_array(data_frame=None, image_shape=(224, 224), data_path=None):
    """
    Converts the images and store them as numpy array

    Args:
        image_shape (tuple()): (224,224) or (299,299)
                                for different networks

        data_frame: dataframe with all the metadata
        data_path: path to the images

    Returns:
        data_numpy_array

    """
    data_numpy_array = np.zeros((len(data_frame), image_shape[0], image_shape[1], 3))

    print('Cropping, resizing and saving as a numpy array')
    for index, row in data_frame.iterrows():
        if index % 100 == 0:
            print(index)
        image_path = data_path + row['image_name']
        img = io.imread(image_path)
        img_crop = img[int(row['ymin']):int(row['ymax']), int(row['xmin']):int(row['xmax']), :3]
        img_resize = resize(img_crop, (image_shape[0], image_shape[1]))
        data_numpy_array[index] = img_resize

    return data_numpy_array


def labels_to_logical(labels_list=None):
    """
    Converts integer labels to one-shot encoding.
    This is required to train our models. Since,
    we predict using categorical cross entropy

    :param labels_list:
    :return:
    """

    numpy_labels = np.array(labels_list)
    numpy_labels_logical = np.zeros((len(numpy_labels), 120))

    for i in range(len(labels_list)):
        index = labels_list[i]
        print(index)
        numpy_labels_logical[i][index-1] = 1

    return numpy_labels_logical


if __name__ == '__main__':
    filename_list, labels_list, annotation_list = load_matfile('/Users/pulkit/Desktop/dog_breed_classifier/data/lists/test_list.mat')
    print(filename_list)
    print(len(annotation_list))

    data_frame = create_dataframe(filename_list[:200], labels_list[:200], annotation_list[:200],
                                  '/Users/pulkit/Desktop/dog_breed_classifier/data/annotation/')

    numpy_labels_logical = labels_to_logical(labels_list[:200])

    print(labels_list[195:200])
    print(numpy_labels_logical[195:200])
    print(len(numpy_labels_logical))


    """
    print(data_frame[:3])
    print(len(data_frame))

    train_numpy = to_numpy_array(data_frame[:200], image_shape=(224, 224), data_path='/Users/pulkit/Desktop/dog_breed_classifier/data/images/')
    print(train_numpy[:5])
    print(len(train_numpy))
    print(np.shape(train_numpy))
    
    """
