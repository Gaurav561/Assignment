import os
import tensorflow as tf
import cv2


def LoadDataset(folder_name : str):
    directory = "..\\dataset\\"+folder_name
    print(directory)
    image_files = sorted([os.path.join(directory, f) for f in os.listdir(directory) if f.__contains__("sat")])
    mask_files = sorted([os.path.join(directory, f) for f in os.listdir(directory) if f.__contains__("mask")])

    image_dataset = tf.data.Dataset.from_tensor_slices(image_files)
    image_dataset = image_dataset.map(parse_image)

    mask_dataset = tf.data.Dataset.from_tensor_slices(mask_files)
    mask_dataset = mask_dataset.map(parse_mask)

    dataset = tf.data.Dataset.zip((image_dataset, mask_dataset))
    dataset = dataset.batch(4).prefetch(tf.data.AUTOTUNE)

    return dataset



def parse_image(filename):
    image = tf.io.read_file(filename)
    image = tf.image.decode_jpeg(image, channels=3)
    # image = tf.image.resize(image, (128, 128))
    image = tf.cast(image, tf.float32) / 255.0
    return image

def parse_mask(filename):
    mask = tf.io.read_file(filename)
    mask = tf.image.decode_png(mask, channels=1)
    mask = cv2.threshold(mask, 128, 255, cv2.THRESH_BINARY)
    # mask = tf.image.resize(mask, (128, 128))    
    mask = tf.cast(mask, tf.float32) / 255.0
    return mask





