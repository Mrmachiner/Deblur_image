import os
from PIL import Image
import numpy as np
import tensorflow as tf
import cv2

RESHAPE = (256,256)

def is_an_image_file(filename):
    IMAGE_EXTENSIONS = ['.png', '.jpg', '.jpeg', '.tiff']
    for ext in IMAGE_EXTENSIONS:
        if ext in filename:
            return True
    return False


def list_image_files(directory):
    files = sorted(os.listdir(directory))
    return [os.path.join(directory, f) for f in files if is_an_image_file(f)]


def load_image(path):
    img = Image.open(path)
    return img


def preprocess_image(cv_img):
    cv_img = cv_img.resize(RESHAPE)
    img = np.array(cv_img)
    img = (img - 127.5) / 127.5
    return img

def preprocess_image_no_resize(cv_img):
    img = np.array(cv_img)
    img = (img - 127.5) / 127.5
    return img

def deprocess_image(img):
    img = img * 127.5 + 127.5
    return img.astype('uint8')


def save_image(np_arr, path):
    img = np_arr * 127.5 + 127.5
    im = Image.fromarray(img)
    im.save(path)


def load_images(path, n_images):
    if n_images < 0:
        n_images = float("inf")
    A_paths, B_paths = os.path.join(path, 'A'), os.path.join(path, 'B')
    all_A_paths, all_B_paths = list_image_files(A_paths), list_image_files(B_paths)
    images_A, images_B = [], []
    images_A_paths, images_B_paths = [], []
    for path_A, path_B in zip(all_A_paths, all_B_paths):
        img_A, img_B = load_image(path_A), load_image(path_B)
        images_A.append(preprocess_image(img_A))
        images_B.append(preprocess_image(img_B))
        images_A_paths.append(path_A)
        images_B_paths.append(path_B)
        if len(images_A) > n_images - 1: break

    return {
        'A': np.array(images_A),
        'A_paths': images_A_paths,
        'B': np.array(images_B),
        'B_paths': images_B_paths
    }
def load_images_score(path):
    blur_paths, deblur_paths, sharp_paths = os.path.join(path, 'img_blur'),os.path.join(path, 'img_deblur'), os.path.join(path, 'img_sharp')
    all_blur_paths, all_deblur_paths, all_sharp_paths = list_image_files(blur_paths), list_image_files(deblur_paths), list_image_files(sharp_paths)
    img_blur, img_deblur, img_sharp, name_img = [], [], [], []
    for blur_name, deblur_name, sharp_name in zip(all_blur_paths, all_deblur_paths, all_sharp_paths):
        img_blur.append(cv2.imread(blur_name))
        img_deblur.append(cv2.imread(deblur_name))
        img_sharp.append(cv2.imread(sharp_name))
        
        name_img.append(blur_name)
    return {
        'img_blur': np.array(img_blur),
        'name_img': np.array(name_img),
        'img_deblur': np.array(img_deblur),
        'img_sharp': np.array(img_sharp)
    }
def write_log(callback, names, logs, batch_no):
    """
    Util to write callback for Keras training
    """
    for name, value in zip(names, logs):
        summary = tf.Summary()
        summary_value = summary.value.add()
        summary_value.simple_value = value
        summary_value.tag = name
        callback.writer.add_summary(summary, batch_no)
        callback.writer.flush()
