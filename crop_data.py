import os
from PIL import Image
import numpy as np
import tensorflow as tf
import cv2
from scripts.measure.crop_graph import add_padding, crop_image

def is_an_image_file(filename):
    IMAGE_EXTENSIONS = ['.png', '.jpg', '.jpeg', '.tiff']
    for ext in IMAGE_EXTENSIONS:
        if ext in filename:
            return True
    return False

def load_image(path):
    img = cv2.imread(path)
    return img

def preprocess_image(cv_img, path, name_img, stt):
    # cv_img = cv_img.resize(RESHAPE)
    img_padding = add_padding(cv_img)
    lst_img_crop = crop_image(img_padding)
    for i in range(len(lst_img_crop)):
        cv2.imwrite(path+str(i)+"_"+name_img, lst_img_crop[i])
    print("save ok", stt)

def list_image_files(directory):
    files = sorted(os.listdir(directory))
    lst_paths = []
    for f in files:
        if is_an_image_file(f):
            path = os.path.join(directory, f) 
            lst_paths.append(path)
    return lst_paths, files

def load_images(path, n_images):
    if n_images < 0:
        n_images = float("inf")
    A_paths, B_paths = os.path.join(path, 'A'), os.path.join(path, 'B')

    A_crop_paths, B_crop_paths = os.path.join(path, 'A_crop/'), os.path.join(path, 'B_crop/')
    
    all_A_paths, name_A_paths = list_image_files(A_paths)
    all_B_paths, name_B_paths = list_image_files(B_paths)
    images_A, images_B = [], []
    images_A_paths, images_B_paths = [], []
    i = 0
    for path_A, path_B, name_A, name_B in zip(all_A_paths, all_B_paths, name_A_paths, name_B_paths):
        i = i+1
        img_A, img_B = load_image(path_A), load_image(path_B)
        preprocess_image(img_A, A_crop_paths, name_A, i)
        preprocess_image(img_B, B_crop_paths, name_B, i)
if __name__ == "__main__":
    data = load_images('./images/train', 5)