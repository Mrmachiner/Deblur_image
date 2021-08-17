import cv2
import numpy as np
import os
import glob
from random import sample
x = 0
y = 0
def add_padding(img):  #add padding image h and w % 256
    right, bot = 0, 0 #add to right and bot
    if (img.shape[1] % 256) != 0:
        right = ((img.shape[1]//256) + 1) * 256 - img.shape[1]
    if (img.shape[0] % 256) != 0:
        bot = ((img.shape[0]//256) + 1) * 256 - img.shape[0]
    imgPadding = cv2.copyMakeBorder(img, 0, bot, 0, right, cv2.BORDER_CONSTANT)
    return imgPadding
    
def crop_image(imgPadding): # crop img_original to many img 256x256 return lst_img
    global x, y
    vt = 0
    lst_img = []
    for i in range(int(imgPadding.shape[0]/256)):
        for j in range(int(imgPadding.shape[1]/256)):
            img_crop = imgPadding[x:x+256, y:y+256, :]
            lst_img.append(img_crop)
            #cv2.imwrite("/home/minhhoang/Desktop/test1/abcd/img_crop"+str('{0:04}'.format(vt))+".jpg",img_crop)
            y += 256
            vt +=1
        y = 0
        x += 256
    x = 0
    y = 0
    return lst_img


def random_index_batch(batch_size, leg):
    index = [i for i in range(leg)]
    index_random = sample(index, batch_size)
    return index_random

def random_image2array_batch(index_random, lst_img):
    lst_img2array_batch = []
    for i in index_random:
        lst_img2array_batch.append(lst_img[i])
    return lst_img2array_batch

def crop_image_to_array(imgPadding, batch_size): # crop img_original to many img 256x256 return lst_img
    global x, y
    vt = 0
    lst_img = []
    lst_img2array_batch = []
    w = int(imgPadding.shape[1]/256)
    h = int(imgPadding.shape[0]/256)
    leg = w * h
    index_random = random_index_batch(batch_size, leg)
    for i in range(int(imgPadding.shape[0]/256)):
        for j in range(w):
            img_crop = imgPadding[x:x+256, y:y+256, :]
            img = np.array(img_crop)
            img = (img - 127.5) / 127.5
            lst_img.append(img)
            y += 256
            vt +=1
        y = 0
        x += 256
    x = 0
    y = 0
    lst_img2array_batch = random_image2array_batch(index_random, lst_img)
    return lst_img2array_batch

def graph_image(path_image, imgPadding):
    lst_img = []
    mask = np.zeros(imgPadding.shape, np.uint8)
    global x, y
    # for image_name in sorted(os.listdir(path_image), key=id):
    #     img = cv2.imread(os.path.join(path_image, image_name))
    #     if img is not None:
    #         lst_img.append(img)
    for img_path in sorted(glob.glob(path_image)):
        img = cv2.imread(img_path)
        if img is not None:
            lst_img.append(img)
    vt = 0
    for i in range(int(imgPadding.shape[0]/256)):
        for j in range(int(imgPadding.shape[1]/256)):
            imgSwap = lst_img[vt]
            mask[x:x+256, y:y+256, :] = imgSwap[0:256, 0:256,:]
            cv2.imshow("lst_img[x]",lst_img[vt])
            vt += 1
            y += 256
            # cv2.imshow("img_crop" + str(j) + str(i),mask)
            # cv2.waitKey()
        y = 0
        x += 256
    return mask
def graph_image(lst_img, image): #graph many img 256x256 to img
    vt = 0
    mask = np.zeros(image.shape, np.uint8)
    global x, y
    for i in range(int(image.shape[0]/256)):
        for j in range(int(image.shape[1]/256)):
            imgSwap = lst_img[vt]
            mask[x:x+256, y:y+256, :] = imgSwap[0:256, 0:256,:]
            #cv2.imshow("lst_img[x]",lst_img[vt])
            vt += 1
            y += 256
        y = 0
        x += 256
    x = 0
    y = 0
    return mask
if __name__ == "__main__":
    img = cv2.imread("/home/minhhoang/Desktop/test1/Img/000001.png")
    img_padd = add_padding(img)
    cv2.imshow("img", img)
    # cv2.waitKey()
    # cv2.destroyAllWindows()
    lst_img = crop_image(img)
    graph_image(lst_img,img_padd)

    # w, h = imgPadding.shape[0], imgPadding.shape[1]
    # mask = np.zeros((w, h, 3), np.uint8)
    # print(mask.shape)
    # path_img = "/home/minhhoang/Desktop/test1/abcd/*.jpg"
    # img_graph = graph_image(path_img,imgPadding)
    # cv2.imshow("img_graph", img_graph)
    # cv2.waitKey()