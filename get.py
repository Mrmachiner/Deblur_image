import os
import glob
from shutil import copyfile
import shutil

path = "/media/minhhoang/Data/DataTu/test/*"
save_img = "/home/minhhoang/Desktop/image_DA/"
lst_path = glob.glob(path)
for p in lst_path:
    _p = p + "/*"
    folders = glob.glob(_p)
    for f in folders:
        check = f.split("/")[-1]
        if check == "blur_gamma":
            continue
        lst_img = glob.glob(f+"/*")
        for p_img in lst_img:
            p_save = save_img + check + "/" + p_img.split("test")[-1].replace("/", "_")
            shutil.copy(p_img, p_save)
    print(folders)