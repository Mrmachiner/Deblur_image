import numpy as np
from PIL import Image
import click
import os
import cv2
import matplotlib.pyplot as plt
import math
import sys
import time
import torch
import torch.backends.cudnn as cudnn
from deblurgan.model import generator_model, generator_model_paper
from deblurgan.utils import load_image, deprocess_image, preprocess_image, preprocess_image_no_resize
from measure.crop_graph import crop_image, graph_image, add_padding
sys.path.append("yolov5/")
from models.experimental import attempt_load
from utils.torch_utils import select_device, load_classifier, time_synchronized
from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path, save_one_box
from utils.plots import colors, plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized
from utils.datasets import letterbox

class compareModel():
    def __init__(self, key = "me"):
        if key == "me":
            self.model = generator_model()
            self.model.load_weights("weights/generator_92_253.h5")
        elif key == "paper":
            self.model = generator_model_paper()
            self.model.load_weights("weights/generator.h5")
    def implement(self, image_original):
        self.image_original = image_original
        img_tif = self.image_original
        print(img_tif.shape)
        h = 4 * math.floor(img_tif.shape[0]/4)
        w = 4 * math.floor(img_tif.shape[1]//4)
        img_tif = cv2.resize(img_tif,(w, h))
        print(img_tif.shape)
        image = np.array([preprocess_image_no_resize(img_tif)])
        x_test = image
        generated_images = self.model.predict(x=x_test)
        generated = np.array([deprocess_image(img) for img in generated_images])
        x_test = deprocess_image(x_test)
        img = generated[0, :, :, :]
        img = cv2.resize(img,(w, h))
        return img
    def paper(self, image_original):
        lst_grap_img = []
        img_tif = image_original
        img_add_padding = add_padding(img_tif) # add padding image % 256
        lst_crop_img = crop_image(img_add_padding) # crop image size(256 x 256)
        for crop in range(len(lst_crop_img)):
            image_preprocess = np.array(lst_crop_img[crop]) 
            image_swap = (image_preprocess - 127.5) / 127.5
            image = np.array([image_swap])
            x_test = image
            generator_images = self.model.predict(x=x_test)
            generator = np.array([deprocess_image(img) for img in generator_images])
            x_test = deprocess_image(x_test)
            
            for i in range(generator_images.shape[0]):
                x = x_test[i, :, :, :]
                img = generator[i, :, :, :]
                im = Image.fromarray(img.astype(np.uint8))
                im_np = np.asarray(im)
                lst_grap_img.append(im_np)
                #im.save(os.path.join(output_dir, image_name))
        img_sharp = graph_image(lst_grap_img, img_add_padding)
        h, w = image_original.shape[0], image_original.shape[1]
        img_sharp = cv2.resize(img_sharp,(w, h))
        return img_sharp

class Yolo():
    def __init__(self, path_weight, conf):
        
        self.conf = conf
        self.device = "cpu"
        self.model = attempt_load(path_weight, map_location=select_device(self.device)) 
        
    def detect(self, image_original):
        self.image = image_original
        stride = int(self.model.stride.max())  # model stride
        imgsz = check_img_size(640, s=stride)  # check img_size
        names = self.model.module.names if hasattr(self.model, 'module') else self.model.names  # get class names
        
        half = False
        t0 = time.time()
        img = letterbox(self.image, imgsz, stride=32)[0]
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        img = np.ascontiguousarray(img)
        
        img = torch.from_numpy(img).to(self.device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        t1 = time_synchronized()
        pred = self.model(img, augment=False)[0]

        # Apply NMS
        pred = non_max_suppression(pred, self.conf, 0.5, classes=None, agnostic=False)
        t2 = time_synchronized()


        # Process detections
        for i, det in enumerate(pred):  # detections per image
            im0 = self.image
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            s = ""
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # Write results
                for *xyxy, conf, cls in reversed(det):# Add bbox to image
                    c = int(cls)  # integer class
                    label = f'{names[c]} {conf:.2f}'
                    plot_one_box(xyxy, im0, label=label, color=colors(c, True), line_thickness=3)

            # Print time (inference + NMS)
            print(f'{s}Done. ({t2 - t1:.3f}s)')

            # Stream results
            return im0, s

if __name__ == "__main__":
    img = cv2.imread("out_92/000001.png")
    compare_me = compareModel(img)
    comaper_paper = compareModel(img, key="paper")
    yolo = Yolo(img, "yolov5/yolov5s.pt", 0.25)
    a = compare_me.implement()
    b = comaper_paper.paper()
    
    a = yolo.detect()
    # cv2.imshow("abcdb",b)
    cv2.imshow("abcd",a)
    cv2.waitKey()
    