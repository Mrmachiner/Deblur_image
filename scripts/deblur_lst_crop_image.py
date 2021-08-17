import numpy as np
from PIL import Image
import click
import os
import cv2
from deblurgan.model import generator_model_paper
from deblurgan.utils import deprocess_image, preprocess_image, load_images_score
from measure.crop_graph import crop_image, graph_image, add_padding
# from measure.score_image import score_blur_image

def deblur(weight_path, input_dir, output_dir):
	g = generator_model_paper()
	g.load_weights(weight_path)
	lst_grap_img = []
	lst_crop_img = [] 
	count = 0
	if not os.path.exists(output_dir):
		os.mkdir(output_dir)
	for image_name in os.listdir(input_dir):
		path_in = input_dir+"/"+image_name  # path image input
		path_out = output_dir+"/"+image_name # path image output
		image_blur = cv2.imread(path_in) # read image
		img_add_padding = add_padding(image_blur) # add padding image % 256
		lst_crop_img = crop_image(img_add_padding) # crop image size(256 x 256)
		for crop in range(len(lst_crop_img)):
			image_preprocess = np.array(lst_crop_img[crop]) 
			image_swap = (image_preprocess - 127.5) / 127.5
			image = np.array([image_swap])
			x_test = image
			generator_images = g.predict(x=x_test)
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
		cv2.imwrite(path_out,img_sharp)
		count +=1
		print("done", count/len(os.listdir(input_dir)))
		lst_grap_img.clear()
		lst_crop_img.clear()

@click.command()
@click.option('--weight_path', default = "weights/generator.h5", 
		help='Model weight')
@click.option('--input_dir', default = "/home/minhhoang/Desktop/abcd/", 
		help='Image to deblur')
@click.option('--output_dir', default = "/home/minhhoang/Desktop/abcde/", 
		help='Deblurred image')

def deblur_command(weight_path, input_dir, output_dir):
	return deblur(weight_path,input_dir,output_dir)

if __name__ == "__main__":
	deblur_command()
