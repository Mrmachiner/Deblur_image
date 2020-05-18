import numpy as np
from PIL import Image
import click
import os
import cv2
import matplotlib.pyplot as plt

from deblurgan.model import generator_model
from deblurgan.utils import load_image, deprocess_image, preprocess_image, preprocess_image_no_resize

def deblur(weight_path, input_dir, output_dir):
	g = generator_model()
	g.load_weights(weight_path)
	for image_name in os.listdir(input_dir):
		# test = cv2.imread(input_dir+image_name)
		# img_original = load_image(os.path.join(input_dir, image_name))
		# img_original.show()
		img_tif = cv2.imread(input_dir + image_name)
		#img_tif = cv2.resize(img_tif,(256, 256))
		#img_tif_2 = Image.fromarray(img_tif)
		#img_tif.show()
		#image = np.array([preprocess_image(load_image(os.path.join(input_dir, image_name)))])
		image = np.array([preprocess_image_no_resize(img_tif)])
		x_test = image
		generated_images = g.predict(x=x_test)
		generated = np.array([deprocess_image(img) for img in generated_images])
		x_test = deprocess_image(x_test)
		for i in range(generated_images.shape[0]):
			x = x_test[i, :, :, :]
			img = generated[i, :, :, :]

			plt.figure()
			plt.imshow(img)
			plt.show()

			#output = np.concatenate((x, img), axis=1)
			img_gen = Image.fromarray(img.astype(np.uint8))
			img_gen.show()
			#im = Image.fromarray(output.astype(np.uint8))
			img_gen.save(os.path.join(output_dir, image_name))
			print("ok")
# @click.command()
# @click.option('--weight_path', default = "/home/minhhoang/Desktop/MinhHoang/ML_DL_inter/deblur-gan-master/Weight/Epoch100/210/generator_99_243.h5", help='Model weight')
# @click.option('--input_dir',default ="/home/minhhoang/Desktop/test1" , help='Image to deblur')
# @click.option('--output_dir',default= "/home/minhhoang/Desktop/testout", help='Deblurred image')
@click.command()
@click.option('--weight_path', default = "/home/minhhoang/Desktop/MinhHoang/ML_DL_inter/deblur-gan-master/weights/415/generator_249_201.h5", help='Model weight')
@click.option('--input_dir', default = "/home/minhhoang/Desktop/test_deblur/test/", help='Image to deblur')
#@click.option('--input_dir', default = "/home/minhhoang/Desktop/out/1", help='Image to deblur') #
@click.option('--output_dir', default = "/home/minhhoang/Desktop/out/", help='Deblurred image')
def deblur_command(weight_path, input_dir, output_dir):
	return deblur(weight_path, input_dir, output_dir)

if __name__ == "__main__":
	# img_tif = cv2.imread("/home/minhhoang/Desktop/out/43.tif")
	# img_tif = cv2.resize(img_tif,(256, 256))
	# print(img_tif.shape)
	deblur_command()
