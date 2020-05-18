import cv2
from imutils import paths
import argparse
import pandas as pd
from scripts.deblurgan.utils import load_images_score
def variance_of_laplacian(image):
    return cv2.Laplacian(image,cv2.CV_64F).var()
path_out_csv = "/home/minhhoang/Desktop/score_Img_valapcian/Folder_calculate_score/filecsv/"

def score_blur_image(path, score, name_img=""):
	for imagePath in paths.list_images(path):
		image = cv2.imread(imagePath)
		image = cv2.resize(image,(256,256))
		if name_img!="":
			name_img.append(imagePath) # add name img to file csv
		gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
		fm = variance_of_laplacian(gray)

		score.append(fm)
def score_img(path, name_csv):
	load_file = load_images_score(path)
	img_blur, img_deblur, img_sharp, img_name = load_file['img_blur'], load_file['img_deblur'], load_file['img_sharp'], load_file['name_img']
	
	score_blur, score_deblur, score_sharp = [], [], []
	for i in range(len(img_blur)) :
		# cv2.imshow("ancd",img_blur[i])
		# cv2.waitKey()
		gray_b = cv2.cvtColor(img_blur[i], cv2.COLOR_BGR2GRAY)
		fm_b = variance_of_laplacian(gray_b)

		gray_d = cv2.cvtColor(img_deblur[i], cv2.COLOR_BGR2GRAY)
		fm_d = variance_of_laplacian(gray_d)

		gray_s = cv2.cvtColor(img_sharp[i], cv2.COLOR_BGR2GRAY)
		fm_s = variance_of_laplacian(gray_s)

		score_blur.append(fm_b)
		score_deblur.append()
		score_sharp.append(fm_s)
	dict = {'name_image': img_name, 'score_Sharp': score_sharp, 
			'score_Blur': score_blur, 'score_GAN_sharp':score_deblur}
	df = pd.DataFrame(dict) 
	df.to_csv(path_out_csv+name_csv) 
	print("Done")
if __name__ == "__main__":
	score_img("/home/minhhoang/Desktop/score_Img_valapcian/Folder_calculate_score/Gen99","Gen99.csv")
	# name_img = []
	# score_sharp = []
	# score_gan_sharp = []
	# score_blu = []
	# score_blur_image(path_blur,score_blu,name_img)

	# score_blur_image(path_sharp,score_sharp)

	# score_blur_image(path_gan_sharp,score_gan_sharp)

	# dict = {'name_image': name_img, 'score_Sharp': score_sharp, 'score_Blur': score_blu, 'score_GAN_sharp':score_gan_sharp}

	# df = pd.DataFrame(dict) 
	
	# # saving the dataframe 
	# df.to_csv('./FileCSV/filesharp.csv') 
