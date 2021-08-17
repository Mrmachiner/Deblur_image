import cv2
import pandas as pd
# from utils import load_images_score
from iqa.measure import IQA
import glob
import pandas as pd
def variance_of_laplacian(image):
    return cv2.Laplacian(image, cv2.CV_64F).var()

path_out_csv = "/home/minhhoang/Desktop/score_Img_valapcian/Folder_calculate_score/filecsv/"

def score_blur_image(paths, name_img=""):
	score = []
	for imagePath in paths:
		image = cv2.imread(imagePath)
		image = cv2.resize(image,(256,256))
		if name_img!="":
			name_img.append(imagePath) # add name img to file csv
		gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
		fm = variance_of_laplacian(gray)

		score.append(fm)
	return score

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
	path_gt = glob.glob("/home/minhhoang/Desktop/image_DA/sharp/*")
	path_ori = glob.glob("out/*")
	path_me = glob.glob("out_92/*")
	mssim_ori_gt = []
	mssim_me_gt = []

	psnr_ori_gt = []
	psnr_me_gt = []

	vifp_ori_gt = []
	vifp_me_gt = []

	name_img = []

	score_paper = []
	score_me = []

	score_paper = score_blur_image(path_ori)
	score_me = score_blur_image(path_me)


	dict = {'name_image': name_img, 'score_paper': score_paper, 'score_me': score_me}

	df = pd.DataFrame(dict) 
	
	# saving the dataframe 
	df.to_csv('./FileCSV/filesharp.csv') 
	# for idx in range(1): #len(path_gt)
	# 	print(idx)
		
	# 	img_gt = cv2.imread(path_gt[idx])
	# 	h, w, _ = img_gt.shape

	# 	img_ori = cv2.imread(path_ori[idx])
	# 	img_ori = img_ori[0:h, 0:w]

	# 	img_me = cv2.imread(path_me[idx])

	# 	print(path_gt[idx], path_ori[idx])
	# 	iqa_me = IQA(img_gt, img_me)
	# 	iqa_ori = IQA(img_gt, img_ori)
	# 	vifp_me_gt.append(iqa_me.vifp(()))
	# 	vifp_ori_gt.append(iqa_ori.vifp(()))

	# 	mssim_me_gt.append(iqa_me.msssim())
	# 	mssim_ori_gt.append(iqa_ori.msssim())

	# 	psnr_me_gt.append(iqa_me.psnr())
	# 	psnr_ori_gt.append(iqa_ori.psnr())
	
	# dict = {"vifp_me_gt": vifp_me_gt, 
	# 		"vifp_ori_gt": vifp_ori_gt, 
	# 		"mssim_me_gt": mssim_me_gt, 
	# 		"mssim_ori_gt": mssim_ori_gt, 
	# 		"psnr_me_gt": psnr_me_gt, 
	# 		"psnr_ori_gt": psnr_ori_gt
	# 		}
	# df = pd.DataFrame(dict)
	# df.to_csv("statisical.csv", header=True, index=True)
	# print("mssim_ori_gt", sum(mssim_ori_gt)/len(mssim_ori_gt))
	# print("mssim_me_gt", sum(mssim_me_gt)/len(mssim_me_gt))
	# print("psnr_ori_gt", sum(psnr_ori_gt)/len(psnr_ori_gt))
	# print("psnr_me_gt", sum(psnr_me_gt)/len(psnr_me_gt))
	# # score_img("/home/minhhoang/Desktop/score_Img_valapcian/Folder_calculate_score/Gen99","Gen99.csv")

