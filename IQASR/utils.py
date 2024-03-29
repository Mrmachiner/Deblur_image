import numpy as np
import warnings
from scipy import signal
from enum import Enum
from PIL import Image
import cv2
from math import log2, log10
class Filter(Enum):
	UNIFORM = 0
	GAUSSIAN = 1
def _initial_check(GT, P):
	assert GT.shape == P.shape, "Supplied images have different sizes" + \
		str(GT.shape) + "and" + str(P.shape)
	if GT.dtype != P.dtype:
		msg = "Supplied images have different dtypes" + \
			str(GT.dtype) + "and" + str(P.dtype)
		warnings.warn(msg)
	if len(GT.shape) == 2:
		GT = GT[:, :, np.newaxis]
		P = P[:, :, np.newaxis]
	return GT.astype(np.float64), P.astype(np.float64)

def fspecial(fltr, ws, **kwargs):
	if fltr == Filter.UNIFORM:
		return np.ones((ws, ws))/ ws ** 2
	elif fltr == Filter.GAUSSIAN:
		x, y = np.mgrid[-ws//2 + 1:ws//2 + 1, -ws//2 + 1:ws//2 + 1]
		g = np.exp(-((x ** 2 + y ** 2)/(2.0 * kwargs['sigma'] ** 2)))
		g[ g < np.finfo(g.dtype).eps * g.max() ] = 0
		assert g.shape == (ws, ws)
		den = g.sum()
		if den !=0:
			g/=den
		return g
	return None

def _get_sums(GT,P,win,mode='same'):
	mu1,mu2 = (filter2(GT,win,mode),filter2(P,win,mode))
	return mu1 * mu1, mu2 * mu2, mu1 * mu2

def _get_sigmas(GT,P,win,mode='same',**kwargs):
	if 'sums' in kwargs:
		GT_sum_sq,P_sum_sq,GT_P_sum_mul = kwargs['sums']
	else:
		GT_sum_sq,P_sum_sq,GT_P_sum_mul = _get_sums(GT, P, win, mode)

	return filter2(GT * GT,win,mode)  - GT_sum_sq,\
			filter2(P * P,win,mode)  - P_sum_sq, \
			filter2(GT * P,win,mode) - GT_P_sum_mul

def filter2(img,fltr,mode='same'):
	return signal.convolve2d(img, np.rot90(fltr,2), mode=mode)

def _power_complex(a,b):
	return a.astype('complex') ** b

def _replace_value(array,value,replace_with):
	array[array == value] = replace_with
	return array


def _compute_bef(im, block_size=8):
	"""Calculates Blocking Effect Factor (BEF) for a given grayscale/one channel image

	C. Yim and A. C. Bovik, "Quality Assessment of Deblocked Images," in IEEE Transactions on Image Processing,
		vol. 20, no. 1, pp. 88-98, Jan. 2011.

	:param im: input image (numpy ndarray)
	:param block_size: Size of the block over which DCT was performed during compression
	:return: float -- bef.
	"""
	if len(im.shape) == 3:
		height, width, channels = im.shape
	elif len(im.shape) == 2:
		height, width = im.shape
		channels = 1
	else:
		raise ValueError("Not a 1-channel/3-channel grayscale image")

	if channels > 1:
		raise ValueError("Not for color images")

	h = np.array(range(0, width - 1))
	h_b = np.array(range(block_size - 1, width - 1, block_size))
	h_bc = np.array(list(set(h).symmetric_difference(h_b)))

	v = np.array(range(0, height - 1))
	v_b = np.array(range(block_size - 1, height - 1, block_size))
	v_bc = np.array(list(set(v).symmetric_difference(v_b)))

	d_b = 0
	d_bc = 0

	# h_b for loop
	for i in list(h_b):
		diff = im[:, i] - im[:, i + 1]
		d_b += np.sum(np.square(diff))

	# h_bc for loop
	for i in list(h_bc):
		diff = im[:, i] - im[:, i + 1]
		d_bc += np.sum(np.square(diff))

	# v_b for loop
	for j in list(v_b):
		diff = im[j, :] - im[j + 1, :]
		d_b += np.sum(np.square(diff))

	# V_bc for loop
	for j in list(v_bc):
		diff = im[j, :] - im[j+1, :]
		d_bc += np.sum(np.square(diff))

	# N code
	n_hb = height * (width / block_size) - 1
	n_hbc = (height * (width - 1)) - n_hb
	n_vb = width * (height / block_size) - 1
	n_vbc = (width * (height - 1)) - n_vb

	# D code
	d_b /= (n_hb + n_vb)
	d_bc /= (n_hbc + n_vbc)

	# Log
	if d_b > d_bc:
		t = log2(block_size) / log2(min(height, width))
	else:
		t = 0

	# BEF
	bef = t * (d_b - d_bc)

	return bef

def filter2(img,fltr,mode='same'):
	return signal.convolve2d(img, np.rot90(fltr,2), mode=mode)

def imresize(arr, size):
	arr_sw = cv2.resize(arr, size)
	return np.array(arr_sw)
if __name__ == "__main__":
	img = Image.open("/home/minhhoang/Desktop/sewar-master/sewar/tests/res/lena512color_constant.tiff")
	abcd = np.array(img.resize((123,123)))
	print(abcd.shape)