import cv2 
import numpy as np 
from scipy.ndimage import uniform_filter, generic_laplace, correlate
from math import log2, log10
from iqa.utils import Help, Filter
import cv2

class IQA:
	def __init__(self, gt, p):
	 gt, p = Help()._initial_check(gt, p)
	 self.gt = gt 
	 self.p = p
	def mse (self):
		"""Calculates mean squared error (mse).

		:param GT: first (original) input image
		:paran P: second (deformed) input image

		:returns: float -- mse value
		"""
		return np.mean((self.gt.astype(np.float64) - self.p.astype(np.float64)) ** 2)

	def psnr (self, MAX = None):
		"""Calculates peak signal-to-noise ratio (psnr)

		:param GT: frist (orginal) input image.
		:param P: second (deformed) input image.
		:param MAX: maximum value of datarange (if None, MAX is calculated using image dtype)

		return: float -- psnr value in dB.
		"""
		if MAX is None:
			MAX = np.amax([np.amax(self.gt[:,:,0]), np.amax(self.gt[:,:,1]), np.amax(self.gt[:,:,2])])
		mse_value = self.mse()
		if mse_value == 0:
			return np.inf
		return 10 * np.log10(MAX ** 2 / mse_value)

	def _ssim_single (self, gt, p, ws, C1, C2, fltr_specs, mode):
		win = Help().fspecial(**fltr_specs)

		GT_sum_sq, P_sum_sq, GT_P_sum_mul = Help()._get_sums(gt, p, win, mode)
		sigmaGT_sq, sigmaP_sq, sigmaGT_P = Help()._get_sigmas(gt, p, win, mode, sums=(GT_sum_sq, P_sum_sq, GT_P_sum_mul))

		assert C1 > 0
		assert C2 > 0

		ssim_map = ((2 * GT_P_sum_mul + C1) * (2 * sigmaGT_P + C2))/((GT_sum_sq + P_sum_sq + C1) * (sigmaGT_sq + sigmaP_sq + C2))
		cs_map = (2 * sigmaGT_P + C2) / (sigmaGT_sq + sigmaP_sq + C2)

		
		return np.mean(ssim_map), np.mean(cs_map)

	def ssim (self, ws = 11, K1 = 0.03, K2 = 0.03, MAX = None, fltr_specs = None, mode = 'valid'):
		"""Calculates structural similarity index (ssim).

		:param GT: first (original) input image
		:param P: second (deformed) input image
		:param ws: sliding window size (default = 11)
		:param K1: first constant for SSIM (default = 0.01)
		:param K2: second constant for SSIM (default = 0.03)
		:param MAX: Maximum value of datarange (if None, MAX is calculated using image dtype).

		:returns:  tuple -- ssim value, cs value.
		"""
		if MAX is None:
			MAX = np.amax([np.amax(self.gt[:,:,0]), np.amax(self.gt[:,:,1]), np.amax(self.gt[:,:,2])])
			# MAX = np.amax([np.amax(self.gt[:,:,0]), np.amax(self.gt[:,:,1]), np.amax(self.gt[:,:,2])])
		

		if fltr_specs is None:
			fltr_specs = dict(fltr = Filter.UNIFORM, ws = ws)
		C1 = (K1 * MAX) ** 2
		C2 = (K2 * MAX) ** 2
		
		ssims = []
		css = []
		for i in range(self.gt.shape[2]):
			ssim, cs = self._ssim_single(self.gt[:, :, i], self.p[:, :, i], ws, C1, C2, fltr_specs, mode)
			ssims.append(ssim)
			css.append(cs)
		return np.mean(ssims), np.mean(css)

	def msssim (self, weights = [0.0448, 0.2856, 0.3001, 0.2363, 0.1333], ws=11, K1=0.01, K2=0.03, MAX=None):
		"""calculates multi-scale structural similarity index (ms-ssim).

		:param GT: first (original) input image.
		:param P: second (deformed) input image.
		:param weights: weights for each scale (default = [0.0448, 0.2856, 0.3001, 0.2363, 0.1333]).
		:param ws: sliding window size (default = 11).
		:param K1: First constant for SSIM (default = 0.01).
		:param K2: Second constant for SSIM (default = 0.03).
		:param MAX: Maximum value of datarange (if None, MAX is calculated using image dtype).

		:returns:  float -- ms-ssim value.
		"""
		if MAX is None:
			MAX = np.amax([np.amax(self.gt[:,:,0]), np.amax(self.gt[:,:,1]), np.amax(self.gt[:,:,2])])
		

		scales = len(weights)

		fltr_specs = dict(fltr = Filter.GAUSSIAN, sigma = 1.5, ws = 11)

		if isinstance(weights, list):
			weights = np.array(weights)

		mssim = []
		mcs = []
		for _ in range(scales):
			_ssim, _cs = self.ssim(ws = ws, K1 = K1, K2 = K2, MAX = MAX, fltr_specs = fltr_specs)
			mssim.append(_ssim)
			mcs.append(_cs)

			filtered = [uniform_filter(im, 2) for im in [self.gt, self.p]]
			gt, P = [x[::2, ::2, :] for x in filtered]
		mssim = np.array(mssim, dtype = np.float64)
		mcs = np.array(mcs, dtype = np.float64)
		return np.prod(Help()._power_complex(mcs[:scales-1],weights[:scales-1])) * Help()._power_complex(mssim[scales-1],weights[scales-1])

	def rmse (self):
		"""Calculates root mean squared error (rmse).

		:param GT: first (original) input image
		:paran P: second (deformed) input image

		:returns: float -- rmse value
		"""
		return np.sqrt(mse(self.gt, self.p))

	def _uqi_single (self, ws):
		N = ws ** 2
		window = np.ones((ws, ws))

		GT_sq = self.gt * self.gt
		P_sq = self.p * self.p
		GT_P = self.gt * self.p
		
		GT_sum = uniform_filter(self.gt, ws)    
		P_sum =  uniform_filter(self.p, ws)     
		GT_sq_sum = uniform_filter(GT_sq, ws)  
		P_sq_sum = uniform_filter(P_sq, ws)  
		GT_P_sum = uniform_filter(GT_P, ws)

		GT_P_sum_mul = GT_sum * P_sum
		GT_P_sum_sq_sum_mul = GT_sum * GT_sum + P_sum * P_sum
		numerator = 4 * (N * GT_P_sum - GT_P_sum_mul) * GT_P_sum_mul
		denominator1 = N * (GT_sq_sum + P_sq_sum) - GT_P_sum_sq_sum_mul
		denominator = denominator1 * GT_P_sum_sq_sum_mul

		q_map = np.ones(denominator.shape)
		index = np.logical_and((denominator1 == 0) , (GT_P_sum_sq_sum_mul != 0))
		q_map[index] = 2 * GT_P_sum_mul[index] / GT_P_sum_sq_sum_mul[index]
		index = (denominator != 0)
		q_map[index] = numerator[index] / denominator[index]

		s = int(np.round(ws / 2))
		return np.mean(q_map[s : -s, s : -s])

	def uqi (self, ws = 8):
		"""calculates universal image quality index (uqi).

		:param GT: first (original) input image.
		:param P: second (deformed) input image.
		:param ws: sliding window size (default = 8).

		:returns:  float -- uqi value.
		"""
		return np.mean([_uqi_single(self.gt[:,:,i], self.p[:,:,i],ws) for i in range(self.gt.shape[2])])

	def _rmse_sw_single (self, ws):
		errors = (self.gt - self.p) ** 2
		errors = uniform_filter(errors.astype(np.float64), ws)
		rmse_map = np.sqrt(errors)
		s = int(np.round((ws / 2)))
		return np.mean(rmse_map[s : -s, s : -s]), rmse_map

	def rmse_sw (self, ws = 8):
		"""calculates root mean squared error (rmse) using sliding window.

		:param GT: first (original) input image.
		:param P: second (deformed) input image.
		:param ws: sliding window size (default = 8).

		:returns:  tuple -- rmse value,rmse map.	
		"""

		rmse_map = np.zeros(self.gt.shape)
		vals = np.zeros(self.gt.shape[2])
		for i in range (self.gt.shape[2]):
			vals[i], rmse_map[:, :, i] = _rmse_sw_single (self.gt[:, :, i], self.p[:, :, i], ws)
		return np.mean(vals), rmse_map

	def rase (self, ws = 8):
		"""calculates relative average spectral error (rase).

		:param GT: first (original) input image.
		:param P: second (deformed) input image.
		:param ws: sliding window size (default = 8).

		:returns:  float -- rase value.
		"""
		_, rmse_map = rmse_sw(self.gt, self.p, ws)
		GT_means = uniform_filter(self.gt, ws) / ws ** 2

		N = self.gt.shape[2]
		M = np.sum(GT_means, axis = 2) / N
		rase_map = (100. / M) * np.sqrt( np.sum(rmse_map ** 2, axis = 2) / N)

		s = int(np.round(ws / 2))
		return np.mean(rase_map[s: -s, s: -s])

	def ergas (self, r = 4, ws = 8):
		"""calculates erreur relative globale adimensionnelle de synthese (ergas).

		:param GT: first (original) input image.
		:param P: second (deformed) input image.
		:param r: ratio of high resolution to low resolution (default=4).
		:param ws: sliding window size (default = 8).

		:returns:  float -- ergas value.
		"""
		gt,P = _initial_check(gt, P)

		rmse_map = None
		nb = 1

		_,rmse_map = rmse_sw(gt, p, ws)

		means_map = uniform_filter(gt, ws) / ws ** 2

		# Avoid division by zero
		idx = means_map == 0
		means_map[idx] = 1
		rmse_map[idx] = 0
		
		ergasroot = np.sqrt(np.sum(((rmse_map ** 2) / (means_map ** 2)), axis = 2) / nb)
		ergas_map = 100 * r * ergasroot

		s = int(np.round(ws / 2))
		return np.mean(ergas_map[s: -s, s: -s])

	def _scc_single (self, win, ws):
		def _scc_filter(inp, axis, output, mode, cval):
			return correlate(inp, win , output, mode, cval, 0)

		GT_hp = generic_laplace(self.gt.astype(np.float64), _scc_filter)
		P_hp = generic_laplace(self.p.astype(np.float64), _scc_filter)
		win = Help().fspecial(Filter.UNIFORM, ws)
		sigmaGT_sq, sigmaP_sq, sigmaGT_P = self._get_sigmas(GT_hp, P_hp, win)

		sigmaGT_sq[sigmaGT_sq<0] = 0
		sigmaP_sq[sigmaP_sq<0] = 0

		den = np.sqrt(sigmaGT_sq) * np.sqrt(sigmaP_sq)
		idx = (den==0)
		den = self._replace_value(den, 0, 1)
		scc = sigmaGT_P / den
		scc[idx] = 0
		return scc

	def scc (self, win=[[-1, -1, -1], [-1, 8, -1],[-1, -1, -1]], ws=8):
		"""calculates spatial correlation coefficient (scc).

		:param GT: first (original) input image.
		:param P: second (deformed) input image.
		:param fltr: high pass filter for spatial processing (default=[[-1,-1,-1],[-1,8,-1],[-1,-1,-1]]).
		:param ws: sliding window size (default = 8).

		:returns:  float -- scc value.
		"""

		coefs = np.zeros(self.gt.shape)
		for i in range(self.gt.shape[2]):
			coefs[:, :, i] = _scc_single(self.gt[:, :, i],self.p[:, :, i], win, ws)
		return np.mean(coefs)	

	def psnrb (self):
		"""Calculates PSNR with Blocking Effect Factor for a given pair of images (PSNR-B)

		:param GT: first (original) input image in YCbCr format or Grayscale.
		:param P: second (corrected) input image in YCbCr format or Grayscale..
		:return: float -- psnr_b.
		"""
		if len(self.gt.shape) == 3:
			GT = self.gt[:, :, 0]

		if len(self.p.shape) == 3:
			P = self.p[:, :, 0]

		imdff = np.double(GT) - np.double(P)

		mse = np.mean(np.square(imdff.flatten()))
		bef = self._compute_bef(P)
		mse_b = mse + bef

		if np.amax(P) > 2:
			psnr_b = 10 * log10(255 ** 2 / mse_b)
		else:
			psnr_b = 10 * log10(1 / mse_b)

		return psnr_b

	def _vifp_single (self, gt, p, sigma_nsq):
		EPS = 1e-10
		num = 0.0
		den = 0.0
		for scale in range(1, 5):
			N=2.0 ** (4-scale+1) + 1
			win = Help().fspecial(Filter.GAUSSIAN,ws = N,sigma = N / 5)

			if scale > 1:
				GT = Help().filter2(gt, win, 'valid')[::2, ::2]
				P = Help().filter2(p, win, 'valid')[::2, ::2]

			GT_sum_sq, P_sum_sq, GT_P_sum_mul = Help()._get_sums(gt, p, win, mode = 'valid')
			sigmaGT_sq, sigmaP_sq, sigmaGT_P = Help()._get_sigmas(gt, p, win, mode='valid', sums = (GT_sum_sq, P_sum_sq, GT_P_sum_mul))


			sigmaGT_sq[sigmaGT_sq < 0] = 0
			sigmaP_sq[sigmaP_sq < 0] = 0

			g = sigmaGT_P / (sigmaGT_sq + EPS)
			sv_sq = sigmaP_sq - g * sigmaGT_P
			
			g[sigmaGT_sq < EPS] = 0
			sv_sq[sigmaGT_sq < EPS] = sigmaP_sq[sigmaGT_sq < EPS]
			sigmaGT_sq[sigmaGT_sq < EPS] = 0
			
			g[sigmaP_sq < EPS] = 0
			sv_sq[sigmaP_sq < EPS] = 0
			
			sv_sq[g < 0] = sigmaP_sq[g < 0]
			g[g < 0] = 0
			sv_sq[sv_sq <= EPS] = EPS
			
		
			num += np.sum(np.log10(1.0 + (g ** 2.) * sigmaGT_sq / (sv_sq + sigma_nsq)))
			den += np.sum(np.log10(1.0 + sigmaGT_sq / sigma_nsq))

		return num / den

	def vifp(self, sigma_nsq = 2):
		"""calculates Pixel Based Visual Information Fidelity (vif-p).

		:param GT: first (original) input image.
		:param P: second (deformed) input image.
		:param sigma_nsq: variance of the visual noise (default = 2)

		:returns:  float -- vif-p value.
		"""
		# gt,P = GT[:,:,np.newaxis],P[:,:,np.newaxis]
		sigma_nsq = 2
		return np.mean([self._vifp_single(self.gt[:, :, i],self.p[:, :, i], sigma_nsq) for i in range(self.gt.shape[2])])

	def sam (self):
		"""calculates spectral angle mapper (sam).

		:param GT: first (original) input image.
		:param P: second (deformed) input image.

		:returns:  float -- sam value.
		"""

		GT = self.gt.reshape((self.gt.shape[0] * self.gt.shape[1], self.gt.shape[2]))
		P = self.p.reshape((self.p.shape[0] * self.p.shape[1], self.p.shape[2]))

		N = GT.shape[1]
		sam_angles = np.zeros(N)
		for i in range(GT.shape[1]):
			val = np.clip(np.dot(GT[:, i], P[:, i]) / (np.linalg.norm(GT[:, i]) * np.linalg.norm(P[:, i])), -1, 1)
			sam_angles[i] = np.arccos(val)
		return np.mean(sam_angles)	

	def d_lambda (self, ms, fused, p = 1):
		"""calculates Spectral Distortion Index (D_lambda).

		:param ms: low resolution multispectral image.
		:param fused: high resolution fused image.
		:param p: parameter to emphasize large spectral differences (default = 1).

		:returns:  float -- D_lambda.
		"""
		L = ms.shape[2]

		M1 = np.zeros((L, L))
		M2 = np.zeros((L, L))

		for l in range(L):
			for r in range(l, L):
				M1[l, r] = M1[r, l] = uqi(fused[:, :, l], fused[:, :, r])
				M2[l, r] = M2[r, l] = uqi(ms[:, :, l], ms[:, :, r])

		diff = np.abs(M1 - M2)**p
		return (1. / (L * (L - 1)) * np.sum(diff)) ** (1. / p)

	def d_s (self, pan, ms, fused, q = 1, r = 4, ws = 7):
		"""calculates Spatial Distortion Index (D_S).

		:param pan: high resolution panchromatic image.
		:param ms: low resolution multispectral image.
		:param fused: high resolution fused image.
		:param q: parameter to emphasize large spatial differences (default = 1).
		:param r: ratio of high resolution to low resolution (default=4).
		:param ws: sliding window size (default = 7).

		:returns:  float -- D_S.
		"""
		pan = pan.astype(np.float64)
		fused = fused.astype(np.float64)

		pan_degraded = uniform_filter(pan.astype(np.float64), size = ws) / (ws ** 2)
		pan_degraded = Help().imresize(pan_degraded, (pan.shape[0] // r, pan.shape[1] // r))
		L = ms.shape[2]

		M1 = np.zeros(L)
		M2 = np.zeros(L)
		for l in range(L):
			M2[l] = uqi(ms[:, :, l], pan_degraded[:, :, l])
			M1[l] = uqi(fused[:, :, l], pan[:, :, l])
			

		diff = np.abs(M1 - M2) ** q
		return ((1. / L) * (np.sum(diff))) ** (1. / q)

	def qnr (self, pan,ms,fused,alpha=1,beta=1,p=1,q=1,r=4,ws=7):
		"""calculates Quality with No Reference (QNR).

		:param pan: high resolution panchromatic image.
		:param ms: low resolution multispectral image.
		:param fused: high resolution fused image.
		:param alpha: emphasizes relevance of spectral distortions to the overall.
		:param beta: emphasizes relevance of spatial distortions to the overall.
		:param p: parameter to emphasize large spectral differences (default = 1).
		:param q: parameter to emphasize large spatial differences (default = 1).
		:param r: ratio of high resolution to low resolution (default=4).
		:param ws: sliding window size (default = 7).

		:returns:  float -- QNR.
		"""
		a = (1-self.d_lambda(ms,fused,p=p))**alpha
		b = (1-self.d_s(pan,ms,fused,q=q,ws=ws,r=r))**beta
		return a*b

if __name__ == "__main__":
	img1 = cv2.imread("gt/000001.png")
	print("shape1", img1.shape)
	h, w, _ = img1.shape
	img2 = cv2.imread("out/000001.png")
	img2 = img2[0:h, 0:w]
	print("shape2", img2.shape)
	img3 = cv2.imread("out_92/000001.png")
	print("shape3", img3.shape)
	# img3 = cv2.resize(img1, (img1.shape[0] * 4, img1.shape[1] * 4))
	# img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
	# img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
	print("1vs2", IQA(img1, img2).msssim())
	print("1 vs 3" ,IQA(img1, img3).msssim())