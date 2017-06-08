import math
import numpy as np
import torch

def addNoise(ipt, miu, std):
	noise = np.random.normal(miu, std, ipt.size())
	noise = np.float32(noise)
	noise = torch.from_numpy(noise)
	return ipt + noise