import opts
import torch

if __name__=='__main__':
	#TODO
	torch.set_default_tensor_type('torch.FloatTensor')
	args = opts.init()

	print(args.GPUs)
	torch.randn(1,2,3,4,5)