#codin:utf8
from config import opt
import os
import models
from data import myData
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torchnet import meter
from utils import Visualizer
from tqdm import tqdm
from torchvision import transforms
import torchvision
import torch
from torchsummary import summary
import json
from torch.optim import lr_scheduler
from loss import FocalLoss
from PIL import ImageFilter
import random
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import pickle
import roc
import cv2
def blur(img):
    img = img.filter(ImageFilter.GaussianBlur(radius=random.random()))
    return img
def maxcrop(img):
	w,h = img.size
	size=min(h,w)
	img=img.crop(((w-size)//2,(h-size)//2, w-(w-size)//2,h-(h-size)//2))
	return img

def visualize(**kwargs):
	# 根据命令行参数更新配置
	opt.parse(kwargs)
	vis = Visualizer(opt.env)
	# step1: 模型
	model = getattr(models, opt.model)()

	'''
	model_ft = torchvision.models.vgg16_bn(pretrained = True)
	pretrained_dict = model_ft.state_dict()
	model_dict = model.state_dict()
	# 将pretrained_dict里不属于model_dict的键剔除掉
	pretrained_dict =  {k: v for k, v in pretrained_dict.items() 
					if k in model_dict}
	model_dict.update(pretrained_dict)
	model.load_state_dict(model_dict)
	'''
	if opt.load_model_path:
	   model.load(opt.load_model_path)
	if opt.use_gpu: 
		model.cuda()
		summary(model, (3,224, 224))
	print(opt)
	# step2: 数据
	train_data = myData(
			filelists=opt.train_filelists,
			#transform = data_transforms['train'],
                        scale = opt.cropscale,
			transform = None,
			test = False,
			data_source='none')
	val_data = myData(
			filelists =opt.test_filelists,
			#transform =data_transforms['val'],
			transform =None,
                        scale = opt.cropscale,
			test = False,data_source = 'none')

	train_loader = DataLoader(dataset=train_data,
			batch_size = 1,shuffle = False)
	val_loader = DataLoader(dataset = val_data,
			batch_size = 1,shuffle = False)

	dataloaders={'train':train_loader,'val':val_loader}
	dataset_sizes={'train':len(train_data),'val':len(val_data)}
	
	imgshape = 64
	imgwidth_num = 32 
	def vis(train_loader, outputjpg,imgshape,imgwidth_num):
		print(len(train_loader))
		showtmp = np.zeros((imgshape,imgshape,3),dtype=np.uint8)
		showall = None
		lastnum = imgwidth_num-len(train_loader)%imgwidth_num
		for step,batch in enumerate(tqdm(train_loader,desc='Visual Cropface On Anti-spoofing', unit='batch')):
			inputs,labels= batch
			inputs = inputs.numpy().squeeze()
			inputs = np.transpose(inputs,(1,2,0))
			inputs = np.uint8(inputs)
			inputs = cv2.resize(inputs, (imgshape,imgshape))
			if step%imgwidth_num==0:
				if showall is not None:
					showall = np.vstack([showall, showtmp])
				elif step >0:
					showall = showtmp
				#print(showtmp.shape)
				showtmp = inputs
			else:
				showtmp = np.hstack([showtmp, inputs])
		#print(showtmp.shape)
		for i in range(lastnum):
			showtmp = np.hstack([showtmp, np.zeros((imgshape,imgshape,3),dtype=np.uint8)])
		#print(showtmp.shape)
		showall = np.vstack([showall, showtmp])
		
		cv2.imwrite(outputjpg,showall)
	vis(train_loader, 'data/showcropface_train.jpg',imgshape,imgwidth_num)	
	vis(val_loader, 'data/showcropface_val.jpg',imgshape,imgwidth_num)	
			#inputs = cv2.cvtColor(inputs, cv2.COLOR_RGB2BGR)
			#print(inputs.shape)
			#cv2.imshow('',inputs)
			#cv2.waitKey()
				
def help():
	'''
	打印帮助的信息： python file.py help
	'''

	print('''
	usage : python {0} <function> [--args=value,]
	<function> := train | test | help
	example: 
		   python {0} train --env='env0701' --lr=0.01
		   python {0} test --dataset='path/to/dataset/root/'
		   python {0} help
	avaiable args:'''.format(__file__))

	from inspect import getsource
	source = (getsource(opt.__class__))
	print(source)


if __name__=='__main__':
	import fire
	fire.Fire()
