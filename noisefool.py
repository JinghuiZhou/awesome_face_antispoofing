#coding:utf8
from config import opt
import os
import models
from data import myData_peppersalt
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
from deepfool import deepfool
import matplotlib.pyplot as plt
import numpy as np
import pickle
import roc
import cv2
import glob
def noisefool(**kwargs):
	pths = glob.glob('checkpoints/%s/*.pth'%(opt.model))
	pths.sort(key=os.path.getmtime,reverse=True)
	print(pths)
	opt.parse(kwargs)
	# 模型
	opt.load_model_path=pths[0]
	model = getattr(models, opt.model)().eval()
	assert os.path.exists(opt.load_model_path)
	if opt.load_model_path:
	   model.load(opt.load_model_path)
	if opt.use_gpu: model.cuda()
	model.train()
	# 数据
	#result_name = '../../model/se-resnet/test_se_resnet50'
	test_data = myData_peppersalt(
			filelists =opt.test_filelists,
			#transform =data_transforms['val'],
			transform =None,
                        scale = opt.cropscale,
			test = True,data_source = 'none')

#	test_data = myData(root = opt.test_roo,datatxt='test.txt',
#				test = True,transform = data_transforms['test'])
	test_loader =DataLoader(dataset = test_data,batch_size = opt.batch_size//2,shuffle = False)
	#test_loader =DataLoader(dataset = test_data,batch_size = opt.batch_size//2,shuffle =True)
	
	result_list=[]

	label_list=[]	
	#workers = multiprocessing.Pool(16)

	for step,batch  in enumerate(tqdm(test_loader,desc='test', unit='batch')):
		data,label,image_path  =  batch
		with torch.no_grad():
			if opt.use_gpu:
				data =  data.cuda()
			outputs = model(data)
			outputs = torch.softmax(outputs,dim=-1)
			preds = outputs.to('cpu').numpy()
			if 0: 
				print(label, outputs)
			for i in range(preds.shape[0]):
				result_list.append(preds[i,1])
				label_list.append(label[i])
	metric =roc.cal_metric(label_list, result_list)
	eer = metric[0]	
	tprs = metric[1]	
	auc = metric[2]
	xy_dic = metric[3]
	pickle.dump(xy_dic, open('result/noisefool.pickle','wb'))	
	print('EER: {:.6f} TPR(1.0%): {:.6f} TPR(.5%): {:.6f} AUC: {:.8f}'.format(eer, tprs["TPR(1.%)"], tprs["TPR(.5%)"], auc),file=open('result/noisefool.txt','a'))
	print('EER: {:.6f} TPR(1.0%): {:.6f} TPR(.5%): {:.6f} AUC: {:.8f}'.format(eer, tprs["TPR(1.%)"], tprs["TPR(.5%)"], auc))
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
