#coding:utf8
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

ATTACK = 1
GENUINE = 0
train_filelists=[
['/home/cv/zjh/aich/dataset/raw/ClientRaw','/home/cv/zjh/aich/dataset/raw/client_train_raw.txt',ATTACK],
['/home/cv/zjh/aich/dataset/raw/ImposterRaw','/home/cv/zjh/aich/dataset/raw/imposter_train_raw.txt',GENUINE]
]
test_filelists=[
['/home/cv/zjh/aich/dataset/raw/ClientRaw','/home/cv/zjh/aich/dataset/raw/client_test_raw.txt',ATTACK],
['/home/cv/zjh/aich/dataset/raw/ImposterRaw','/home/cv/zjh/aich/dataset/raw/imposter_test_raw.txt',GENUINE]
]

def blur(img):
    img = img.filter(ImageFilter.GaussianBlur(radius=random.random()))
    return img
def maxcrop(img):
	w,h = img.size
	size=min(h,w)
	img=img.crop(((w-size)//2,(h-size)//2, w-(w-size)//2,h-(h-size)//2))
	return img

'''

data_transforms = {
	'train' : transforms.Compose([
		#transforms.RandomRotation((45)),
		transforms.RandomHorizontalFlip(),
		#transforms.RandomVerticalFlip(),
		#transforms.Lambda(maxcrop),
		#transforms.Lambda(blur),
		transforms.Resize((224,224)) ,
	   	transforms.ToTensor() ,
		transforms.Normalize([0.485 , 0.456 , 0.406] , [0.229 , 0.224 , 0.225])
	]) ,
	'val' : transforms.Compose([
		#transforms.Lambda(maxcrop),
		transforms.Resize((224,224)) ,
		#transforms.RandomHorizontalFlip(),
		transforms.ToTensor() ,
		transforms.Normalize([0.485 , 0.456 , 0.406] , [0.229 , 0.224 , 0.225])
	]),
	'test' : transforms.Compose([
		#transforms.Lambda(maxcrop),
		transforms.Resize((224,224)) ,
		#transforms.RandomHorizontalFlip(),
		transforms.ToTensor() ,
		transforms.Normalize([0.485 , 0.456 , 0.406] , [0.229 , 0.224 , 0.225])
	]) ,}
'''
def train(**kwargs):
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
			filelists=train_filelists,
			#transform = data_transforms['train'],
                        scale = opt.cropscale,
			transform = None,
			test = False,
			data_source='none')
	val_data = myData(
			filelists =test_filelists,
			#transform =data_transforms['val'],
			transform =None,
                        scale = opt.cropscale,
			test = False,data_source = 'none')

	train_loader = DataLoader(dataset=train_data,
			batch_size = opt.batch_size,shuffle = True)
	val_loader = DataLoader(dataset = val_data,
			batch_size = opt.batch_size//2,shuffle = False)

	dataloaders={'train':train_loader,'val':val_loader}
	dataset_sizes={'train':len(train_data),'val':len(val_data)}
	
	# step3: 目标函数和优化器
	criterion = FocalLoss(2)
	#criterion = torch.nn.CrossEntropyLoss()
	lr = opt.lr
	#optimizer = t.optim.Adam(model.parameters(),
	#                       lr = lr,
	#                       weight_decay = opt.weight_decay)
	optimizer = torch.optim.SGD(model.parameters() , 
							lr =opt.lr , 
							momentum = 0.9,
							weight_decay= opt.weight_decay)
	exp_lr_scheduler = lr_scheduler.StepLR(optimizer , 
					step_size = 8 , gamma = 0.5)
	#set learning rate every 10 epoch decrease 10%
	# step4: 统计指标：平滑处理之后的损失，还有混淆矩阵

	confusion_matrix = meter.ConfusionMeter(2)
	train_loss = meter.AverageValueMeter()#为了可视化增加的内容
	val_loss = meter.AverageValueMeter()
	train_acc = meter.AverageValueMeter()#为了可视化增加的内容
	val_acc = meter.AverageValueMeter()
	previous_loss = 1e100
	best_acc = 0.0
	# 训练
	for epoch in range(opt.max_epoch):
		print('Epoch {}/{}'.format(epoch ,opt.max_epoch - 1))
		print('-' * 10)
		train_loss.reset()
		train_acc.reset()
		running_loss = 0.0
		running_corrects = 0
		exp_lr_scheduler.step() 
		for step,batch in enumerate(tqdm(train_loader,desc='Train On Anti-spoofing', unit='batch')):
			inputs,labels= batch
			 
			if opt.use_gpu:
				inputs = Variable(inputs.cuda())
				labels = Variable(labels.cuda())
			else:
				inputs = Variable(inputs)
				lables = Variable(labels)
			optimizer.zero_grad()   #zero the parameter gradients
			with torch.set_grad_enabled(True):
				outputs = model(inputs)
				#print(outputs.shape)
				_ , preds = torch.max(outputs , 1)
			
				loss0 = criterion(outputs , labels)
				loss = loss0
				loss.backward()  #backward of gradient
				optimizer.step()  #strategy to drop
				if step%20==0:
					pass
					#print('epoch:%d/%d step:%d/%d loss: %.4f loss0: %.4f loss1: %.4f'%(epoch, opt.max_epoch, step, len(train_loader), 
					#loss.item(),loss0.item(),loss1.item()))	
			running_loss += loss.item() * inputs.size(0)
			running_corrects += torch.sum(preds == labels.data)
			'''
			if step%opt.print_freq==opt.print_freq-1:
				vis.plot('loss', train_loss.value()[0])
			   
			   # 如果需要的话，进入debug模式
			   if os.path.exists(opt.debug_file):
				   import ipdb;
				   ipdb.set_trace()	
			'''
		epoch_loss = running_loss / dataset_sizes['train']
		epoch_acc = running_corrects.double() / float(dataset_sizes['train'])
		print('Train Loss: {:.8f} Acc: {:.4f}'.format(epoch_loss,epoch_acc))
		train_loss.add(epoch_loss)
		train_acc.add(epoch_acc)
		
		val_loss.reset()
		val_acc.reset()
		val_cm,v_loss,v_accuracy = val(model,val_loader,dataset_sizes['val'])
		print('Val Loss: {:.8f} Acc: {:.4f}'.format(v_loss,v_accuracy))
		val_loss.add(v_loss)
		val_acc.add(v_accuracy)
		

		vis.plot_many_stack({'train_loss':train_loss.value()[0],\
						'val_loss':val_loss.value()[0]},win_name ="Loss")
		vis.plot_many_stack({'train_acc':train_acc.value()[0],\
						'val_acc':val_acc.value()[0]},win_name = 'Acc')
		vis.log("epoch:{epoch},lr:{lr},\
				train_loss:{train_loss},train_acc:{train_acc},\
				val_loss:{val_loss},val_acc:{val_acc},\
				train_cm:{train_cm},val_cm:{val_cm}"
	   .format(
				   epoch = epoch,
				   train_loss = train_loss.value()[0],
				   train_acc = train_acc.value()[0],
				   val_loss = val_loss.value()[0],
				   val_acc = val_acc.value()[0],
				   train_cm=str(confusion_matrix.value()),
				   val_cm = str(val_cm.value()),
				   lr=lr))
		'''
		if v_loss > previous_loss:          
			lr = lr * opt.lr_decay
			for param_group in optimizer.param_groups:
				param_group['lr'] = lr
		'''
		vis.plot_many_stack({'lr':lr},win_name ='lr')
		previous_loss = val_loss.value()[0]
		if v_accuracy > best_acc:
			best_acc = v_accuracy
			best_acc_epoch = epoch
			#best_model_wts = model.state_dict()
			os.system('mkdir -p %s'%(os.path.join('checkpoints',opt.model)))
			model.save(name = 'checkpoints/'+opt.model+'/'+str(epoch)+'.pth')
			print('Epoch: {:d} Val Loss: {:.8f} Acc: {:.4f}'.format(epoch,v_loss,v_accuracy),file=open('result/val.txt','a'))
		#model.load_state_dict(best_model_wts)	
	print('Best val Epoch: {},Best val Acc: {:4f}'.format(best_acc_epoch,best_acc))
def val(model,dataloader,data_len):
	# 把模型设为验证模式
	criterion = FocalLoss(2)
	model.train(False)
	running_loss = 0
	running_corrects = 0
	confusion_matrix = meter.ConfusionMeter(2)
	for ii, data in enumerate(tqdm(dataloader,desc='Val On Anti-spoofing', unit='batch')):
		input, label = data
		with torch.no_grad():
			val_input = Variable(input)
			val_label = Variable(label)
		if opt.use_gpu:
			val_input = val_input.cuda()
			val_label = val_label.cuda()
		score = model(val_input)
		_ , preds = torch.max(score , 1)
		loss = criterion(score, val_label)
		confusion_matrix.add(score.data.squeeze(), val_label)
		running_loss += loss.item() * val_input.size(0)
		running_corrects += torch.sum(preds == val_label.data)
	# 把模型恢复为训练模式
	model.train(True)

	cm_value = confusion_matrix.value()
	val_loss = running_loss / data_len
	val_accuracy = running_corrects.double() / float(data_len)
	return confusion_matrix, val_loss,val_accuracy


def test(**kwargs):
	import glob
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
	model.train(False)
	# 数据
	#result_name = '../../model/se-resnet/test_se_resnet50'
	test_data = myData(root = opt.test_root,datatxt='test.txt',
				test = True,transform = data_transforms['test'])
	test_loader =DataLoader(dataset = test_data,batch_size = opt.batch_size,shuffle = False)
	
	result_list=[]

	for step,batch  in enumerate(tqdm(test_loader,desc='test', unit='batch')):
		data,name  =  batch
		with torch.no_grad():
			if opt.use_gpu:
				data =  data.cuda()
			outputs = model(data)
			_ , preds = torch.max(outputs , 1)
			#print(preds)
			preds = preds.to("cpu").numpy()
			preds=preds.data
			for i in range(len(name)):
				result_dict={}
				result_dict["image_id"]=name[i]
				result_dict["disease_class"] = preds[i]
				result_list.append(result_dict)
	with open('checkpoints/'+opt.model+'/'+opt.result_name+'.json','w') as outfile:
		json.dump(result_list,outfile,ensure_ascii=False)
		outfile.write('\n')

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
