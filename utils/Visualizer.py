#coding:utf8
import visdom
import time
import numpy as np
class Visualizer(object):
	'''
	封装了visdom的基本操作，但是你仍然可以通过`self.vis.function`
	或者`self.function`调用原生的visdom接口
	比如 
	self.text('hello visdom')
	self.histogram(t.randn(1000))
	self.line(t.arange(0, 10),t.arange(1, 11))
	'''

	def __init__(self, env='default', **kwargs):
		self.vis = visdom.Visdom(env=env, **kwargs)
	   
		# 画的第几个数，相当于横坐标
		# 比如（’loss',23） 即loss的第23个点
		self.index = {} 
		self.log_text = ''
	def reinit(self, env='default', **kwargs):
		'''
		修改visdom的配置
		'''
		self.vis = visdom.Visdom(env=env, **kwargs)
		return self

	def plot_many(self, d):
		'''
		一次plot多个
		@params d: dict (name, value) i.e. ('loss', 0.11)
		'''
		for k, v in d.iteritems():
			self.plot(k, v)

	def img_many(self, d):
		for k, v in d.iteritems():
			self.img(k, v)

	def plot(self, name, y, **kwargs):
		'''
		self.plot('loss', 1.00)
		'''
		x = self.index.get(name, 0)
		self.vis.line(Y=np.array([y]), X=np.array([x]),
					 win=unicode(name),
					 opts=dict(title=name),
					 update=None if x == 0 else 'append',
					 **kwargs
					 )
		self.index[name] = x + 1
	def plot_many_stack(self, d, win_name):
		name=list(d.keys())
		name_total=" ".join(name)
		x = self.index.get(name_total, 0)
		val=list(d.values())
		if len(val)==1:
			y=np.array(val)
		else:
			y=np.array(val).reshape(-1,len(val))
		#print(x)
		self.vis.line(Y=y,X=np.ones(y.shape)*x,
					win=str(win_name),#unicode
					opts=dict(legend=name,
						title=win_name),
					update=None if x == 0 else 'append'
					)
		self.index[name_total] = x + 1     
	def img(self, name, img_, **kwargs):
		'''
		self.img('input_img', t.Tensor(64, 64))
		self.img('input_imgs', t.Tensor(3, 64, 64))
		self.img('input_imgs', t.Tensor(100, 1, 64, 64))
		self.img('input_imgs', t.Tensor(100, 3, 64, 64), nrows=10)
		'''
		self.vis.images(img_.cpu().numpy(),
					  win=unicode(name),
					  opts=dict(title=name),
					  **kwargs
					  )

	def log(self, info, win='log_text'):
		'''
		self.log({'loss':1, 'lr':0.0001})
		'''

		self.log_text += ('[{time}] {info} <br>'.format(
						   time=time.strftime('%m%d_%H%M%S'),\
						   info=info)) 
		self.vis.text(self.log_text, win)   

	def __getattr__(self, name):
		'''
		self.function 等价于self.vis.function
		自定义的plot,image,log,plot_many等除外
		'''
		return getattr(self.vis, name)
 


if __name__ == '__main__':
	  
	from torchnet import meter
	#用 torchnet来存放损失函数，如果没有，请安装conda install torchnet
	'''
	训练前的模型、损失函数设置 
	vis = Visualizer(env='my_wind')#为了可视化增加的内容
	loss_meter = meter.AverageValueMeter()#为了可视化增加的内容
	for epoch in range(10):
		#每个epoch开始前，将存放的loss清除，重新开始记录
		loss_meter.reset()#为了可视化增加的内容
		model.train()
		for ii,(data,label)in enumerate(trainloader):     
			...
			out=model(input)
			loss=...
			loss_meter.add(loss.data[0])#为了可视化增加的内容
			
		#loss可视化
		#loss_meter.value()[0]返回存放的loss的均值
		vis.plot_many_stack({'train_loss': loss_meter.value()[0]})#为了可视化增加的内容    
	'''
	#示例
	vis = Visualizer(env='loss')

	train_loss = meter.AverageValueMeter()#为了可视化增加的内容
	val_loss = meter.AverageValueMeter()
	train_acc = meter.AverageValueMeter()#为了可视化增加的内容
	val_acc = meter.AverageValueMeter()
	for epoch in range(1,10):
		train_loss.reset()#为了可视化增加的内容
		val_loss.reset()
		time.sleep(1)
		train_loss.add(np.exp(epoch+1))#假设loss=epoch
		val_loss.add(np.log(epoch))#print(loss_meter.value())
		train_acc.add(np.exp(epoch+1))
		val_acc.add(epoch+2)
		vis.plot_many_stack({'train_loss':train_loss.value()[0],'test_loss':val_loss.value()[0]},win_name = "resnet18/loss")
		#vis.plot_many_stack({'train_acc':train_acc.value()[0]})
		#time.sleep(3)
		#vis.plot_many_stack({'train_loss': loss_meter.value()[0]})#为了可视化增加的内容 
		#如果还想同时显示test loss，如法炮制,并用字典的形式赋值，如下。还可以同时显示train和test accuracy
		