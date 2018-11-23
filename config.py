# coding: utf8
import torch
import warnings

class DefaultConfig(object):
   model = 'MyresNet34' # 使用的模型，名字必须与models/__init__.py中的名字一致
   env = model # visdom 环境
   train_root = '' # 训练集存放路径
   val_root = '' # 验证集存放路径
   test_root = 'data/AgriculturalDisease_testA/' # 测试集存放路径
   
   #load_model_path = 'checkpoints/model.pth' # 加载预训练的模型的路径，为None代表不加载
   load_model_path = None # 加载预训练的模型的路径，为None代表不加载

   batch_size = 16# batch size
   use_gpu = torch.cuda.is_available() # use GPU or not
   #use_gpu = True # use GPU or not
   num_workers = 8 # how many workers for loading data
   print_freq = 20 # print info every N batch
   debug_file = '/tmp/debug' # if os.path.exists(debug_file): enter ipdb
   result_name = 'result'

   max_epoch = 10
   lr = 0.01 # initial learning rate
   lr_decay = 0.5 # when val_loss increase, lr = lr*lr_decay
   weight_decay = 1e-5 # 损失函数
   cropscale = 3.5
def parse(self, kwargs):
   '''
   根据字典kwargs 更新 config参数
   '''
   # 更新配置参数
   for k, v in kwargs.items():
      if not hasattr(self, k):
          # 警告还是报错，取决于你个人的喜好
          warnings.warn("Warning: opt has not attribut %s" %k)
      setattr(self, k, v)
      
   # 打印配置信息  
   print('user config:')
   for k, v in self.__class__.__dict__.items():
      if not k.startswith('__'):
          print(k, getattr(self, k))

DefaultConfig.parse = parse
opt = DefaultConfig()
