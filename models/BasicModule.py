# coding: utf8
import time
import torch
#import pretrainedmodels
from torchvision import datasets, models,transforms
import torch.nn as nn
class BasicModule(torch.nn.Module):
   '''
   封装了nn.Module，主要提供save和load两个方法
   '''
   def __init__(self,opt=None):
       super(BasicModule,self).__init__()
       self.model_name = str(type(self)) # 模型的默认名字

   def load(self, path):
       '''
       可加载指定路径的模型
       '''
       self.load_state_dict(torch.load(path))

   def save(self,name=None):
       '''
       保存模型，默认使用“模型名字+时间”作为文件名，
       如AlexNet_0710_23:57:29.pth
       '''

       if name is None:
           prefix = '/checkpoints/' + self.model_name + '_'
           name = time.strftime(prefix + '%m%d_%H:%M:%S.pth')
       torch.save(self.state_dict(), name)
       return name


class MyVggNet11(BasicModule):
    def __init__(self):
        super(MyVggNet11, self).__init__()
        model = models.vgg11_bn(pretrained = True)
        self.vggnet_lay=nn.Sequential(*list(model.children())[:-1])
        self.fc_Linear_lay2 = nn.Linear(128,2)
        self.drop = nn.Dropout(0.5)
        self.conv_lay1 = nn.Conv2d(512, 256, kernel_size=(2,2), stride=(1,1))
        self.batch_lay1 = nn.BatchNorm2d(256)
        self.relu_lay1 = nn.ReLU(inplace=True)
        self.avgpool_lay1=nn.AvgPool2d((2,2))

        self.conv_lay2 = nn.Conv2d(256, 128,kernel_size=(2,2),stride=(1,1))
        self.batch_lay2=nn.BatchNorm2d(128)
        self.relu_lay2 = nn.ReLU(inplace=True)
        self.avgpool_lay2 = nn.AvgPool2d((2,2))
        
    def forward(self, x):
        x = self.vggnet_lay(x)
        x = self.conv_lay1(x)
        x = self.batch_lay1(x)
        x = self.relu_lay1(x)
        x = self.avgpool_lay1(x)
        
        x = self.conv_lay2(x)
        x = self.batch_lay2(x)
        x = self.relu_lay2(x)
        
        x = self.avgpool_lay2(x)
        x = self.drop(x)
        x = x.view(x.size(0),-1)
        x = self.fc_Linear_lay2(x)
        
        return x
class MyVggNet13(BasicModule):
    def __init__(self):
        super(MyVggNet13, self).__init__()
        model = models.vgg13_bn(pretrained = True)
        self.vggnet_lay=nn.Sequential(*list(model.children())[:-1])
        self.fc_Linear_lay2 = nn.Linear(128,2)
        self.drop = nn.Dropout(0.5)
        self.conv_lay1 = nn.Conv2d(512, 256, kernel_size=(2,2), stride=(1,1))
        self.batch_lay1 = nn.BatchNorm2d(256)
        self.relu_lay1 = nn.ReLU(inplace=True)
        self.avgpool_lay1=nn.AvgPool2d((2,2))

        self.conv_lay2 = nn.Conv2d(256, 128,kernel_size=(2,2),stride=(1,1))
        self.batch_lay2=nn.BatchNorm2d(128)
        self.relu_lay2 = nn.ReLU(inplace=True)
        self.avgpool_lay2 = nn.AvgPool2d((2,2))
        
    def forward(self, x):
        x = self.vggnet_lay(x)
        x = self.conv_lay1(x)
        x = self.batch_lay1(x)
        x = self.relu_lay1(x)
        x = self.avgpool_lay1(x)
        
        x = self.conv_lay2(x)
        x = self.batch_lay2(x)
        x = self.relu_lay2(x)
        
        x = self.avgpool_lay2(x)
        x = self.drop(x)
        x = x.view(x.size(0),-1)
        x = self.fc_Linear_lay2(x)
        
        return x
class MyVggNet16(BasicModule):
    def __init__(self):
        super(MyVggNet16, self).__init__()
        model = models.vgg16_bn(pretrained = True)
        self.vggnet_lay=nn.Sequential(*list(model.children())[:-1])
        self.fc_Linear_lay2 = nn.Linear(128,2)
        self.drop = nn.Dropout(0.5)
        self.conv_lay1 = nn.Conv2d(512, 256, kernel_size=(2,2), stride=(1,1))
        self.batch_lay1 = nn.BatchNorm2d(256)
        self.relu_lay1 = nn.ReLU(inplace=True)
        self.avgpool_lay1=nn.AvgPool2d((2,2))

        self.conv_lay2 = nn.Conv2d(256, 128,kernel_size=(2,2),stride=(1,1))
        self.batch_lay2=nn.BatchNorm2d(128)
        self.relu_lay2 = nn.ReLU(inplace=True)
        self.avgpool_lay2 = nn.AvgPool2d((2,2))
        
    def forward(self, x):
        x = self.vggnet_lay(x)
        x = self.conv_lay1(x)
        x = self.batch_lay1(x)
        x = self.relu_lay1(x)
        x = self.avgpool_lay1(x)
        
        x = self.conv_lay2(x)
        x = self.batch_lay2(x)
        x = self.relu_lay2(x)
        
        x = self.avgpool_lay2(x)
        x = self.drop(x)
        x = x.view(x.size(0),-1)
        x = self.fc_Linear_lay2(x)
        
        return x
class MyVggNet19(BasicModule):
    def __init__(self):
        super(MyVggNet19, self).__init__()
        model = models.vgg19_bn(pretrained = True)
        self.vggnet_lay=nn.Sequential(*list(model.children())[:-1])
        self.fc_Linear_lay2 = nn.Linear(128,2)
        self.drop = nn.Dropout(0.5)
        self.conv_lay1 = nn.Conv2d(512, 256, kernel_size=(2,2), stride=(1,1))
        self.batch_lay1 = nn.BatchNorm2d(256)
        self.relu_lay1 = nn.ReLU(inplace=True)
        self.avgpool_lay1=nn.AvgPool2d((2,2))

        self.conv_lay2 = nn.Conv2d(256, 128,kernel_size=(2,2),stride=(1,1))
        self.batch_lay2=nn.BatchNorm2d(128)
        self.relu_lay2 = nn.ReLU(inplace=True)
        self.avgpool_lay2 = nn.AvgPool2d((2,2))
        
    def forward(self, x):
        x = self.vggnet_lay(x)
        x = self.conv_lay1(x)
        x = self.batch_lay1(x)
        x = self.relu_lay1(x)
        x = self.avgpool_lay1(x)
        
        x = self.conv_lay2(x)
        x = self.batch_lay2(x)
        x = self.relu_lay2(x)
        
        x = self.avgpool_lay2(x)
        x = self.drop(x)
        x = x.view(x.size(0),-1)
        x = self.fc_Linear_lay2(x)
        
        return x

class MyresNet18(BasicModule):
    def __init__(self):
        super(MyresNet18, self).__init__()
        model = models.resnet18(pretrained = True)
        self.resnet_lay=nn.Sequential(*list(model.children())[:-2])
        #self.drop_lay=nn.Dropout2d(0.5)
        self.conv1_lay = nn.Conv2d(512, 256, kernel_size = (1,1),stride=(1,1))
        self.relu1_lay = nn.ReLU(inplace = True)
        self.drop_lay = nn.Dropout2d(0.5)
        self.global_average = nn.AdaptiveAvgPool2d((1,1))
        self.fc_Linear_lay2 = nn.Linear(256,2)
        

    def forward(self, x):
        
        x= self.resnet_lay(x)
        x = self.conv1_lay(x)
        x = self.relu1_lay(x)
        x = self.drop_lay(x)
        x= self.global_average(x)
        x = x.view(x.size(0),-1)
        x = self.fc_Linear_lay2 (x)
        return x

class MyresNet34(BasicModule):
    def __init__(self):
        super(MyresNet34, self).__init__()
        model = models.resnet34(pretrained = True)
        self.resnet_lay=nn.Sequential(*list(model.children())[:-2])
        #self.drop_lay=nn.Dropout2d(0.5)
        self.conv1_lay = nn.Conv2d(512, 256, kernel_size = (1,1),stride=(1,1))
        self.relu1_lay = nn.ReLU(inplace = True)
        self.drop_lay = nn.Dropout2d(0.5)
        self.global_average = nn.AdaptiveAvgPool2d((1,1))
        self.fc_Linear_lay2 = nn.Linear(256,2)
        

    def forward(self, x):
        
        x= self.resnet_lay(x)
        x = self.conv1_lay(x)
        x = self.relu1_lay(x)
        x = self.drop_lay(x)
        x= self.global_average(x)
        x = x.view(x.size(0),-1)
        x = self.fc_Linear_lay2 (x)
        return x

class MultiscaleresNet18(BasicModule):
    def __init__(self):
        super(MultiscaleresNet18, self).__init__()
        model = models.resnet18(pretrained = True)
        self.resnet_lay=nn.Sequential(*list(model.children())[:-2])
        self.conv1_lay = nn.Conv2d(512*3, 256, kernel_size = (1,1),stride=(1,1))
        self.relu1_lay = nn.ReLU(inplace = True)
        self.drop_lay = nn.Dropout2d(0.5)
        self.global_average = nn.AdaptiveAvgPool2d((1,1))
        self.fc_Linear_lay2 = nn.Linear(256,2)
        

    def forward(self, x):
        
        for i in range(3):
            x[i] = self.resnet_lay(x[i])
            x[i] = self.global_average(x[i])
        output = [x[0],x[1],x[2]]
        output = (torch.cat(output, 1))

        output = self.conv1_lay(output)
        output = self.relu1_lay(output)
        output = self.drop_lay(output)
        output= self.global_average(output)
        output = output.view(output.size(0),-1)
        output = self.fc_Linear_lay2 (output)


class MyresNet50(BasicModule):
    def __init__(self):
        super(MyresNet50, self).__init__()
        model = models.resnet50(pretrained = True)
        self.resnet_lay=nn.Sequential(*list(model.children())[:-2])
        self.conv1_lay = nn.Conv2d(2048, 512, kernel_size = (1,1),stride=(1,1))
        self.relu1_lay = nn.ReLU(inplace = True)
        self.drop_lay = nn.Dropout2d(0.5)
        self.global_average = nn.AdaptiveAvgPool2d((1,1))
        self.fc_Linear_lay2 = nn.Linear(512,2)
        

    def forward(self, x):
        x= self.resnet_lay(x)
        x = self.conv1_lay(x)
        x = self.relu1_lay(x)
        x = self.drop_lay(x)
        x= self.global_average(x)
        x = x.view(x.size(0),-1)
        x = self.fc_Linear_lay2 (x)
        
        return x


class MyresNet50(BasicModule):
    def __init__(self):
        super(MyresNet50, self).__init__()
        model = models.resnet50(pretrained = True)
        self.resnet_lay=nn.Sequential(*list(model.children())[:-2])
        self.conv1_lay = nn.Conv2d(2048, 512, kernel_size = (1,1),stride=(1,1))
        self.relu1_lay = nn.ReLU(inplace = True)
        self.drop_lay = nn.Dropout2d(0.5)
        self.global_average = nn.AdaptiveAvgPool2d((1,1))
        self.fc_Linear_lay2 = nn.Linear(512,2)
        

    def forward(self, x):
        x= self.resnet_lay(x)
        x = self.conv1_lay(x)
        x = self.relu1_lay(x)
        x = self.drop_lay(x)
        x= self.global_average(x)
        x = x.view(x.size(0),-1)
        x = self.fc_Linear_lay2 (x)
        
        return x
'''
class MyInceptionV3(BasicModule):
    def __init__(self):
        super(MyInceptionV3, self).__init__()
        model = models.inception_v3(pretrained = True)
        self.resnet_lay=nn.Sequential(*list(model.children())[:-1])
        self.resnet_lay=nn.Sequential(*list(model.children())[:-1])
        self.conv1_lay = nn.Conv2d(2048, 512, kernel_size = (1,1),stride=(1,1))
        self.relu1_lay = nn.ReLU(inplace = True)
        self.drop_lay = nn.Dropout2d(0.5)
        self.global_average = nn.AdaptiveAvgPool2d((1,1))
        self.fc_Linear_lay2 = nn.Linear(512,2)
        

    def forward(self, x):
        x= self.resnet_lay(x)
        x = self.conv1_lay(x)
        x = self.relu1_lay(x)
        x = self.drop_lay(x)
        x= self.global_average(x)
        x = x.view(x.size(0),-1)
        x = self.fc_Linear_lay2 (x)
        
        return x
'''
class MydenseNet121(BasicModule):
    def __init__(self):
        super(MydenseNet121, self).__init__()
        model = models.densenet121(pretrained = True)
        self.resnet_lay=nn.Sequential(*list(model.children())[:-1])
        self.conv1_lay = nn.Conv2d(1024, 512, kernel_size = (1,1),stride=(1,1))
        self.relu1_lay = nn.ReLU(inplace = True)
        self.drop_lay = nn.Dropout2d(0.5)
        self.global_average = nn.AdaptiveAvgPool2d((1,1))
        self.fc_Linear_lay2 = nn.Linear(512,2)
        

    def forward(self, x):
        x= self.resnet_lay(x)
        x = self.conv1_lay(x)
        x = self.relu1_lay(x)
        x = self.drop_lay(x)
        x= self.global_average(x)
        x = x.view(x.size(0),-1)
        x = self.fc_Linear_lay2 (x)
        
        return x
class MydenseNet161(BasicModule):
    def __init__(self):
        super(MydenseNet161, self).__init__()
        model = models.densenet161(pretrained = True)
        self.resnet_lay=nn.Sequential(*list(model.children())[:-1])
        self.conv1_lay = nn.Conv2d(2208, 512, kernel_size = (1,1),stride=(1,1))
        self.relu1_lay = nn.ReLU(inplace = True)
        self.drop_lay = nn.Dropout2d(0.5)
        self.global_average = nn.AdaptiveAvgPool2d((1,1))
        self.fc_Linear_lay2 = nn.Linear(512,2)
        

    def forward(self, x):
        x= self.resnet_lay(x)
        x = self.conv1_lay(x)
        x = self.relu1_lay(x)
        x = self.drop_lay(x)
        x= self.global_average(x)
        x = x.view(x.size(0),-1)
        x = self.fc_Linear_lay2 (x)
        
        return x
class MydenseNet169(BasicModule):
    def __init__(self):
        super(MydenseNet169, self).__init__()
        model = models.densenet169(pretrained = True)
        self.resnet_lay=nn.Sequential(*list(model.children())[:-1])
        self.conv1_lay = nn.Conv2d(1664, 512, kernel_size = (1,1),stride=(1,1))
        self.relu1_lay = nn.ReLU(inplace = True)
        self.drop_lay = nn.Dropout2d(0.5)
        self.global_average = nn.AdaptiveAvgPool2d((1,1))
        self.fc_Linear_lay2 = nn.Linear(512,2)
        

    def forward(self, x):
        x= self.resnet_lay(x)
        x = self.conv1_lay(x)
        x = self.relu1_lay(x)
        x = self.drop_lay(x)
        x= self.global_average(x)
        x = x.view(x.size(0),-1)
        x = self.fc_Linear_lay2 (x)
        
        return x
class MydenseNet201(BasicModule):
    def __init__(self):
        super(MydenseNet201, self).__init__()
        model = models.densenet201(pretrained = True)
        self.resnet_lay=nn.Sequential(*list(model.children())[:-1])
        self.conv1_lay = nn.Conv2d(1920, 512, kernel_size = (1,1),stride=(1,1))
        self.relu1_lay = nn.ReLU(inplace = True)
        self.drop_lay = nn.Dropout2d(0.5)
        self.global_average = nn.AdaptiveAvgPool2d((1,1))
        self.fc_Linear_lay2 = nn.Linear(512,2)
        

    def forward(self, x):
        x= self.resnet_lay(x)
        x = self.conv1_lay(x)
        x = self.relu1_lay(x)
        x = self.drop_lay(x)
        x= self.global_average(x)
        x = x.view(x.size(0),-1)
        x = self.fc_Linear_lay2 (x)
        
        return x
        
class MyseNet1_0(BasicModule):
    def __init__(self):
        super(MyseNet1_0, self).__init__()
        model = models.squeezenet1_0(pretrained = True)
        self.resnet_lay=nn.Sequential(*list(model.children())[:-1])
        self.conv1_lay = nn.Conv2d(512, 512, kernel_size = (1,1),stride=(1,1))
        self.relu1_lay = nn.ReLU(inplace = True)
        self.drop_lay = nn.Dropout2d(0.5)
        self.global_average = nn.AdaptiveAvgPool2d((1,1))
        self.fc_Linear_lay2 = nn.Linear(512,2)
        

    def forward(self, x):
        x= self.resnet_lay(x)
        x = self.conv1_lay(x)
        x = self.relu1_lay(x)
        x = self.drop_lay(x)
        x= self.global_average(x)
        x = x.view(x.size(0),-1)
        x = self.fc_Linear_lay2 (x)
        
        return x
class MyseNet1_1(BasicModule):
    def __init__(self):
        super(MyseNet1_1, self).__init__()
        model = models.squeezenet1_1(pretrained = True)
        self.resnet_lay=nn.Sequential(*list(model.children())[:-1])
        self.conv1_lay = nn.Conv2d(512, 512, kernel_size = (1,1),stride=(1,1))
        self.relu1_lay = nn.ReLU(inplace = True)
        self.drop_lay = nn.Dropout2d(0.5)
        self.global_average = nn.AdaptiveAvgPool2d((1,1))
        self.fc_Linear_lay2 = nn.Linear(512,2)
        

    def forward(self, x):
        x= self.resnet_lay(x)
        x = self.conv1_lay(x)
        x = self.relu1_lay(x)
        x = self.drop_lay(x)
        x= self.global_average(x)
        x = x.view(x.size(0),-1)
        x = self.fc_Linear_lay2 (x)
        
        return x

if __name__ == '__main__':
    model = MyVgg19Net()
    print(model)
    model.load()
