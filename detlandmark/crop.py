# coding: utf8
import pickle
import torch.utils.data as Data
from PIL import ImageFilter
import random
import torch
from torchvision import transforms
from  torch.autograd import Variable
import numpy as np 
from PIL import Image
import json
import os
import cv2
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


class myData(torch.utils.data.Dataset):
    
    def __init__(self,filelists,scale=3.5,image_size=224,transform=None,test=False,data_source = None):
        self.transform = transform
        self.test = test
        self.img_label=[]
        self.scale = scale
        self.image_size = image_size
        print('myData, test=',self.test)
        if self.test == False:
            for tmp in filelists:
                root,f,label = tmp
                filedir = os.path.join(f)
                fopen = open(filedir,'r')
                datas = fopen.readlines()
                for d in datas:
                    d = d.replace('\n','').split(' ')
                    imgdir,l = d[0],d[1:]
                    imgdir = imgdir.replace('\\','/')
                    self.img_label.append({'path':os.path.join(root, imgdir), 'class': label, 'ldmk': l})
        else:
            for tmp in filelists:
                root,f,label = tmp
                filedir = os.path.join(f)
                fopen = open(filedir,'r')
                datas = fopen.readlines()
                for d in datas:
                    d = d.replace('\n','').split(' ')
                    imgdir,l = d[0],d[1:]
                    imgdir = imgdir.replace('\\','/')
                    self.img_label.append({'path':os.path.join(root, imgdir), 'class': label, 'ldmk': l})

        
    def crop_with_ldmk(self,image, landmark):
        ct_x, std_x = landmark[:,0].mean(), landmark[:,0].std()
        ct_y, std_y = landmark[:,1].mean(), landmark[:,1].std()

        std_x, std_y = self.scale * std_x, self.scale * std_y

        src = np.float32([(ct_x, ct_y), (ct_x + std_x, ct_y + std_y), (ct_x + std_x, ct_y)])
        dst = np.float32([((self.image_size -1 )/ 2.0, (self.image_size -1)/ 2.0), 
				  ((self.image_size-1), (self.image_size -1 )), 
				  ((self.image_size -1 ), (self.image_size - 1)/2.0)])
        retval = cv2.getAffineTransform(src, dst)
        result = cv2.warpAffine(image, retval, (self.image_size, self.image_size), flags = cv2.INTER_LINEAR, borderMode = cv2.BORDER_CONSTANT)
        return result 

    def __getitem__(self,index):#第二步装载数据，返回[img,label]
        if self.test == False:
            image_path =self.img_label[index]['path']
            label = self.img_label[index]['class']
            #img = Image.open( image_path).convert('RGB')
            img = cv2.imread(image_path)
            ldmk = pickle.load(open(image_path.replace('.jpg','_ldmk.pickle'),'rb'))[0]
            img =self.crop_with_ldmk(img, ldmk) 
        else:
            image_path =self.img_label[index]['path']
            label = self.img_label[index]['class']
            #img = Image.open( image_path).convert('RGB')
            img = cv2.imread(image_path)
            ldmk = pickle.load(open(image_path.replace('.jpg','_ldmk.pickle'),'rb'))[0]
            img =self.crop_with_ldmk(img, ldmk) 


       

        if self.transform is not None:
            #print(self.transform)
            img = self.transform(img)
        if 0:
            cv2.imshow('crop face',img) 
            cv2.waitKey()

        if(self.test ==False):
            return np.transpose(np.array(img, dtype = np.float32), (2, 0, 1)), int(label)
        else:
            return np.transpose(np.array(img, dtype = np.float32), (2, 0, 1)), int(label), image_path

    def __len__(self):
        return len(self.img_label)

def blur(img):
    w,h = img.size
    #size=min(h,w)
    img.show()
    img = img.filter(ImageFilter.GaussianBlur(radius=random.random()))
    img.show()
    return img
def maxcrop(img):
    w,h = img.size
    #size=min(h,w)
    img.show()
    img=img.crop(((w-size)//3,(h-size)//3, w-(w-size)//3,h-(h-size)//3))
    #img.show()
    return img
if 0:
    data_transforms = {
    'train' : transforms.Compose([
        #transforms.Lambda(blur),
        transforms.Resize((224,224)) ,
        transforms.RandomHorizontalFlip() ,
       transforms.ToTensor() ,
        transforms.Normalize([0.485 , 0.456 , 0.406] , [0.229 , 0.224 , 0.225])
    ]) ,
    'val' : transforms.Compose([
        transforms.Resize((224,224)) ,
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor() ,
        transforms.Normalize([0.485 , 0.456 , 0.406] , [0.229 , 0.224 , 0.225])
    ]),
    'test' : transforms.Compose([
        transforms.Resize((224,224)) ,
        transforms.ToTensor(),
        transforms.Normalize([0.485 , 0.456 , 0.406] , [0.229 , 0.224 , 0.225])
    ]),
    }
    
    train_data = myData(
			filelists=train_filelists,
			transform = None,
			test = False,
			data_source='none')
    val_data = myData(
			filelists =test_filelists,
			transform =None,
			test = False,data_source = 'none')

    train_loader = Data.DataLoader(dataset=train_data,batch_size = 1,shuffle = True)
    val_loader = Data.DataLoader(dataset = val_data,batch_size = 1,shuffle = True)


    
    for step,batch in enumerate(train_loader):
        data,target = batch
        #data.show()
        if torch.cuda.is_available():
            data,target = data.cuda(),target.cuda()
        data,target = Variable(data, volatile=True), Variable(target)
    
if __name__ == '__main__':
    import os
    import sys
    import pickle
    imgdir = sys.argv[1]
    train_data = myData(
			filelists=train_filelists,
			transform = None,
			test = False,
			data_source='none')
    image = cv2.imread(imgdir)
    landmark = pickle.load(open(imgdir.replace('.jpg','.pickle'),'rb'))[0]
    result=train_data.crop_with_ldmk(image, landmark)
    cv2.imwrite('crop.jpg',result)
    
