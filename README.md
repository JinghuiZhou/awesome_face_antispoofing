# awesome_face_antispoofing
- This is a single shot face anti-spoofing project.
- The deep learning framework is Pytorch. Python3.5 is used.
# Installation
- sudo -s
- sh install_requirements.sh
- exit
## Face landmarks
- face_alignment is used for landmarks extraction. Page [face_alignment](https://github.com/1adrianb/face-alignment). Thanks to them.
### Landmarks extraction scripts
- cd detlandmark&&python3 detlandmark_imgs.py NUAA_raw_dir
### Data
- I have upload data and detected landmarks into [GOOGLE DRIVE-raw.tar.gz](https://drive.google.com/file/d/1fe80Vo366h4uKylFwsSN3apvLXZZm02L/view?usp=sharing)
- I have upload data and detected landmarks into [Baidu DRIVE-raw.tar.gz](https://pan.baidu.com/s/1xeW2wJuxGPafgBTqhLKExg)

- You can change corresponding directory and filename in config.py
- For example train_filelists=[
    ['raw/ClientRaw','raw/client_train_raw.txt',GENUINE],
    ['raw/ImposterRaw','imposter_train_raw.txt',ATTACK]
    ]
   test_filelists=[
    ['raw/ClientRaw','raw/client_test_raw.txt',GENUINE],
    ['raw/ImposterRaw','raw/imposter_test_raw.txt',ATTACK]
    ]
## Method
- Our method is straightforward. Small patched containing a face is cropped with corresponding landmarks. A binary classification network is used to distinguish the attack patches.
![alt text](https://github.com/JinghuiZhou/awesome_face_antispoofing/blob/master/pipeline.png "Our Pipeline")
## Training
- First, edit file *config.py*, choose the target network and proper batch_size.
- Then, in terminal command: `make clean&&make&&python3 main.py train`
## Inference
- In terminal command: `python3 inference.py inference --images='detlandmark/inference_images/*/*.jpg'`
- The inference report is result/inference.txt, you can check it in commad: `cat result/inference.txt`
## Visualize Dataset
- We have fixed the bug of choice wrong face in multiple detected faces with standard of coordinates. 
- To visualize cropped faces in dataset. Please run command: `python3 vis_cropface.py visualize`
- All faces will be shown in data/showcropface_train.jpg and data/showcropface_val.jpg
- The training data are shown here. [Training](https://github.com/JinghuiZhou/awesome_face_antispoofing/blob/master/data/showcropface_train.jpg) [Validation](https://github.com/JinghuiZhou/awesome_face_antispoofing/blob/master/data/showcropface_val.jpg)

## Experiments
-  Experiments results on NUAA[1] Image input size is as same as the imagenet.
-  State-of-the-art networks are used, e.g. VGG[2], ResNet[3], DenseNet[4], Inception[5], Xception[6], DetNet[7]

|    Network    | Acc  | AUC  | EER  | TPR(1.0%) | TPR(.5%)| 
|---------------|---|---|---|---|---|
| VGG-11        |  0.9416 | 0.99600562  | 0.031592  | 0.948099  | 0.931262  |
| VGG-13        |  0.9452 | 0.99261419  | 0.034890  | 0.908696  | 0.869814  |
| VGG-16        |  0.9591 | 0.99449404  | 0.027599  | 0.952283  | 0.926575  |
| VGG-19        |  0.9013 | 0.99623176  | 0.023086  | 0.958378  | 0.941503  |
| Res-18        |  0.9813 | 0.99872778  | 0.008158  | 0.992470  | 0.989585  |
| Res-34        |  0.9656 | 0.99978646  | 0.003992  | 0.998091  | 0.996181  |
| Res-50        |  0.8677 | 0.99951550  | 0.008923  | 0.991668  | 0.986544  |
| denseNet121   |  0.9803 | 0.99872628  | 0.014754  | 0.981534  | 0.970144  |
| denseNet161   |  0.9757 | 0.99610439  | 0.016664  | 0.977222  | 0.967020  |
| denseNet169   |  0.9334 | 0.99532942  | 0.029744  | 0.949662  | 0.932130  |
| denseNet201   |  0.9263 | 0.99833348  | 0.012195  | 0.985718  | 0.975525  |
| InceptionV3   |  0.8078 | 0.99172718  | 0.036278  | 0.927270  | 0.907655  |  
| Xception      |  0.9843 | 0.99973281  | 0.005728  | 0.996431  | 0.993101  |
| DetNet        |  0.9072 | 0.99998322  | 0.000892  | 0.999705  | 0.999703  |

## Reference
- [1]Tan X, Li Y, Liu J, et al. Face liveness detection from a single image with sparse low rank bilinear discriminative model[C]// European Conference on Computer Vision. Springer-Verlag, 2010:504-517.
- [2]Simonyan K, Zisserman A. Very deep convolutional networks for large-scale image recognition[J]. arXiv preprint arXiv:1409.1556, 2014.
- [3]He K, Zhang X, Ren S, et al. Deep residual learning for image recognition[C]//Proceedings of the IEEE conference on computer vision and pattern recognition. 2016: 770-778.
- [4]Huang G, Liu Z, Van Der Maaten L, et al. Densely connected convolutional networks[C]//CVPR. 2017, 1(2): 3.
- [5]Szegedy C, Vanhoucke V, Ioffe S, et al. Rethinking the inception architecture for computer vision[C]//Proceedings of the IEEE conference on computer vision and pattern recognition. 2016: 2818-2826.
- [6]Chollet F. Xception: Deep learning with depthwise separable convolutions[J]. arXiv preprint, 2017: 1610.02357.
- [7]Li Z, Peng C, Yu G, et al. DetNet: A Backbone network for Object Detection[J]. arXiv preprint arXiv:1804.06215, 2018.
