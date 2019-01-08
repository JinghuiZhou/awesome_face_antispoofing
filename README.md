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
- Then, in terminal command: *make clean&&make&&python3 main.py train*
## Visualize Dataset
- We have fixed the bug of choice wrong face in multiple detected faces with standard of coordinates. 
- To visualize cropped faces in dataset. Please run command: python3 vis_cropface.py visualize
- All faces will be shown in data/showcropface_train.jpg and data/showcropface_val.jpg
- The training data are shown here.[Training](https://github.com/JinghuiZhou/awesome_face_antispoofing/blob/master/data/showcropface_train.jpg) [Validation](https://github.com/JinghuiZhou/awesome_face_antispoofing/blob/master/data/showcropface_val.jpg)

## Experiments
-  Experiments results on NUAA[1] Image input size is as same as the imagenet.
-  State-of-the-art networks are used, e.g. VGG[2], ResNet[3], DenseNet[4], Inception[5], Xception[6], DetNet[7]

|    Network    | Acc  | AUC  | EER  | TPR(1.0%) | TPR(.5%)| 
|---------------|---|---|---|---|---|
| VGG-11        |  0.9398 | 0.98400675  | 0.059481  | 0.595922  | 0.505393  |
| VGG-13        |  0.9476 | 0.99564721  | 0.030042  | 0.887567  | 0.802796  |
| VGG-16        |  0.7659 | 0.99653556  | 0.029682  | 0.905065  | 0.844735  |
| VGG-19        |  0.7809 | 0.96179324  | 0.105563  | 0.444676  | 0.395925  |
| Res-18        |  0.8759 | 0.99767664  | 0.022308  | 0.944378  | 0.919988  |
| Res-34        |  0.8363 | 0.99806763  | 0.014277  | 0.969363  | 0.859012  |
| Res-50        |  0.9231 | 0.99820910  | 0.013192  | 0.978418  | 0.902439  |
| denseNet121   |  0.9847 | 0.99913086  | 0.015169  | 0.975312  | 0.955384  |
| denseNet161   |  0.8419 | 0.99655236  | 0.027079  | 0.933076  | 0.891731  |
| denseNet169   |  0.9801 | 0.99968893  | 0.004535  | 0.999703  | 0.997323  |
| denseNet201   |  0.9912 | 0.99963239  | 0.008891  | 0.991969  | 0.984838  |
| Xception      |  0.9843 | 0.99973281  | 0.005728  | 0.996431  | 0.993101  |
| DetNet        |  0.9072 | 0.99998322  | 0.000892  | 0.999705  | 0.999703  |
## Checkpoints
-  I have upload trained models into [Baidu DRIVE-trained_models_NUAA](https://pan.baidu.com/s/19hVdkqiiX4aLxRodS9UAZg)
## Reference
- [1]Tan X, Li Y, Liu J, et al. Face liveness detection from a single image with sparse low rank bilinear discriminative model[C]// European Conference on Computer Vision. Springer-Verlag, 2010:504-517.
- [2]Simonyan K, Zisserman A. Very deep convolutional networks for large-scale image recognition[J]. arXiv preprint arXiv:1409.1556, 2014.
- [3]He K, Zhang X, Ren S, et al. Deep residual learning for image recognition[C]//Proceedings of the IEEE conference on computer vision and pattern recognition. 2016: 770-778.
- [4]Huang G, Liu Z, Van Der Maaten L, et al. Densely connected convolutional networks[C]//CVPR. 2017, 1(2): 3.
- [5]Szegedy C, Vanhoucke V, Ioffe S, et al. Rethinking the inception architecture for computer vision[C]//Proceedings of the IEEE conference on computer vision and pattern recognition. 2016: 2818-2826.
- [6]Chollet F. Xception: Deep learning with depthwise separable convolutions[J]. arXiv preprint, 2017: 1610.02357.
- [7]Li Z, Peng C, Yu G, et al. DetNet: A Backbone network for Object Detection[J]. arXiv preprint arXiv:1804.06215, 2018.
