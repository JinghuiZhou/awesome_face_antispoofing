# awesome_face_antispoofing
- This is a single shot face anti-spoofing project.
- The deep learning framework is Pytorch. Python3.5 is used.
## Face landmarks
- face_alignment is used for landmarks extraction. Page [face_alignment](https://github.com/1adrianb/face-alignment). Thanks to them.
### Landmarks extraction scripts
- cd detlandmark&&python3 detlandmark_imgs.py NUAA_raw_dir
### Data
- I have upload data and detected landmarks into [GOOGLE DRIVE-raw.tar.gz](https://drive.google.com/file/d/1fe80Vo366h4uKylFwsSN3apvLXZZm02L/view?usp=sharing)
- You can change corresponding directory and filename in config.py
- For example train_filelists=[
    ['raw/ClientRaw','raw/client_train_raw.txt',ATTACK],
    ['raw/ImposterRaw','imposter_train_raw.txt',GENUINE]
    ]
   test_filelists=[
    ['raw/ClientRaw','raw/client_test_raw.txt',ATTACK],
    ['raw/ImposterRaw','raw/imposter_test_raw.txt',GENUINE]
    ]
## Method
- Our method is straightforward. Small patched containing a face is cropped with corresponding landmarks. A binary classification network is used to distinguish the attack patches.  
## Training
- First, edit file *config.py*, choose the target network and proper batch_size.
- Then, in terminal command: *make clean&&make&&python3 main.py train*
## Experiments
-  Experiments results on NUAA[1] Image input size is as same as the imagenet.

|    Network    | Acc  | AUC  | EER  | TPR(1.0%) | TPR(.5%)| 
|---------------|---|---|---|---|---|
| VGG-11[2]        |  0.9601 | 0.99319872  | 0.038188  | 0.832243  | 0.778703  |
| VGG-13[2]        |  0.9549 | 0.99473387  | 0.036166  | 0.876264  | 0.824372  |
| VGG-16[2]        |  0.9642 | 0.99509249  | 0.035693  | 0.886377  | 0.805989  |
| VGG-19[2]        |  0.9546 | 0.99366445  | 0.041047  | 0.868531  | 0.829566  |
| Res-18[3]        |  0.9329 | 0.99650029  | 0.026037  | 0.927127  | 0.857467  |
| Res-34[3]        |  0.9568 | 0.99489955  | 0.033501  | 0.877933  | 0.791791  |
| Res-50[3]        |  0.9941 | 0.99981480  | 0.005998  | 0.997271  | 0.992861  |
| DenseNet-121  |  0.9467 | 0.99159944  | 0.046996  | 0.806068  | 0.709994  |
| DenseNet-169  |  0.9817 | 0.99859679  | 0.015622  | 0.961695  | 0.919752  |
| DenseNet-201  |  0.9235 | 0.97937384  | 0.096862  | 0.798037  | 0.713266  |
| Densenet-161  |  0.9540 | 0.99784970  | 0.020363  | 0.953331  | 0.894706  |
| Inception-V3  |  0.9114 | 0.98636584  | 0.061354  | 0.693337  | 0.583299  |
| Xception      |  0.9841 | 0.99929783  | 0.012338  | 0.983641  | 0.970793  |
| DetNet        |  0.9685 | 0.99998105  | 0.001562  | 1.000000  | 0.999703  |

## Reference
- [1]Tan X, Li Y, Liu J, et al. Face liveness detection from a single image with sparse low rank bilinear discriminative model[C]// European Conference on Computer Vision. Springer-Verlag, 2010:504-517.
- [2]Simonyan K, Zisserman A. Very deep convolutional networks for large-scale image recognition[J]. arXiv preprint arXiv:1409.1556, 2014.
- [3]He K, Zhang X, Ren S, et al. Deep residual learning for image recognition[C]//Proceedings of the IEEE conference on computer vision and pattern recognition. 2016: 770-778.
