# awesome_face_antispoofing
- The deep learning framework is Pytorch. Python3.5 is used.
## Face landmarks
- face_alignment is used for landmarks extraction. Page [face_alignment](https://github.com/1adrianb/face-alignment). Thanks to them.
### Landmarks extraction scripts
- cd detlandmark&&python3 detlandmark_imgs.py NUAA_raw_dir
## Method
- Our method is straightforward. Small patched containing a face is cropped with corresponding landmarks. A binary classification network is used to distinguish the attack patches.  
## Training
- First, edit file *config.py*, choose the target network and proper batch_size.
- Then, in terminal command: *make clean&&make&&python3 main.py train*
## Experiments
-  Experiments results on NUAA[1] Image input size is (224,224).

|    Network    | Acc  | AUC  | EER  | TPR(1.0%) | TPR(.5%)| 
|---------------|---|---|---|---|---|
| VGG-11        |  0.9601 | 0.99319872  | 0.038188  | 0.832243  | 0.778703  |
| VGG-13        |  0.9549 |   |   |   |   |
| VGG-16        |  0.9642 |   |   |   |   |
| VGG-19        |  0.9546 |   |   |   |   |
| Res-18        |  0.9329 |   |   |   |   |
| Res-34        |  0.9568 |   |   |   |   |
| Res-50        |  0.9941 |   |   |   |   |
| DenseNet-121  |  0.9467 |   |   |   |   |
| DenseNet-169  |  0.9817 |   |   |   |   |
| DenseNet-201  |  0.9235 |   |   |   |   |
| Densenet-161  |  0.9540 |   |   |   |   |

## Reference
[1]Tan X, Li Y, Liu J, et al. Face liveness detection from a single image with sparse low rank bilinear discriminative model[C]// European Conference on Computer Vision. Springer-Verlag, 2010:504-517.
