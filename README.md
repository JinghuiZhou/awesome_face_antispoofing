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
-  Experiments results on NUAA[1]

|    Network    | Acc  | AUC  | EER  |
|---------------|---|---|---|
| VGG-11        |  0.9628 |   |   |
| VGG-13        |   |   |   |
| VGG-16        |   |   |   |
| VGG-19        |   |   |   |
| Res-18        |  0.9329 |   |   |
| Res-34        |   |   |   |
| Res-50        |  0.9941 |   |   |
| SENet-1.0     |   |   |   |
| SENet-1.1     |   |   |   |
| DenseNet-121  |   |   |   |
| DenseNet-169  |   |   |   |
| DenseNet-201  |   |   |   |
| Densenet-161  |   |   |   |

## Reference
[1]Tan X, Li Y, Liu J, et al. Face liveness detection from a single image with sparse low rank bilinear discriminative model[C]// European Conference on Computer Vision. Springer-Verlag, 2010:504-517.
