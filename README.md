# MSDNet-for-Medical-Image-Fusion
Published in:The 10th International Conference on Image and Graphics(ICIG2019)

[The paper link](https://link.springer.com/chapter/10.1007/978-3-030-34110-7_24)

## Abstract
Considering the DenseFuse only works in a single scale, we propose a multi-scale DenseNet(MSDNet) for medical image fusion. The main architecture of network is constructed by encoding network, fusion layer and decoding network. To utilize features at different scales, we add a multi-scale mechanism which uses three filters of different sizes to extract features in encoding network. More image details are obtained by increasing the encoding network’s width. Then, we adopt fusion strategy to fuse features of different scales respectively. Finally, the fused image is reconstructed by decoding network. Compared with the existing methods, the proposed method can achieve state-of-the-art fusion performance in objective and subjective assessment.

## The framework of fusion method
![image](https://github.com/songxujay/MSDNet-for-Medical-Image-Fusion/blob/master/figures/framework.png)


## The architecture of MSDNet
![image](https://github.com/songxujay/MSDNet-for-Medical-Image-Fusion/blob/master/figures/MSDNet.png)

## Training
![image](https://github.com/songxujay/MSDNet-for-Medical-Image-Fusion/blob/master/figures/train.png)

## Loss
![image](https://github.com/songxujay/MSDNet-for-Medical-Image-Fusion/blob/master/figures/compareLoss.png)

## Result
![image](https://github.com/songxujay/MSDNet-for-Medical-Image-Fusion/blob/master/figures/result.png)

## Acknowledgement
Many thanks to Professor Xiao-Jun Wu and [Dr.Hui Li](https://github.com/hli1221)
