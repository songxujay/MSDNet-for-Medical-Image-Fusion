# MSDNet-for-Medical-Image-Fusion
Accepted in:The 10th International Conference on Image and Graphics(ICIG2019)

## Abstract
Considering the DenseFuse only works in a single scale, we propose a multi-scale DenseNet(MSDNet) for medical image fusion. The main architecture of network is constructed by encoding network, fusion layer and decoding network. To utilize features at different scales, we add a multi-scale mechanism which uses three filters of different sizes to extract features in encoding network. More image details are obtained by increasing the encoding network’s width. Then, we adopt fusion strategy to fuse features of different scales respectively. Finally, the fused image is reconstructed by decoding network. Compared with the existing methods, the proposed method can achieve state-of-the-art fusion performance in objective and subjective assessment.

## The framework of fusion method
<img src="https://github.com/songxujay/MSDNet-for-Medical-Image-Fusion/blob/master/figures/framework.png"  width="600">

## The architecture of MSDNet
<img src="https://github.com/songxujay/MSDNet-for-Medical-Image-Fusion/blob/master/figures/MSDNet.png"  width="600">

## Training
<img src="https://github.com/songxujay/MSDNet-for-Medical-Image-Fusion/blob/master/figures/train.png"  width="600">

## Loss
<img src="https://github.com/songxujay/MSDNet-for-Medical-Image-Fusion/blob/master/figures/compareLoss.png"  width="600">

## Result
<img src="https://github.com/songxujay/MSDNet-for-Medical-Image-Fusion/blob/master/figures/result.png"  width="600">

## Acknowledgement
Many thanks for Professor Xiao-Jun Wu and [Dr.Hui Li](https://github.com/hli1221)


## If you have any question about this code, feel free to reach me(825849512@qq.com)
