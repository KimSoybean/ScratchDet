# ScratchDet: Training Single-Shot Object Detectors from Scratch


By [Rui Zhu*](https://kimsoybean.github.io/), [Shifeng Zhang*](http://www.cbsr.ia.ac.cn/users/sfzhang/), [Xiaobo Wang](http://www.cbsr.ia.ac.cn/users/xiaobowang/), [Longyin Wen](http://www.cbsr.ia.ac.cn/users/lywen/), [Hailin Shi†](http://hailin-ai.xyz/),  [Liefeng Bo](https://research.cs.washington.edu/istc/lfb/), [Tao Mei](https://taomei.me/). (\*Equal Contribution, †Corresponding author)

The code is originally based on the [SSD-caffe](https://github.com/weiliu89/caffe/tree/ssd) and [RefineDet-caffe](https://github.com/sfzhang15/RefineDet) framework. We also implement on [mmdetection](https://github.com/open-mmlab/mmdetection). If you want to use one kind of codes, please follow their instructions to finish the initial install. 


Please cite our [paper](https://arxiv.org/abs/1810.08425) in your publications if it helps your research:


```
@inproceedings{zhu2019scratchdet,
  title={ScratchDet: Training Single-Shot Object Detectors From Scratch},
  author={Zhu, Rui and Zhang, Shifeng and Wang, Xiaobo and Wen, Longyin and Shi, Hailin and Bo, Liefeng and Mei, Tao},
  booktitle={CVPR},
  year={2019}
}
```

		
## Introduction

ScratchDet focus on training object detectors from scratch in order to tackle the problems caused by ﬁne-tuning from pretrained networks. In this paper, we study the effects of BatchNorm in the backbone and detection head subnetworks, and successfully train detectors from scratch. By taking the pretaining-free advantage, we are able to explore various architectures for detector designing. Please see our paper for more details.

<div align=center>
<img src="https://raw.githubusercontent.com/KimSoybean/ScratchDet/master/gradient_analysis.png" width="740">
</div>

<div align=center>
Figure : Gradient Analysis.
</div> 

## Codes

You can see details and codes in the folder caffe/ and mmdetection/ .

## Contact

Rui Zhu (zhur5 at mail2.sysu.edu.cn)

Any comments or suggestions are welcome!
