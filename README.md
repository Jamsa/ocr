# ocr

fork 自：https://github.com/jimmyleaf/ocr_tensorflow_cnn


数据集为发票编号，截取自不同省份的增值税发票，样本的图像大小和数字字体都不统一。

链接:https://pan.baidu.com/s/17457sXD-dsA2iWOKK6-xYw  密码:9uq5

ocr1.py跑到3000 batch左右能得到较好的效果。

num_hidden 从64调整至512能得到更快的收敛速度，在500 batch左右就可以得到较好的效果。