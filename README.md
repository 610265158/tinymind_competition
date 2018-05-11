# My_Mxnet_toolkit
Use Mxnet do Classification

this code is for https://www.tinymind.cn/competitions/41


## env
ubuntu16.04 mxnet 1.2.0, cuda9, cudnn7, python3

## How to use

#1.把数据解压缩放在当前目录

#2.python get_list.py --ratio 0.8 将生成train.lst 和val.lst

#3.bash ./downmodel.sh 下载预训练模型， 也可以去mxnet model zoo去下载

#4.bash ./train.sh

结束后， 选择一个不错的epoch，这里选择第四个epoch

#python predict.py --epoch 4 输出结果

#
你可以得到一个0.98+以上的结果


代码比较糙， :）

#
## Update
Some improvement were made. Now, 0.989+ for single model,

Do as below:

python get_list.py --ratio 0.9

bash ./downmodel.sh

bash ./train.sh

......chose the best model with good validation top-5 acc, epoch 6 for example

python predict.py --epoch 6


#
if there is something wrong, contact me with e-mail: 2120140200@mail.nankai.edu.cn

