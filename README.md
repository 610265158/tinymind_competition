# My_Mxnet_toolkit
Use Mxnet do Classification

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

