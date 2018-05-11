# My_Mxnet_toolkit
Use Mxnet do Classification

this code is for https://www.tinymind.cn/competitions/41


## env
ubuntu16.04 mxnet 1.2.0, cuda9, cudnn7, python3

## How to use

#1.release the data to the current directory

#2.python get_list.py --ratio 0.8      //produce train.lst and val.lst

#3.bash ./downmodel.sh                 //download the pretrained model， (mxnet model zoo)

#4.bash ./train.sh                     //start to train

when it converged, chose a good one with high top-5 acc

#python predict.py --epoch 4 

then,get a result with 0.98+.


代码比较糙， :）

#
## Update

### Some improvements were made. Now, 0.989+ for single model,

Do as below:

python get_list.py --ratio 0.9

bash ./downmodel.sh

bash ./train.sh

......chose the best model with good validation top-5 acc, epoch 6 for example.

python predict.py --epoch 6

### show
run: python show.py

then, you can visualize the data after augmentation.

Now, it is very easy to get a result over 0.99, just play with it ：)
#

if there is something wrong, contact me with e-mail: 2120140200@mail.nankai.edu.cn

