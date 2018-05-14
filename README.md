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

#
## Update
now,it reached aound 0.99 

### show
run: python show.py

then, you can visualize the data after augmentation.
![image](https://github.com/610265158/tinymind_competition/blob/master/show.jpg)

Now, it is very easy to get a result over 0.99, just play with it ：)

### add view lrpolicy
it can help view the lrscheduler, choose a ideal one

for example, in train.py #98 lr_scheduler = mx.lr_scheduler.PolyScheduler(8000,0.01, 3)
run: python view_learnpolicy.py --base_lr 0.01 --max_update 8000 --power 3
![image](https://github.com/610265158/tinymind_competition/blob/master/lr_scheduler.png)
#

if there is something wrong, contact me with e-mail: 2120140200@mail.nankai.edu.cn

