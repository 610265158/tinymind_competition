# My_Mxnet_toolkit
Use Mxnet do Classification


#1.把数据放入/data下，不同的类别分别在一个目录，例如 /data/dog; /data/cat

#2. python get_list.py 将生成train.lst 和val.lst

#3. python train.py --fintune 1 --num_class 2
默认的是以mobilenet进行fintune，可以进入train.py 调节参数，也可以用其他的基础网络结构来运行


Myaugmentation.py 可以用来增加数据集增强的方法
Mydataiter.py 可以用来更改数据迭代形式


