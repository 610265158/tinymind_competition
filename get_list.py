# -*- coding:utf-8 -*-

import os
import random

def main(ratio):
    workdir=os.getcwd()
    datadir=os.path.join(workdir,'train')
    name_list=os.listdir(datadir)

    ratio=ratio
    if not os.access('./cvlst',os.F_OK):
        os.mkdir('./cvlst')
    train_list=open('./cvlst/train.lst',mode="w+", encoding='utf-8');
    val_list=open('./cvlst/val.lst',mode="w+", encoding='utf-8');
    label_syntext = open('label_syntext.txt', mode="w+", encoding='utf-8');
    count=0
    count_val=0

    #########save the class with num of data in class
    class_2num={}


    for name in name_list:
        pics_dir=os.path.join(datadir,name)
        pic_list=os.listdir(pics_dir)

        random.shuffle(pic_list)
        train_l=pic_list[0:int(len(pic_list)*ratio)]
        val_l = pic_list[int(len(pic_list) * ratio):]
        for pic in train_l:
            class_2num[name]=class_2num[name]+1 if name in class_2num else 1
            tmp_string=str(count)+'\t'+str((name_list.index(name)))+'\t'+pics_dir+'/'+pic +'\n'
            train_list.write(tmp_string)
            count+=1
        for pic in val_l:
            class_2num[name] += 1 if name in class_2num else 1
            tmp_string = str(count_val) + '\t' + str((name_list.index(name))) + '\t' +pics_dir+'/'+pic +'\n'
            val_list.write(tmp_string)
            count_val+=1


    train_list.close()
    val_list.close()

    ####class 2 label
    print("class name to label:")
    for i in range(len(name_list)):
        print('{}  with label {} '.format(name_list[i], i))
        tmp_str=str(name_list[i])+'\t'+str(i)+'\n'
        label_syntext.write(tmp_str)
    label_syntext.close()
    ####data distribution
    print("\nData distribution:")
    for (class_,count_) in class_2num.items():
        print(class_,': ',count_ ,'  with  train:',int(count_*ratio),' val:',count_-int(count_*ratio))

if __name__=="__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Process train and val list.')
    parser.add_argument('--ratio', dest='ratio',type=float, default=0.8,  help='the ratio between train an val (default: 0.8)')
    args = parser.parse_args()

    main(args.ratio)
