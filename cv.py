# -*- coding:utf-8 -*-

import os
import random

def label():
    workdir=os.getcwd()
    datadir=os.path.join(workdir,'train')
    name_list=os.listdir(datadir)

    train_list=open('alldata.lst',mode="w+", encoding='utf-8');

    label_syntext = open('label_syntext.txt', mode="w+", encoding='utf-8');
    count=0

    #########save the class with num of data in class
    class_2num={}

    for name in name_list:
        pics_dir=os.path.join(datadir,name)
        pic_list=os.listdir(pics_dir)

        random.shuffle(pic_list)
        train_l=pic_list

        for pic in train_l:
            class_2num[name]=class_2num[name]+1 if name in class_2num else 1
            tmp_string=str(count)+'\t'+str((name_list.index(name)))+'\t'+pics_dir+'/'+pic +'\n'
            train_list.write(tmp_string)
            count+=1


    train_list.close()

    ####class 2 label
    print("class name to label:")
    for i in range(len(name_list)):
        print('{}  with label {} '.format(name_list[i], i))
        tmp_str=str(name_list[i])+'\t'+str(i)+'\n'
        label_syntext.write(tmp_str)
    label_syntext.close()

def fold(k):
    train_list = open('alldata.lst', mode="r", encoding='utf-8');
    pse_list = (open('pseduolabel.txt', mode="r", encoding='utf-8')).readlines()

    count=40000
    for i,line in enumerate(pse_list):
        line=line.rstrip()
        line=line.split('\t')
        pse_list[i]=str(count+i)+'\t'+line[-2]+'\t'+line[-1]+'\n'

    img_list=train_list.readlines()+pse_list


    random.shuffle(img_list)

    num=len(img_list)

    if not os.access('./cvlst',os.F_OK):
        os.mkdir('./cvlst')
    for i in range(k):

        train_file=open('./cvlst/train'+str(i)+'.lst',mode='w+',encoding='utf-8')
        val_file = open('./cvlst/val'+str(i)+'.lst', mode='w+', encoding='utf-8')


        val=img_list[int((i/k)*num):int(((i+1)/k)*num)]
        train=img_list[0:int((i/k)*num)]+img_list[int(((i+1)/k)*num):]
        for line in train:
            train_file.write(line)
        for line in val:
            val_file.write(line)

        train_file.close()
        val_file.close()




if __name__=="__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Process train and val list.')
    parser.add_argument('--fold', dest='fold',type=int, default=5,  help='fold')
    args = parser.parse_args()

    label()

    fold(args.fold)