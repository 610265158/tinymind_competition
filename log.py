# -*- coding:utf-8 -*-

import os
import matplotlib.pyplot as plt


log_file=open("log.log","r")


val_acc=[]


for line in log_file:
    if "Validation-accuracy=" in line:
        score=line.split("=")[1]
        val_acc.append(score)



plt.plot(val_acc)
plt.ylabel('accuracy')
plt.ylabel('epoch')
plt.show()