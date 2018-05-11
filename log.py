# -*- coding:utf-8 -*-

import os
import matplotlib.pyplot as plt


log_file=open("log.log","r")


val_acc=[]
train_acc=[]

for line in log_file:
    if "Validation-top_k_accuracy_5=" in line:
        score=line.split("=")[1]
        val_acc.append(score)

    if "	top_k_accuracy_5=" in line:
        score=line.split("=")[1]
        train_acc.append(score)


plt.plot(val_acc)
plt.ylabel('val_top5_accuracy')
plt.xlabel('epoch')
plt.show()

plt.plot(train_acc)
plt.ylabel('train_top5_accuracy')
plt.show()