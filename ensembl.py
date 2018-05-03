import pandas as pd
import  numpy as np



csv_file=pd.read_csv('label-test1-fake.csv')

name_list=csv_file['filename']

label=csv_file['label']

label_file=open('label_syntext.txt','r')
lines = label_file.readlines()
label_dict={}
for s in lines:
    label_dict[int(s.split('\t')[1])]=s.split('\t')[0]


#######ensembel these files


file_list=['resnext-50with_p_result.csv',
           'resnxet2with_p_result.csv',
           'densenet121with_p_result.csv',
        'densenet169with_p_result.csv'
           ]


csv_list=[]

for item in file_list:
    csv_list.append(pd.read_csv(item,header=None))

mat_list=[]
for csv_item in csv_list:
    mat_list.append(np.asarray(csv_item.as_matrix()))

# for mat in mat_list:
#     mat_list[0] += mat
#
#
# mat=mat_list[0]/len(mat_list)



mat=0.25*mat_list[0]+0.25*mat_list[1]+0.25*mat_list[2]+0.25*mat_list[3]
for i in range(mat.shape[0]):
    result=mat[i,:]
    result = np.squeeze(result)

    result = np.argsort(result)[::-1]


    tmp_str = (label_dict[result[0]] + label_dict[result[1]] + label_dict[result[2]] + label_dict[result[3]] +
               label_dict[result[4]])

    label[i] = tmp_str
    ##top K=5


csv_file.to_csv('ensemble_result.csv',index=None)