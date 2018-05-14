import pandas as pd
import  numpy as np
import  random
import os
workdir = os.getcwd()


csv_file=pd.read_csv('label-test1-fake.csv')

name_list=csv_file['filename']

label=csv_file['label']

label_file=open('label_syntext.txt','r')
lines = label_file.readlines()
label_dict={}
for s in lines:
    label_dict[int(s.split('\t')[1])]=s.split('\t')[0]


#######ensembel these files

import argparse
parser = argparse.ArgumentParser(description='Start train.')

parser.add_argument('--bagging', dest='bagging', type=int, default=0,
                    help='using bagging, and 1 for avg; 2 for vote; 3 for rand')
parser.add_argument('--stacking', dest='stacking', type=int, default=0,
                    help='using stacking, no implement')
parser.add_argument('--boost', dest='boost', type=int, default=0,
                    help='using xgboost')
args = parser.parse_args()



##load the syntext
label_file=open('label_syntext.txt','r')
lines = label_file.readlines()
label_dict={}
id_to_label={}
for s in lines:
    label_dict[int(s.split('\t')[1])]=s.split('\t')[0]
    id_to_label[(s.split('\t')[0])] = int(s.split('\t')[1])


if args.bagging:
    file_list=['resnext50_1with_p_result.csv',
               'resnext50-2with_p_result.csv',
               'resnext50-3with_p_result.csv',
                'resnext50-4with_p_result.csv'
               ]


    csv_list=[]

    for item in file_list:
        csv_list.append(pd.read_csv(item,header=None))

    mat_list=[]
    for csv_item in csv_list:
        mat_list.append(np.asarray(csv_item.as_matrix()))




    if args.bagging==1:
        for mat in mat_list:
            mat_list[0] += mat

        mat=mat_list[0]/len(mat_list)

        # mat=0.25*mat_list[0]+0.25*mat_list[1]
        for i in range(mat.shape[0]):
            result=mat[i,:]
            result = np.squeeze(result)

            result = np.argsort(result)[::-1]

            tmp_str = (label_dict[result[0]] + label_dict[result[1]] + label_dict[result[2]] + label_dict[result[3]] +
                       label_dict[result[4]])

            label[i] = tmp_str
            ##top K=5


        csv_file.to_csv('bagging_avg_ensemble_result.csv',index=None)

    ##############semi
    pseduolabel=0
    if pseduolabel:
        pseduolabel_list = open('pseduolabel.txt', mode="w+", encoding='utf-8');

        pseduolabel_count = 36000

        csv_file=pd.read_csv('label-test1-fake.csv')

        name_list=csv_file['filename']
        label=csv_file['label']
        for i, singlepic in enumerate(name_list):

            result = mat[i, :]
            result = np.squeeze(result)

            result = np.argsort(result)[::-1]


            tmp_str_pseduolabel = str(pseduolabel_count) + '\t' + str(
                    result[0]) + '\t' + workdir+'/test1/' + singlepic + '\n'
            pseduolabel_list.write(tmp_str_pseduolabel)
            pseduolabel_count += 1
        pseduolabel_list.close()

    if args.bagging==2:

        ####may be better...
        for mat in mat_list:
            mat_list[0] += mat
            ######

        from collections import Counter

        for i in range(mat_list[0].shape[0]):
            vote=[]
            for mat in mat_list:
                res=np.argsort(np.squeeze(mat[i,:]))[::-1][0:5]
                res=res.tolist()
                vote=vote+res

            vote_counts = Counter(vote)
            res=vote_counts.most_common(5)

            tmp_str = (label_dict[res[0][0]] + label_dict[res[1][0]] + label_dict[res[2][0]] + label_dict[res[3][0]] +
                       label_dict[res[4][0]])

            label[i] = tmp_str
            ##top K=5

        csv_file.to_csv('bagging_vote_ensemble_result.csv', index=None)

    if args.bagging == 3:
        random_length=7
        lst=[0,1,2,3,4,5,6]
        mat=np.zeros(shape=mat_list[0].shape)
        for i in range(mat.shape[0]):
            num=random.randint(6,7)
            index_tmp=random.sample(lst, num)
            print(index_tmp)


            for j in index_tmp:
                mat[i,:]+=mat_list[j][i,:]

        mat/=random_length

        # mat=0.25*mat_list[0]+0.25*mat_list[1]
        for i in range(mat.shape[0]):
            result=mat[i,:]
            result = np.squeeze(result)

            result = np.argsort(result)[::-1]

            tmp_str = (label_dict[result[0]] + label_dict[result[1]] + label_dict[result[2]] + label_dict[result[3]] +
                       label_dict[result[4]])

            label[i] = tmp_str
            ##top K=5

        csv_file.to_csv('bagging_random_ensemble_result.csv',index=None)




if args.boost==1:
    ##use xgboost to boot:


    import numpy as np
    import xgboost as xgb
    import pandas as pd
    # label need to be 0 to num_class -1
    data = pd.read_csv('resnext-50with_p-label_result.csv')

    num_class=100
    #####process the input data

    tmp_str=[]
    for i in range(100):
        tmp_str.append(str(i))
    tmp_str.append('label')
    data=np.asarray(data[tmp_str])

    random.shuffle(data)

    ratio=0.8
    train_X=data[0:int(0.8*data.shape[0]),0:100]
    train_Y=data[0:int(0.8*data.shape[0]),100]
    test_X=data[int(0.8*data.shape[0]):,0:100]
    test_Y=data[int(0.8*data.shape[0]):,100]



    xg_train = xgb.DMatrix(train_X, label=train_Y)
    xg_test = xgb.DMatrix(test_X, label=test_Y)
    # setup parameters for xgboost

    # use softmax multi-class classification

    # scale weight of positive examples

    params = {'eta': 0.01,
              'max_leaves': 5,
              'max_depth': 6,
              'subsample': 0.9,
              'objective': 'multi:softmax',
              'alpha':0.02,
              'silent': True,
              'num_class':num_class,
              'eval_metric':'merror'}

    watchlist = [(xg_train, 'train'), (xg_test, 'test')]
    num_round = 20
    bst = xgb.train(params, xg_train, num_round, watchlist)
    # get prediction
    pred = bst.predict(xg_test)
    error_rate = np.sum(pred != test_Y) / test_Y.shape[0]
    print('Test error using softmax = {}'.format(error_rate))

    #do the same thing again, but output probabilities
    params['objective'] = 'multi:softprob'
    bst = xgb.train(params, xg_train, num_round, watchlist)
    # Note: this convention has been changed since xgboost-unity
    # get prediction, this is in 1D array, need reshape to (ndata, nclass)
    pred_prob = bst.predict(xg_test).reshape(test_Y.shape[0], num_class)
    pred_label = np.argmax(pred_prob, axis=1)
    error_rate = np.sum(pred_label != test_Y) / test_Y.shape[0]
    print('Test error using softprob = {}'.format(error_rate))

