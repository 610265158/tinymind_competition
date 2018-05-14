import matplotlib.pyplot as plt
'''
####param:
set 


'''

print('visualize the lr lr_scheduler, choose a ideal one,')

###lr_scheduler = mx.lr_scheduler.PolyScheduler(8000,0.005, 2)


max_update=8000

base_lr_orig=0.01
power=3
x=[]
y=[]

for num_update in range(max_update):
    base_lr = base_lr_orig * pow(1.0 - float(num_update) / float(max_update),power)
    y.append(base_lr)
    x.append(num_update)


plt.plot(x,y)
plt.show()