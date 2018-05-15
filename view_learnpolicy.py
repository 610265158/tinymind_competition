import matplotlib.pyplot as plt
import math
'''
####param:

'''
import argparse
parser = argparse.ArgumentParser(description='Start view lr policy.')
parser.add_argument('--base_lr', dest='base_lr',type=float, default=0.01,  \
                    help='')
parser.add_argument('--max_update', dest='max_update',type=int, default=128,  \
                    help='')
parser.add_argument('--power', dest='power',type=float, default=3,  \
                    help='')

args = parser.parse_args()
print('visualize the lr lr_scheduler, choose a ideal one,')

###lr_scheduler = mx.lr_scheduler.PolyScheduler(8000,0.005, 2)


max_update=args.max_update

base_lr_orig=args.base_lr
power=args.power
x=[]
y=[]

for num_update in range(max_update):
    base_lr = base_lr_orig * pow(1.0 - float(num_update) / float(max_update),power)
    y.append(base_lr)
    x.append(num_update)


plt.plot(x,y)
plt.show()

