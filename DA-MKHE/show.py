import numpy as np
import matplotlib.pyplot as plt
import os


def deal(data):
    final_data=[]
    for i in range(40):
        temp=[]
        for j in range(5):
            temp.append(data[i+j*40])
        final_data.append(temp)
    return final_data

def loaddata(filepath,filename):
    filepath=os.path.join(filepath,filename)
    with open(filepath,'r+') as f:
        lines=f.readlines()
        acc=[]
        for line in lines:
            temp=line.strip().split(',')
            acc.append(float(temp[1]))
    return deal(acc)


def mean_std(data):
    acc_mean,acc_std=[],[]
    for i in range(len(data)):
        acc_mean.append(np.mean(data[i],axis=1))
        acc_std.append(np.std(data[i],axis=1))
    return acc_mean,acc_std


def data_acc():
    filepath="/home/b1107/user/ct/code/DA-MKHE"
    filename=['acc.txt','noencryptacc.txt','Laplace40.txt','Laplace80.txt','Laplace120.txt']
    filename1=['acc.txt','noencryptacc.txt','Gaussian80.txt','Gaussian100.txt','Gaussian120.txt']
    acc,acc1=[],[]
    for i in range(len(filename)):
        acc.append(np.array(loaddata(filepath,filename[i])))
        acc1.append(np.array(loaddata(filepath,filename1[i])))
    return mean_std(acc)[0],mean_std(acc)[1],mean_std(acc1)[0],mean_std(acc1)[1]
    


def figure(acc_mean,acc_std,filename):
    x=np.arange(1,41)
    color = ['#E889BD','#67C2A3','#FC8C63','#8EA0C9','#CBA8A2']
    linestyle = ['-', '-', '--','-.',':']
    marker=['.', '.', '.','.','.','.']
    font1={
    'weight':'semibold',
      'size':11.5}
    label=['DA-MKHE FL','Traditional FL','FL with Laplace I','FL with Laplace II','FL with Laplace III']
    xlabel='Training Rounds '
    ylabel='Model Accuracy(%)' 
    for i in range(len(acc_mean)):
        if i==0:
            plt.plot(x,acc_mean[i],linewidth = 1.5, color = color[i],marker=marker[i],linestyle = linestyle[i],label=label[i])
            plt.fill_between(x,acc_mean[i]-acc_std[i],acc_mean[i]+acc_std[i],color=color[i],alpha=0.3)
        else:
            plt.plot(x,acc_mean[i],linewidth = 1.5, color = color[i],linestyle = linestyle[i],label=label[i])
            plt.fill_between(x,acc_mean[i]-acc_std[i],acc_mean[i]+acc_std[i],color=color[i],alpha=0.3)
    plt.annotate('ε=40', xy=(28,75),xytext=(33, 60),xycoords='data',textcoords='data',arrowprops=dict(arrowstyle="fancy",color='#FC8C63',connectionstyle="arc3,rad=0.3"),fontsize=16)
    plt.annotate('ε=80',xy=(16,86),xytext=(23,50),xycoords='data',textcoords='data',arrowprops=dict(arrowstyle='fancy',color='#8EA0C9',connectionstyle="arc3,rad=0.3" ),fontsize=16)
    plt.annotate('ε=120',xy=(12,91),xytext=(6,50),xycoords='data',textcoords='data',arrowprops=dict(arrowstyle='fancy',color='#CBA8A2',connectionstyle="arc3,rad=0.3" ),fontsize=16)
    plt.xticks(np.arange(5,45,5))
    plt.yticks(np.arange(0,110,10))
    plt.tick_params(labelsize=11)
    plt.xlabel(xlabel,fontsize=16.5)
    plt.ylabel(ylabel,fontsize=16.5)
    plt.grid(ls = '--', lw = 1, color='#D3D3D3')
    plt.legend(loc='lower right',edgecolor='k',prop=font1,ncol=2)
    plt.savefig('Laplace.pdf',dpi=600,format='pdf')
    plt.close()

def figure1(acc_mean,acc_std,filename):
    x=np.arange(1,41)
    color = ['#E889BD','#67C2A3','#FC8C63','#8EA0C9','#CBA8A2']
    linestyle = ['-', '-', '--','-.',':']
    marker=['.', '.', '.','.','.','.']
    font_size=16
    font1={
    'weight':'semibold',
      'size':13}
    label=['DA-MKHE FL','Traditional FL','FL with Gaussian I','FL with Gaussian II','FL with Gaussian III']
    xlabel='Training Rounds '
    ylabel='Model Accuracy(%)' 
    for i in range(len(acc_mean)):
        if i==0:
            plt.plot(x,acc_mean[i],linewidth = 1.5, color = color[i],marker=marker[i],linestyle = linestyle[i],label=label[i])
            plt.fill_between(x,acc_mean[i]-acc_std[i],acc_mean[i]+acc_std[i],color=color[i],alpha=0.3)
        else:
            plt.plot(x,acc_mean[i],linewidth = 1.5, color = color[i],linestyle = linestyle[i],label=label[i])
            plt.fill_between(x,acc_mean[i]-acc_std[i],acc_mean[i]+acc_std[i],color=color[i],alpha=0.3)
    plt.annotate('ε=80', xy=(31,60),xytext=(36, 73),xycoords='data',textcoords='data',arrowprops=dict(arrowstyle="fancy",color='#FC8C63',connectionstyle="arc3,rad=0.3"),fontsize=15)
    plt.annotate('ε=100',xy=(20,88),xytext=(16,60),xycoords='data',textcoords='data',arrowprops=dict(arrowstyle='fancy',color='#8EA0C9',connectionstyle="arc3,rad=0.3" ),fontsize=15)
    plt.annotate('ε=120',xy=(10,90),xytext=(6,50),xycoords='data',textcoords='data',arrowprops=dict(arrowstyle='fancy',color='#CBA8A2',connectionstyle="arc3,rad=0.3" ),fontsize=15)
    plt.xticks(np.arange(5,45,5))
    plt.yticks(np.arange(0,110,10))
    plt.tick_params(labelsize=11)
    plt.xlabel(xlabel,size=font_size)
    plt.ylabel(ylabel,size=font_size)
    plt.grid(ls = '--', lw = 1, color='#D3D3D3')
    plt.legend(loc=(6/40,1/100),edgecolor='k',prop=font1)
    plt.savefig('Gaussian.pdf',dpi=600,format='pdf')

'''
'#E889BD','#67C2A3','#FC8C63','#8EA0C9','#CBA8A2'
'''

def main():
    data=data_acc()
    figure(data[0],data[1],'Laplace')
    figure1(data[2],data[3],'Gaussian')
    # figure(data[2],data[3],'Gaussian',label1)


if __name__=='__main__':
    main()