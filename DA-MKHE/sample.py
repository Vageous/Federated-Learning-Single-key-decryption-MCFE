import numpy as np
from torchvision import datasets,transforms
from torch.utils.data import DataLoader,Dataset
import dataset
import option


def mnist_iid(dataset,num_user):
    num_items=int(len(dataset)/num_user)
    dict_users,all_idx={},[i for i in range(len(dataset))]
    for i in range(num_user):
    #numpy.random.choice(a, size=None, replace=True, p=None)
        dict_users[i]=set(np.random.choice(all_idx,num_items,replace=False))
        all_idx=list(set(all_idx)-dict_users[i])
    return dict_users

def mnist_noiid(dataset,num_user):
    num_shards,num_imgs=200,300
    idx_shard=[i for i in range(num_shards)]
    dict_users={i:np.array([],dtype='int64')for i in range(num_user)}
    idxs=np.arange(num_shards*num_imgs)
    labels=dataset.train_labels.numpy()

    idxs_labels=np.vstack((idxs,labels))
    idxs_labels=idxs_labels[:,idxs_labels[1,:].argsort()]
    idxs=idxs_labels[0,:]

    for i in range(num_user):
        rand_set = set(np.random.choice(idx_shard, 2, replace=False))
        idx_shard = list(set(idx_shard) - rand_set)
        for rand in rand_set:
            dict_users[i] = np.concatenate((dict_users[i], idxs[rand*num_imgs:(rand+1)*num_imgs]), axis=0)
    return dict_users


class datasetsplit(Dataset):

    def __init__(self,dataset,idx) -> None:
        self.dataset=dataset
        self.idx=list(idx)

    def __len__(self):
        return len(self.idx)

    def __getitem__(self,item):
        image,label=self.dataset[self.idx[item]]
        return image,label






# if __name__=='__main__':
#     args=option.parser_args()
#     mnist_train,mnist_test=dataset.get_data(args.dataset)
#     # print(mnist_train)
#     idxs=mnist_iid(mnist_train,args.num_user)
#     # 第0个客户端数据集
#     train_set=DataLoader(datasetsplit(mnist_train,idxs[0]),batch_size=args.batchsize,shuffle=True)
#     for batchidx, (image,label) in enumerate(train_set):
#         print(label)

