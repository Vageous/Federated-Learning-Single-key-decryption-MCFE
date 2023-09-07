import torch
import model
import sample
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.optim
from option import parser_args
import mkhe_paillier
import dp_mechanism
import transform
import time

class LocalUpdate(object):
    def __init__(self,args,dataset,select_idxs,pk_u,pk_s,pp) -> None:
        self.args=args
        self.dataset=dataset
        self.idxs=select_idxs
        self.num_user=args.num_user
        self.batchsize=args.batchsize
        self.lr=args.lr
        self.local_round=args.local_round
        self.momentum=args.momentum
        self.layer=args.layer
        self.pk_u=pk_u
        self.pk_s=pk_s
        self.pp=pp
        self.flag=args.flag
        self.scale=args.scale
        self.dp_mechanism=args.dp_mechanism
        self.device=torch.device("cuda" if torch.cuda.is_available()==args.device else "cpu")
        self.local_model=model.cnn().to(self.device)
        self.train_set=DataLoader(sample.datasetsplit(self.dataset,select_idxs),batch_size=self.batchsize,shuffle=True)


    def encrypt(self,local_model,):
        User=mkhe_paillier.participant(self.pp)
        for name,params in local_model.items():
            if name==self.layer:
                new_local,index=transform.encode(local_model[name],self.scale)
                local_model[name]=User.encrypt(new_local,self.pk_u,self.pk_s)
        return index

    def decrypt(self,aggre_cipher,sk_u,sk_s,index):
        User=mkhe_paillier.participant(self.pp)
        for name,params in self.local_model.state_dict().items():
            if name==self.layer:
                plain=User.decrypt(aggre_cipher[name],sk_u,sk_s)
                aggre_cipher[name]=transform.decode(plain,index,self.scale)
                aggre_cipher[name]=aggre_cipher[name].reshape(params.shape)
        return aggre_cipher

    def train(self,global_model):
        local_model={}
        for name,param in global_model.state_dict().items():
            self.local_model.state_dict()[name].copy_(param.clone())

        optimizer=torch.optim.SGD(self.local_model.parameters(),lr=self.lr,momentum=self.momentum)

        self.local_model.train()
        for i in range(self.local_round):
            for batch_idx, batch in enumerate(self.train_set):
                image=batch[0].to(self.device)
                label=batch[1].to(self.device)

                optimizer.zero_grad()
                output=self.local_model(image)

                loss=F.cross_entropy(output,label)
                loss.backward()
                if self.dp_mechanism != 'no_dp':
                    dp_mechanism.clip_gradients(self.local_model,self.args)
                optimizer.step()

        if self.flag==0 and self.dp_mechanism=='no_dp':
            return self.local_model.state_dict()
        elif self.flag==0 and self.dp_mechanism!='no_dp':
            dp_mechanism.add_noise(self.local_model,self.args,self.idxs)
            return self.local_model.state_dict()
        else:
            for name,params in self.local_model.state_dict().items():
                local_model[name]=params
            time1=time.time()
            index=self.encrypt(local_model)
            time2=time.time()-time1

            return local_model,index


# if __name__=='__main__':
#     args=parser_args()
#     kgc=IBBE.KGC(args.bits,args.num_user,args.ID)
#     pp=IBBE.params(kgc.n,kgc.N,kgc.g1,kgc.g2,kgc.g3,kgc.hash1,kgc.hash2,kgc.num)
#     new_local=[]
#     User=IBBE.participant(pp)
#     mnist_train,mnist_test=dataset.get_data(args.dataset)
#     idx=sample.mnist_iid(mnist_train,args.num_user)
#     test=LocalUpdate(args,mnist_train,idx[0],kgc.usk2,pp)
#     device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     global_model=model.cnn().to(device)
#     local=test.train(global_model)
#     for name,params in local.items():
#         if name=="conv2.weight":
#             new_local,index=transform.encode(local[name],args.scale)
#             cipher=User.private_enc(new_local,kgc.usk2[0])
#             print(cipher[0])