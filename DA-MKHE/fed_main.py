import model
import torch
import client
import dataset
import central_server
import sample
import numpy as np
import mkhe_paillier
import init
from option import parser_args
from memory_profiler import profile
import sys

# @profile(precision=4,stream=open('./communication/log/total.log','a+'))
def fed_main():
    args=parser_args()
    select_users_num=int(args.frac*args.num_user)
    kgc=mkhe_paillier.KGC(args.bits,args.num_user)
    pp=mkhe_paillier.params(kgc.g,kgc.n,kgc.N,kgc.psi)
    train_set,test_set=dataset.get_data(args.dataset)
    server=central_server.Server(args,test_set,pp)
    global_model=model.cnn().to(device=torch.device("cuda" if torch.cuda.is_available()==args.device else "cpu"))
    if args.iid==0:
        idxs=sample.mnist_iid(train_set,args.num_user)
    else:
        idxs=sample.mnist_noiid(train_set,args.num_user)

    for iter in range(args.epoch):
        weight_model={}
        # select user data for local training
        select_idxs=np.random.choice(range(args.num_user),select_users_num,replace=False)

        if args.flag==0:
            weight_model=init.init(global_model)
        else:
            weight_model=init.init_cipher(global_model,args)

        if args.droptype==0:
            for i in range(len(select_idxs)):
                participant=client.LocalUpdate(args,train_set,idxs[select_idxs[i]],kgc.pk,kgc.pk1,pp)
                if args.flag==0:
                    local_model=participant.train(global_model)
                    server.model_aggregate(local_model,weight_model)
                else:
                    cipher_local,index=participant.train(global_model)
                    server.cipher_model_aggregate(cipher_local,weight_model)

            if args.flag==0:
                server.model_average(weight_model,select_users_num)
                global_model.load_state_dict(weight_model)
            else:
                participant.decrypt(weight_model,kgc.sk[0],kgc.sk1,index)
                server.model_average(weight_model,select_users_num)
                global_model.load_state_dict(weight_model)



        acc,total_loss=server.model_test(global_model)
        # temp.append(acc)
        with open("./Gaussian100.txt",'a+') as f:
            f.write("{},{},{}\n".format(iter,acc,total_loss))
        # print("Epoch %d, acc: %f, loss: %f\n" % (iter, acc, total_loss))
    # return temp

if __name__=='__main__':
    for i in range(5):
        fed_main()