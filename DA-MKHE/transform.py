# Given a tensor which type is float as input
# transfrom it to a list, which type is int and the elements of it are both integer
import torch

def encode(tensor,scale):
    mark=[]
    new_list=torch.LongTensor((tensor*scale).cpu().numpy()).reshape(-1).tolist()
    for i in range(len(new_list)):
        if new_list[i]<0:
            new_list[i]=-new_list[i]
            mark.append(i)

    return new_list,mark

def decode(list,index,scale):
    
    for i in range(len(index)):
        list[index[i]]=-list[index[i]]
    new_tensor=torch.FloatTensor(torch.tensor(list).numpy() / scale)
    return new_tensor