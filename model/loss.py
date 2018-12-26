import torch.nn.functional as F
import torch
from torch.autograd import Variable
import numpy as np
import pdb

def similarity(inputs_):
    # Compute similarity mat of deep feature
    # n = inputs_.size(0)
    sim = torch.matmul(inputs_, inputs_.t())
    return sim

def nll_loss(output, target,device=torch.device('cpu')):
    return F.nll_loss(output, target)


def triplet_loss(inputs, targets, device=torch.device('cpu'),margin=0):
    n = inputs.size(0)
    # Compute similarity matrix
    sim_mat = similarity(inputs) # n by n
    # print(sim_mat)
    targets = targets.to(device)
    # split the positive and negative pairs
    eyes_ = Variable(torch.eye(n, n)).to(device)
    # eyes_ = Variable(torch.eye(n, n))
    pos_mask = targets.expand(n, n).eq(targets.expand(n, n).t())
    neg_mask = eyes_.eq(eyes_) - pos_mask
    pos_mask = pos_mask - eyes_.eq(1)

    pos_sim = sim_mat.mul(pos_mask.to(device,dtype=torch.float32))
    neg_sim = sim_mat.mul(neg_mask.to(device,dtype=torch.float32))
    max_pos_dist,max_pos_index = torch.max(pos_sim, 0)
    min_neg_dist,min_neg_index = torch.min(neg_sim, 0)
    loss_x = torch.max(max_pos_dist - min_neg_dist + margin,torch.zeros_like(max_pos_dist))
    loss = torch.mean(loss_x)

    return loss



def main():
    data_size = 128
    input_dim = 40
    output_dim = 10
    num_class = 10
    # margin = 0.5
    x = Variable(torch.rand(data_size, input_dim), requires_grad=False)
    # print(x)
    w = Variable(torch.rand(input_dim, output_dim), requires_grad=True)
    inputs = x.mm(w)
    y_ = np.random.randint(10,size=128)
    targets = Variable(torch.IntTensor(y_))
    print(targets.size())
    print(triplet_loss(inputs, targets))


if __name__ == '__main__':
    main()
    print('Congratulations to you!')