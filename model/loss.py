import torch.nn.functional as F
import torch
from torch.autograd import Variable


def similarity(inputs_):
    # Compute similarity mat of deep feature
    # n = inputs_.size(0)
    sim = torch.matmul(inputs_, inputs_.t())
    return sim

def nll_loss(output, target):
    return F.nll_loss(output, target)


def triplet_loss(inputs, targets):
    n = inputs.size(0)
    # Compute similarity matrix
    sim_mat = similarity(inputs)
    # print(sim_mat)
    targets = targets.cuda()
    # split the positive and negative pairs
    eyes_ = Variable(torch.eye(n, n)).cuda()
    # eyes_ = Variable(torch.eye(n, n))
    pos_mask = targets.expand(n, n).eq(targets.expand(n, n).t())
    neg_mask = eyes_.eq(eyes_) - pos_mask
    pos_mask = pos_mask - eyes_.eq(1)

    pos_sim = torch.masked_select(sim_mat, pos_mask)
    neg_sim = torch.masked_select(sim_mat, neg_mask)

    num_instances = len(pos_sim)//n + 1
    num_neg_instances = n - num_instances

    pos_sim = pos_sim.resize(len(pos_sim)//(num_instances-1), num_instances-1)
    neg_sim = neg_sim.resize(
        len(neg_sim) // num_neg_instances, num_neg_instances)

    #  clear way to compute the loss first
    loss = list()
    c = 0
    for i, pos_pair_ in enumerate(pos_sim):
        # print(i)
        pos_pair_ = torch.sort(pos_pair_)[0]
        neg_pair_ = torch.sort(neg_sim[i])[0]

        neg_pair = torch.masked_select(neg_pair_, neg_pair_ > pos_pair_[0] - self.margin)
        pos_pair = torch.masked_select(pos_pair_, pos_pair_ < neg_pair_[-1] + self.margin)
        # pos_pair = pos_pair[1:]
        if len(neg_pair) < 1:
            c += 1
            continue

        pos_loss = torch.mean(1 - pos_pair)
        neg_loss = torch.mean(neg_pair)
        loss.append(pos_loss + neg_loss)

    loss = torch.sum(torch.stack(loss))/n
    prec = float(c)/n
    neg_d = torch.mean(neg_sim).data[0]
    pos_d = torch.mean(pos_sim).data[0]

    return loss