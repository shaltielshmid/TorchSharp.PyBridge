import torch
from torch.optim import *
from torch.nn import Linear, Sequential

# base sequence for all of these
l1 = Linear(10, 10, bias=True);
l2 = Linear(10, 10, bias=True);
seq = Sequential(l1, l2);

# Go through the list of all the optimizers and make sure they all managed to 
# load both the regular and withloss saved files
for (optim, name) in [(Adadelta, "Adadelta"), (Adagrad, "Adagrad"), (Adam, "Adam"), (Adamax, "Adamax"), (AdamW, "AdamW"), (ASGD, "ASGD"), (NAdam, "NAdam"), (RAdam, "RAdam"), (RMSprop, "RMSProp"), (Rprop, "Rprop"), (SGD, "SGD")]:
    opt = optim(seq.parameters(), 0.99) # init with 0.99 to make sure it changed
    assert opt.param_groups[0]['lr'] == 0.99
     
    # load and make sure the lr changed
    opt.load_state_dict(torch.load(name + '_save.bin'))
    assert opt.param_groups[0]['lr'] != 0.99

    # load the after loss model, and confirm that it loads without error
    opt.load_state_dict(torch.load(name + '_withloss_save.bin'))
    # TODO: check that the parameters were actually loaded

