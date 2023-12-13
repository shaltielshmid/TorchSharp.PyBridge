import torch
from torch.nn import Linear
from torch.optim import *

def calc_loss(opt, linears):
    opt.zero_grad()
    out = torch.rand(10)
    for lin in linears: out = lin(out)
    torch.nn.functional.mse_loss(out, torch.rand(10)).backward()
    opt.step()

def save_sgd():
    lin = Linear(10, 10)
    opt = SGD(lin.parameters(), 0.01, 0.1)
    calc_loss(opt, [lin])
    torch.save(opt.state_dict(), 'sgd_load.pth')
save_sgd()

def save_asgd():
    lin = Linear(10, 10)
    opt = ASGD(lin.parameters(), 0.01, 1e-3, 0.65, 1e5)
    calc_loss(opt, [lin])
    torch.save(opt.state_dict(), 'asgd_load.pth')
save_asgd()

def save_rmsprop():
    lin1 = Linear(10, 10)
    lin2 = Linear(10, 10)
    opt = RMSprop(lin1.parameters(), 0.001, momentum=0.1)
    opt.add_param_group(dict(params=lin2.parameters(), lr=0.01, momentum=0, centered=True))
    calc_loss(opt, [lin1, lin2])
    torch.save(opt.state_dict(), 'rmsprop_load.pth')
save_rmsprop()

def save_rprop():
    lin1 = Linear(10, 10)
    lin2 = Linear(10, 10)
    opt = Rprop(lin1.parameters(), lr=0.001, etas=(0.35, 1.5), step_sizes=(1e-5, 5), maximize=False)
    opt.add_param_group(dict(params=lin2.parameters(), etas=(0.45, 1.5), step_sizes=(1e-5, 5), lr=0.01, maximize=True))
    calc_loss(opt, [lin1, lin2])
    torch.save(opt.state_dict(), 'rprop_load.pth')
save_rprop()

def save_adam():
    lin1 = Linear(10, 10)
    lin2 = Linear(10, 10)
    opt = Adam(lin1.parameters(), lr=0.001, betas=(0.8, 0.9), amsgrad=False)
    opt.add_param_group(dict(params=lin2.parameters(), betas=(0.7, 0.79), lr=0.01, amsgrad=True))
    calc_loss(opt, [lin1, lin2])
    torch.save(opt.state_dict(), 'adam_load.pth')
save_adam()

def save_adamw():
    lin1 = Linear(10, 10)
    lin2 = Linear(10, 10)
    opt = AdamW(lin1.parameters(), lr=0.001, betas=(0.8, 0.9), amsgrad=False)
    opt.add_param_group(dict(params=lin2.parameters(), betas=(0.7, 0.79), lr=0.01, amsgrad=True))
    calc_loss(opt, [lin1, lin2])
    torch.save(opt.state_dict(), 'adamw_load.pth')
save_adamw()

def save_nadam():
    lin1 = Linear(10, 10)
    lin2 = Linear(10, 10)
    opt = NAdam(lin1.parameters(), lr=0.001, betas=(0.8, 0.9), weight_decay=0)
    opt.add_param_group(dict(params=lin2.parameters(), betas=(0.7, 0.79), lr=0.01, weight_decay=0.3))
    calc_loss(opt, [lin1, lin2])
    torch.save(opt.state_dict(), 'nadam_load.pth')
save_nadam()

def save_radam():
    lin1 = Linear(10, 10)
    lin2 = Linear(10, 10)
    opt = RAdam(lin1.parameters(), lr=0.001, betas=(0.8, 0.9), weight_decay=0)
    opt.add_param_group(dict(params=lin2.parameters(), betas=(0.7, 0.79), lr=0.01, weight_decay=0.3))
    calc_loss(opt, [lin1, lin2])
    torch.save(opt.state_dict(), 'radam_load.pth')
save_radam()

def save_adadelta():
    lin1 = Linear(10, 10)
    lin2 = Linear(10, 10)
    opt = Adadelta(lin1.parameters(), lr=0.001, rho=0.85, weight_decay=0.3, maximize=False)
    opt.add_param_group(dict(params=lin2.parameters(), rho=0.79, lr=0.01, weight_decay=0.3, maximize=True))
    calc_loss(opt, [lin1, lin2])
    torch.save(opt.state_dict(), 'adadelta_load.pth')
save_adadelta()

def save_adagrad():
    lin = Linear(10, 10)
    opt = Adagrad(lin.parameters(), lr=0.001, lr_decay=0.85, weight_decay=0.3)
    calc_loss(opt, [lin])
    torch.save(opt.state_dict(), 'adagrad_load.pth')
save_adagrad()

def save_adamax():
    lin1 = Linear(10, 10)
    lin2 = Linear(10, 10)
    opt = Adamax(lin1.parameters(), lr=0.001, betas=(0.8, 0.9), weight_decay=0)
    opt.add_param_group(dict(params=lin2.parameters(), betas=(0.7, 0.79), lr=0.01, weight_decay=0.3))
    calc_loss(opt, [lin1, lin2])
    torch.save(opt.state_dict(), 'adamax_load.pth')
save_adamax()

def save_adam_empty_state():
    lin1 = Linear(10, 10)
    lin2 = Linear(10, 10)
    lin3 = Linear(10, 10)
    opt = Adam(lin1.parameters(), lr=0.001, betas=(0.8, 0.9), amsgrad=False)
    opt.add_param_group(dict(params=lin2.parameters(), betas=(0.7, 0.79), lr=0.01, amsgrad=True))
    opt.add_param_group(dict(params=lin3.parameters(), betas=(0.6, 0.69), lr=0.01, amsgrad=True))
    calc_loss(opt, [lin1, lin2])
    torch.save(opt.state_dict(), 'adam_emptystate_load.pth')
save_adam_empty_state()