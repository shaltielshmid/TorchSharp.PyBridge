import torch
from torch.nn import *
from collections import OrderedDict

# start by building our model
model = Sequential(OrderedDict([("lin1", Linear(5, 1, bias=False)), ("lin2", Linear(1, 2, bias=False))]))

import sys
path = sys.argv[1]
model.load_state_dict(torch.load(path))

with torch.no_grad():
	print(model.forward(torch.tensor([1, 1, 1, 1, 1]).float()))