import torch
import sys

path = sys.argv[1]
sd = torch.load(path)

for (k,v) in sd.items():
	print(k)
	print(v)