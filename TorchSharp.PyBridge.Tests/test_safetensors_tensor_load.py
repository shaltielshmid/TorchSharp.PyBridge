from safetensors.torch import load_file
import sys

path = sys.argv[1]
tensors = load_file(path)

for (k,v) in tensors.items():
	print(k)
	print(v)