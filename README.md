# TorchSharp.PyBridge

[![NuGet](https://img.shields.io/nuget/v/TorchSharp.PyBridge.svg)](https://www.nuget.org/packages/TorchSharp.PyBridge/)

TorchSharp.PyBridge is an extension library for [TorchSharp](https://github.com/dotnet/TorchSharp), providing seamless interoperability between .NET and Python for model serialization. It simplifies the process of saving and loading PyTorch models in a .NET environment, enabling developers to easily develop models in both .NET and Python and transfer models easily.

## Features

- `module.load_py(...)`, `optim.load_py(...)`: Extension method for modules and optimizers for easily loading PyTorch models saved in the standard Python format (using `torch.save`) directly into TorchSharp.

    > This only works for when the `state_dict` was saved and not the whole model, see example below.

- `module.save_py(...)`, `optim.save_py(...)`: Extension method for modules and optimizers for easily saving TorchSharp models in a format that can be directly loaded in PyTorch (using `torch.load`), offering cross-platform model compatibility.

- `module.load_safetensors(...)`, `module.save_safetensors(...)`: Extension methods for modules for easily saving and loading model weights using the [safetensors](https://github.com/huggingface/safetensors) format. 

- `module.load_checkpoint(...)`: Extension method for loading in a checkpoint (both safetensors and regular pytorch, including sharded models) from a directory saved using HuggingFace's `PreTrainedModel.save_pretrained()` method.  

## Getting Started

### Installation

TorchSharp.PyBridge is available on NuGet. You can install it using the following command:

#### .NET CLI
```bash
dotnet add package TorchSharp.PyBridge
```

#### NuGet Package Manager
```powershell
Install-Package TorchSharp.PyBridge
```

### Prerequisites

- .NET SDK
- TorchSharp library

## Usage

### Loading a PyTorch Model in .NET

Saving the model in Python:

```python
import torch 

model = ...
torch.save(model.state_dict(), 'path_to_your_model.pth')
```

Loading it in C#:

```csharp
using TorchSharp.PyBridge;

var model = ...;
model.load_py("path_to_your_model.pth");
```

### Saving a TorchSharp Model for PyTorch

To save a model in a format compatible with PyTorch:

```csharp
using TorchSharp.PyBridge;

var model = ...;
model.save_py("path_to_save_model.pth");
```

And loading it in in Python:

```python
import torch

model = ...
model.load_state_dict(torch.load('path_to_save_model.pth'))
```

## Contributing

Contributions to TorchSharp.PyBridge are welcome. 

## Acknowledgments

This project makes use of the `pickle` library, a Java and .NET implementation of Python's pickle serialization protocol, developed by Irmen de Jong. The `pickle` library plays a vital role in enabling the serialization features within TorchSharp.PyBridge. We extend our thanks to the developer for their significant contributions to the open-source community. For more details about the `pickle` library, please visit their [GitHub repository](https://github.com/irmen/pickle).

## Support and Contact

For support, questions, or feedback, please open an issue in the [GitHub repository](https://github.com/shaltielshmid/TorchSharp.PyBridge).
