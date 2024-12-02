# TorchSharp.PyBridge Release Notes

1.4.3:
- Fixed #21: `strict` is not passed to `load_safetensor` in `load_checkpoint` extension

1.4.2:
- PR #20: Optimize load_py for memory and speed (@ejhg)

1.4.1:
- Fixed #17: How to disable tqdm output when loading sharded safetensors

1.4.0:
- Exposed `Safetensors`, `PytorchPickler` and `PytorchUnpickler` to allow for loading/saving python tensors outside of a model.
- Fixed #16: SaveStateDict calls itself recursively and fails on locked file

1.3.2:
- Fixed #13: UnpickleStateDict on BatchNorm2d error

1.3.1:
- Fixed error on Apple Silicon devices

1.3.0:
- Added support for loading tensors that are greater than 2GB (following the update in TorchSharp 0.102.0)
- Added support for loading and saving safetensors when model isn't on CPU.

1.1.0:
- Added `load_py` and `save_py` extensions to optimizers.

1.0.0:
- Initial release of `load_py` and `save_py` extensions for `torch.nn.Module`
