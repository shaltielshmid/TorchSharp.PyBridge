using System.Collections;
using System.IO.Compression;
using Razorvine.Pickle;
using Razorvine.Pickle.Objects;

namespace TorchSharp.PyBridge {
    public static class PyTorchUnpickler {
        static PyTorchUnpickler() {
            Unpickler.registerConstructor("torch._utils", "_rebuild_tensor", new TensorObjectConstructor());
            Unpickler.registerConstructor("torch._utils", "_rebuild_tensor_v2", new TensorObjectConstructor());
            Unpickler.registerConstructor("torch._utils", "_rebuild_parameter", new ParameterObjectConstructor());
            Unpickler.registerConstructor("collections", "OrderedDict", new OrderedDictObjectConstructor());
        }

        /// <summary>
        /// Unpickle the state_dict from a python file saved using `torch.save`
        /// </summary>
        /// <param name="file">Path to the file</param>
        /// <returns>The loaded state_dict</returns>
        public static Hashtable UnpickleStateDict(string file) {
            return UnpickleStateDict(File.OpenRead(file));
        }

        /// <summary>
        /// Unpickle the state_dict from a python file saved using `torch.save`
        /// </summary>
        /// <param name="stream">Stream of the file to load</param>
        /// <param name="leaveOpen">true to leave the stream open after saving the file</param>
        /// <param name="skipTensorRead">true to return descriptor objects and streams instead of tensors so that they can be loaded later</param>
        /// <returns>The loaded state_dict</returns>
        public static Hashtable UnpickleStateDict(Stream stream, bool leaveOpen = false, bool skipTensorRead = false) {
            if (skipTensorRead && !leaveOpen)
                throw new ArgumentException("leaveOpen must be true when skipTensorRead is true");

            // Make sure it's a zip file
            // If it's not, then it was saved using legacy torch save and we don't support it (yet, at least)
            // Check the local file signature
            byte[] signature = new byte[4];
            stream.Read(signature, 0, 4);
            if (signature[0] != 0x50 || signature[1] != 0x4b || signature[2] != 0x03 || signature[3] != 0x04)
                throw new NotImplementedException("The model being loaded was saved using the old PyTorch format and isn't supported in TorchSharp. Please re-save using the new PyTorch format.");

            // Open the archive, since we know it's a zip file
            stream.Seek(0, SeekOrigin.Begin);
            using var archive = new ZipArchive(stream, ZipArchiveMode.Read, leaveOpen);

            // Find the data.pkl file, this is our main file
            var pklEntry = archive.Entries.First(e => e.Name.EndsWith("data.pkl"));

            // Create our unpickler with the archive, so it can pull all the relevant files
            // using the persistentId
            var unpickler = new CustomUnpickler(archive, skipTensorRead);
            // The unpickle returns a hash mapping ["key"] to the tensor
            return (Hashtable)unpickler.load(pklEntry.Open());
        }

        /// <summary>
        /// This class implements custom behavior for unpickling, specifically regarding persistent storage.
        /// In the PyTorch library, instead of serializing the tensors using Pickle, they replace the tensor
        /// objects in the pickle file with the metadata of the tensor, plus a link to an external file in the
        /// archive which contains all the byte data of the tensor.
        /// Therefore, our custom unpickler defines the behavior for restoring the data from the archive
        /// when the unpickler encounters a persistentId.
        /// </summary>
        class CustomUnpickler : Unpickler {
            readonly ZipArchive _archive;

            readonly bool _skipTensorRead;

            public CustomUnpickler(ZipArchive archive, bool skipTensorRead) {
                _archive = archive;
                _skipTensorRead = skipTensorRead;
            }

            protected override object persistentLoad(object pid) {
                // The persistentLoad function in pickler is a way of pickling a key and then loading
                // the data yourself from another source. The `torch.save` function uses this functionality
                // and lists for the pid a tuple with the following items:
                var opid = (object[])pid;

                // Tuple Item0: "storage"
                if ((string)opid[0] != "storage")
                    throw new NotImplementedException("Unknown persistent id loaded");

                // Tuple Item1: storage_type (e.g., torch.LongTensor), which is broken into module=torch, name=LongTensor
                string storageType = ((ClassDictConstructor)opid[1]).name;
                // Tuple Item2: key (filename in the archive)
                string archiveKey = (string)opid[2];
                // Tuple Item3: location (cpu/gpu), but we always load onto CPU.
                // Tuple Item4: numElems (the number of elements in the tensor)

                // Convert the storage name into the relevant scalar type (e.g., LongStorage => torch.long)
                // and then check how many bytes each element is
                var dtype = GetScalarTypeFromStorageName(storageType);

                // Retrieve the entry from the archive
                var entry = _archive.Entries
                    .Select((archiveEntry, index) => (archiveEntry, index))
                    .First(e => e.archiveEntry.FullName.EndsWith($"data/{archiveKey}"));

                // Send this back, so our TensorObjectConstructor can create our torch.tensor from the object.
                return new TensorStream {
                    ArchiveIndex = entry!.index,
                    ArchiveEntry = entry!.archiveEntry,
                    DType = dtype,
                    SkipTensorRead = _skipTensorRead,
                };
            }

            static torch.ScalarType GetScalarTypeFromStorageName(string storage) {
                return storage switch {
                    "DoubleStorage" => torch.float64,
                    "FloatStorage" => torch.@float,
                    "HalfStorage" => torch.half,
                    "LongStorage" => torch.@long,
                    "IntStorage" => torch.@int,
                    "ShortStorage" => torch.int16,
                    "CharStorage" => torch.int8,
                    "ByteStorage" => torch.uint8,
                    "BoolStorage" => torch.@bool,
                    "BFloat16Storage" => torch.bfloat16,
                    "ComplexDoubleStorage" => torch.cdouble,
                    "ComplexFloatStorage" => torch.cfloat,
                    _ => throw new NotImplementedException()
                };
            }
        }

        /// <summary>
        /// The unpickler implementation requires a __setstate__ function for unpickling an ordered dict, due
        /// to the way it was saved. This class is just a regular Hashtable with an implementation for the
        /// __setstate__.
        /// </summary>
        class OrderedDict : Hashtable {
            public void __setstate__(Hashtable arg) {
                foreach (string key in arg.Keys) {
                    if (arg[key] is torch.Tensor)
                        this[key] = arg[key];
                }
            }
        }

        /// <summary>
        /// The PyTorch library stores the parameters in an OrderedDict, and we don't have that class in C#,
        /// so instead we treat it as a Hashtable.
        /// </summary>
        class OrderedDictObjectConstructor : IObjectConstructor {
            public object construct(object[] args) {
                return new OrderedDict();
            }
        }

        /// <summary>
        /// This constructor recreated the behavior from the `torch._utils._rebuild_tensor_V2` method, which
        /// gets all the parameters for a tensor and constructs the tensor with all the relevant properties.
        /// </summary>
        class TensorObjectConstructor : IObjectConstructor {
            public object construct(object[] args) {
                // Arg 0: returned from our custom pickler
                var tensorStream = (TensorStream)args[0];

                var constructor = new TensorConstructorArgs {
                    ArchiveIndex = tensorStream.ArchiveIndex,
                    Data = tensorStream.ArchiveEntry!.Open(),
                    DType = tensorStream.DType,
                    // Arg 1: storage_offset
                    StorageOffset = (int)args[1],
                    // Arg 2: tensor_shape
                    Shape = ((object[])args[2]).Select(i => (long)(int)i).ToArray(),
                    // Arg 3: stride
                    Stride = ((object[])args[3]).Select(i => (long)(int)i).ToArray(),
                    // Arg 4: requires_grad
                    RequiresGrad = (bool)args[4],
                };

                // Arg 5: backward_hooks, we don't support adding them in and it's not recommended
                // in PyTorch to serialize them.

                return tensorStream.SkipTensorRead
                    ? constructor
                    : constructor.ReadTensorFromStream();
            }
        }

        /// <summary>
        /// This object constructor identifies when a torch.nn.Parameter object is being reconstructed from
        /// the pickle file. This is used to identify if the user is trying to load a saved model and not a
        /// saved state dict.
        /// </summary>
        class ParameterObjectConstructor : IObjectConstructor {
            public object construct(object[] args) {
                // If the user got here, that means that he saved the entire model and not the state dictionary
                // And we only support loading the state dict
                throw new NotImplementedException("The file trying to be load contains the entire model and not just the state_dict. Please resave use `torch.save(model.state_dict(), ...)`");
            }
        }

        /// <summary>
        /// This class is an intermediate record which contains all the information to construct the tensor from a stream
        /// provided in the class (usually this stream is set to a DeflateStream from a ZipArchiveEntry). This can be used
        /// to read the tensor directly, or to store a pointer to a tensor to be read at a later time, and avoid excessive memory use. 
        /// </summary>
        internal class TensorConstructorArgs {
            public int ArchiveIndex { get; init; }
            public Stream Data { get; init; }
            public torch.ScalarType DType { get; init; }
            public int StorageOffset { get; init; }
            public long[] Shape { get; init; }
            public long[] Stride { get; init; }
            public bool RequiresGrad { get; init; }

            private bool _alreadyRead = false;
            public torch.Tensor ReadTensorFromStream() {
                if (_alreadyRead)
                    throw new InvalidOperationException("The tensor has already been constructed, cannot read tensor twice.");

                var temp = torch
                    .empty(Shape, DType, device: torch.CPU)
                    .as_strided(Shape, Stride, StorageOffset);
                temp.ReadBytesFromStream(Data);
                Data.Close();

                _alreadyRead = true;
                return temp;
            }
        }

        /// <summary>
        /// When the unpickler first loads in the tensor, it only has access to metadata about the storage
        /// of the tensor, but not the info about stride/shape etc. That part is done in the TensorReconstructor.
        /// Therefore, this class is a simple wrapper for the bytes + dtype of the storage.
        /// </summary>
        class TensorStream {
            public int ArchiveIndex { get; init; }
            public ZipArchiveEntry ArchiveEntry { get; init; }
            public torch.ScalarType DType { get; init; }
            public bool SkipTensorRead { get; init; }
        }
    }
}
