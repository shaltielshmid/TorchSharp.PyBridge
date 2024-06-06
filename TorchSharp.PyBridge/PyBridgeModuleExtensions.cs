using System.Collections;
using System.IO;
using System.Text.Json;
using System.Text.Json.Nodes;
using TqdmSharp;
using static TorchSharp.torch.nn;

namespace TorchSharp.PyBridge {
    public static class PyBridgeModuleExtensions {

        /// <summary>
        /// Save the parameters and buffers of the module to a python-compatible file to be loaded using `torch.load`.
        /// </summary>
        /// <param name="location">The file path.</param>
        /// <param name="skip">A list of keys not to consider when saving the weights.</param>
        /// <returns></returns>
        public static Module save_py(this Module module, string location, IList<string>? skip = null) {
            using var stream = System.IO.File.Create(location);
            module.save_py(stream, skip);

            return module;
        }

        /// <summary>
        /// Save the parameters and buffers of the module to a python-compatible file to be loaded using `torch.load`.
        /// </summary>
        /// <param name="stream">A writable stream instance.</param>
        /// <param name="skip">A list of keys not to consider when saving the weights.</param>
        /// <param name="leaveOpen">true to leave the stream open after saving the file</param>
        /// <returns></returns>
        public static Module save_py(this Module module, System.IO.Stream stream, IList<string>? skip = null, bool leaveOpen = false) {
            using var d = torch.NewDisposeScope(); // Create a new dispose scope for any tensors we create

            // Construct our state_dict, without the skip parameters
            var sd = module.state_dict();
            if (skip is not null) sd.RemoveKeys(skip);

            PyTorchPickler.PickleStateDict(stream, sd, leaveOpen);

            return module;
        }


        /// <summary>
        /// Save the parameters and buffers of the module to a file using the safetensors format (https://github.com/huggingface/safetensors)
        /// </summary>
        /// <param name="location">The file path.</param>
        /// <param name="skip">A list of keys not to consider when saving the weights.</param>
        /// <returns></returns>
        public static Module save_safetensors(this Module module, string location, IList<string>? skip = null) {
            using var stream = System.IO.File.Create(location);
            module.save_safetensors(stream, skip);

            return module;
        }

        /// <summary>
        /// Save the parameters and buffers of the module to a file using the safetensors format (https://github.com/huggingface/safetensors)
        /// </summary>
        /// <param name="stream">A writable stream instance.</param>
        /// <param name="skip">A list of keys not to consider when saving the weights.</param>
        /// <param name="leaveOpen">true to leave the stream open after saving the file</param>
        /// <returns></returns>
        public static Module save_safetensors(this Module module, System.IO.Stream stream, IList<string>? skip = null, bool leaveOpen = false) {
            using var d = torch.NewDisposeScope(); // Create a new dispose scope for any tensors we create

            // Construct our state_dict, without the skip parameters
            var sd = module.state_dict();
            if (skip is not null) sd.RemoveKeys(skip);

            Safetensors.SaveStateDict(stream, sd, leaveOpen);

            return module;
        }

        /// <summary>
        /// Load the parameters and buffers from a file saved using `torch.save`
        /// </summary>
        /// <param name="location">The file path.</param>
        /// <param name="strict">
        /// If true, will only load a module if it exactly corresponds to the current module's state.
        /// If false, will load the parameters and buffers that it finds in the saved file,
        /// leaving everything else alone.
        /// </param>
        /// <param name="skip">A list of keys not to consider when loading the dictionary.</param>
        /// <param name="loadedParameters">A dictionary to populate with the list of parameters loaded and whether they were matched/skipped. Useful when loading in non-strict mode.</param>
        /// <returns>The module, with parameters and buffers loaded.</returns>
        /// <remarks>
        /// This method only supports loading the newer format used by `torch.save`, using a zip file. 
        /// The model will be fully loaded and all the validation checks will only run after the state
        /// dictionary has been fully loaded. 
        /// </remarks>
        public static Module load_py(this Module module, string location, bool strict = true, IList<string>? skip = null, Dictionary<string, bool>? loadedParameters = null) {
            if (!System.IO.File.Exists(location))
                throw new System.IO.FileNotFoundException(location);

            using var stream = System.IO.File.OpenRead(location);
            module.load_py(stream, strict, skip, loadedParameters);

            return module;
        }

        /// <summary>
        /// Load the parameters and buffers from a file saved using `torch.save`
        /// </summary>
        /// <param name="stream">A readable stream instance.</param>
        /// <param name="strict">
        /// If true, will only load a module if it exactly corresponds to the current module's state.
        /// If false, will load the parameters and buffers that it finds in the saved file,
        /// leaving everything else alone.
        /// </param>
        /// <param name="skip">A list of keys not to consider when loading the dictionary.</param>
        /// <param name="loadedParameters">A dictionary to populate with the list of parameters loaded and whether they were matched/skipped. Useful when loading in non-strict mode.</param>
        /// <param name="leaveOpen">true to leave the stream open after saving the file</param>
        /// <returns>The module, with parameters and buffers loaded.</returns>
        /// <remarks>
        /// This method only supports loading the newer format used by `torch.save`, using a zip file. 
        /// The model will be fully loaded and all the validation checks will only run after the state
        /// dictionary has been fully loaded. 
        /// </remarks>
        public static Module load_py(this Module module, System.IO.Stream stream, bool strict = true, IList<string>? skip = null, Dictionary<string, bool>? loadedParameters = null, bool leaveOpen = false) {
            // Create a dispose score so that we don't keep anyof the loaded tensors past this function
            using var d = torch.NewDisposeScope();
            using var d2 = torch.no_grad(); // To circumvent a bug introduced in 0.102.0

            // Unpickle the state dictionary into memory
            var stateHashtable = PyTorchUnpickler.UnpickleStateDict(stream, leaveOpen);

            // Convert the hashtable to a dictionary of string->tensor
            var stateDict = new Dictionary<string, torch.Tensor>();
            foreach (string key in stateHashtable.Keys)
                stateDict.Add(key, (torch.Tensor)stateHashtable[key]!);

            // Load it in using the builtin function
            var (_, unexpectedKeys) = module.load_state_dict(stateDict, strict, skip);

            // Fill in the loadedParameters dictionary, if relevant
            if (loadedParameters is not null) {
                foreach (string key in stateDict.Keys)
                    loadedParameters[key] = true;
                foreach (string key in unexpectedKeys)
                    loadedParameters[key] = false;
            }

            return module;
        }

        /// <summary>
        /// Load the parameters and buffers from a file saved using the safetensors format (https://github.com/huggingface/safetensors)
        /// </summary>
        /// <param name="location">The file path.</param>
        /// <param name="strict">
        /// If true, will only load a module if it exactly corresponds to the current module's state.
        /// If false, will load the parameters and buffers that it finds in the saved file,
        /// leaving everything else alone.
        /// </param>
        /// <param name="skip">A list of keys not to consider when loading the dictionary.</param>
        /// <param name="loadedParameters">A dictionary to populate with the list of parameters loaded and whether they were matched/skipped. Useful when loading in non-strict mode.</param>
        /// <returns>The module, with parameters and buffers loaded.</returns>
        public static Module load_safetensors(this Module module, string location, bool strict = true, IList<string>? skip = null, Dictionary<string, bool>? loadedParameters = null) {
            if (!System.IO.File.Exists(location))
                throw new System.IO.FileNotFoundException(location);

            using var stream = System.IO.File.OpenRead(location);
            module.load_safetensors(stream, strict, skip, loadedParameters);

            return module;
        }

        /// <summary>
        /// Load the parameters and buffers from a file saved using the safetensors format (https://github.com/huggingface/safetensors)
        /// </summary>
        /// <param name="stream">A readable stream instance.</param>
        /// <param name="strict">
        /// If true, will only load a module if it exactly corresponds to the current module's state.
        /// If false, will load the parameters and buffers that it finds in the saved file,
        /// leaving everything else alone.
        /// </param>
        /// <param name="skip">A list of keys not to consider when loading the dictionary.</param>
        /// <param name="loadedParameters">A dictionary to populate with the list of parameters loaded and whether they were matched/skipped. Useful when loading in non-strict mode.</param>
        /// <param name="leaveOpen">true to leave the stream open after saving the file</param>
        /// <returns>The module, with parameters and buffers loaded.</returns>
        public static Module load_safetensors(this Module module, System.IO.Stream stream, bool strict = true, IList<string>? skip = null, Dictionary<string, bool>? loadedParameters = null, bool leaveOpen = false) {
            // Create a dispose score so that we don't keep anyof the loaded tensors past this function
            using var d = torch.NewDisposeScope();
            using var d2 = torch.no_grad(); // To circumvent a bug introduced in 0.102.0

            // Retrieve the current state dict of the module, so that we can make sure to only load the relevant
            // tensors from the file.
            var curStateDict = module.state_dict();
            if (skip is not null) curStateDict.RemoveKeys(skip);
            // Unlike the pickler format, here we can load in the whole index quickly to check for mismatches
            var index = Safetensors.LoadIndex(stream);
            index.Remove("__metadata__");
            if (skip is not null) index.RemoveKeys(skip);

            if (strict) {
                // Make sure the keys match exactly
                if (index.Count != curStateDict.Count || !index.Keys.All(curStateDict.ContainsKey))
                    throw new InvalidOperationException("The specified state dict is not identical to the target dictionary.");
            }

            // Load in the state dict, only the relevant keys (make sure to reset the position)
            stream.Position = 0;
            var loadedStateDict = Safetensors.LoadStateDict(stream, leaveOpen, curStateDict.Keys.ToList());

            // Load it in using the builtin function
            var (_, unexpectedKeys) = module.load_state_dict(loadedStateDict, strict, skip);
            // Add to the unexpected all the keys in the index that weren't in the state dict
            unexpectedKeys = unexpectedKeys.Concat(index.Keys.Except(curStateDict.Keys)).ToList();

            // Fill in the loadedParameters dictionary, if relevant
            if (loadedParameters is not null) {
                foreach (string key in loadedStateDict.Keys)
                    loadedParameters[key] = true;
                foreach (string key in unexpectedKeys)
                    loadedParameters[key] = false;
            }

            return module;
        }

        /// <summary>
        /// Load the parameters and buffers from a directory containing a potentially sharded checkpoint saved using the regular pytorch format or the safetensors format (https://github.com/huggingface/safetensors).
        /// The filenames are expected to be the way HuggingFace's `save_pretrained` saves them, which are "pytorch_model.bin.index.json" and "model.safetensors.index.json".
        /// Alternatively, one can specify the exact name of the main checkpoint. 
        /// </summary>
        /// <param name="path">A path to a directory containing the checkpoint.</param>
        /// <param name="checkpointName">
        /// Optional; The function defaults to look for a model.safetensors, then pytorch_model.bin, with checking for a sharded index equivalent of them. 
        /// If specified then will use that file instead. If the checkpoint you are specifying is sharded, make sure to point to the index.json file
        /// Note that this parameter should be a filename without a path.</param>
        /// <param name="strict">
        /// If true, will only load a module if it exactly corresponds to the current module's state.
        /// If false, will load the parameters and buffers that it finds in the saved file,
        /// leaving everything else alone.
        /// </param>
        /// <param name="skip">A list of keys not to consider when loading the dictionary.</param>
        /// <param name="loadedParameters">A dictionary to populate with the list of parameters loaded and whether they were matched/skipped. Useful when loading in non-strict mode.</param>
        /// <param name="useTqdm">Display the tqdm progress bar when loading in sharded checkpoint.</param>
        /// <returns>The module, with parameters and buffers loaded.</returns>
        public static Module load_checkpoint(this Module module, string path, string? checkpointName = null, bool strict = true, IList<string>? skip = null, Dictionary<string, bool>? loadedParameters = null, bool useTqdm = true) {
            if (!Directory.Exists(path))
                throw new DirectoryNotFoundException();
            
            // Figure out the name of the checkpoint. If unspecified, try the hierarchy used by huggingface
            if (checkpointName is null) {
                foreach (var potential in new[] { "model.safetensors", "pytorch_model.bin" }) {
                    foreach (var suffix in new[] { "", ".index.json" }) {
                        string name = Path.Combine(path, potential + suffix);
                        if (File.Exists(name)) {
                            checkpointName = potential + suffix;
                            break;
                        }
                    }// next potential suffix

                    if (checkpointName is not null)
                        break;
                }// next potential checkpoint

                if (checkpointName is null)
                    throw new ArgumentException("Couldn't find checkpoint in given directory. Make sure it is named correctly or specify the name of the checkpoint explicitly.");
            }
            else {
                // Make sure that checkpoint name isn't a full path, but just the name of the file
                if (checkpointName.Contains('/') || checkpointName.Contains('\\'))
                    throw new ArgumentException("The checkpoint name should be just the name of a file, not a path", nameof(checkpointName));
            }

            string mainFilename = Path.Combine(path, checkpointName!);
            // If the file ends with .safetensors - load it in using that method
            if (mainFilename.EndsWith(".safetensors")) 
                return module.load_safetensors(mainFilename);
            // If the file doesn't end with .json - try loading it in using the regular pytorch method
            if (!mainFilename.EndsWith(".json"))
                return module.load_py(mainFilename);

            // We have an index json for a sharded file.
            string indexJson = File.ReadAllText(mainFilename);
            var fullIndex = JsonSerializer.Deserialize<Dictionary<string, JsonNode>>(indexJson) ?? throw new NotImplementedException("Invalid JSON encountered when loading in sharded index");
            
            // Extract just the weight map
            if (!fullIndex.ContainsKey("weight_map"))
                throw new NotImplementedException("Invalid JSON encountered when loading in sharded index");

            var weightMap = fullIndex["weight_map"].Deserialize<Dictionary<string, string>>() ?? throw new NotImplementedException("Invalid JSON encountered when loading in sharded index");
            if (skip is not null) weightMap.RemoveKeys(skip);

            // Retrieve the current state dict of the module, so that we can make sure to only load the relevant
            // tensors from the file and to check for strictness
            var curStateDict = module.state_dict();
            if (skip is not null) curStateDict.RemoveKeys(skip);
            
            // If we requested strict - confirm the state dicts match exactly
            if (strict) {
                // Make sure the keys match exactly
                if (weightMap.Count != curStateDict.Count || !weightMap.Keys.All(curStateDict.ContainsKey))
                    throw new InvalidOperationException("The specified state dict is not identical to the target dictionary.");
            }
            // Otherwise, add to the skip list all the parameters that aren't in the state dict. Also remove them from the
            // weight map, so that we won't even look at a file where we don't want to load any of the tensors.
            else {
                skip ??= new List<string>();
                foreach (var badKey in weightMap.Keys.Except(curStateDict.Keys)) {
                    skip.Add(badKey);
                    weightMap.Remove(badKey);
                    loadedParameters?.TryAdd(badKey, false);
                }// next bad key
            }

            if (weightMap.Count == 0)
                return module;

            // Load in each of the files with an optional progress bar progress bar 
            var weightMapFiles = weightMap.Values.ToHashSet();
            var iterWeightMapFiles = useTqdm ? Tqdm.Wrap(weightMapFiles) : weightMapFiles;
            foreach (var key in iterWeightMapFiles) {
                string fullPath = Path.Combine(path, key);
                if (fullPath.EndsWith(".safetensors"))
                    module.load_safetensors(fullPath, false, skip: skip, loadedParameters: loadedParameters); 
                else 
                    module.load_py(fullPath, false, skip: skip, loadedParameters: loadedParameters);
            }

            return module;
        }
    }
}