using Google.Protobuf.WellKnownTypes;
using System.Collections;
using System.Reflection;
using TorchSharp.Modules;
using static Tensorboard.Summary.Types;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace TorchSharp.PyBridge {
    public static class PyBridgeOptimizerExtensions {

        /// <summary>
        /// Saves the optimizer state to a python-compatible file to be loaded using `torch.load`.
        /// </summary>
        /// <param name="location">The file path.</param>
        public static void save_py(this OptimizerHelper optim, string location) {
            using var stream = System.IO.File.Create(location);
            optim.save_py(stream);
        }

        /// <summary>
        /// Saves the optimizer state to a python-compatible file to be loaded using `torch.load`.
        /// </summary>
        /// <param name="stream">A writable stream instance.</param>
        /// <param name="leaveOpen">true to leave the stream open after saving the file</param>
        /// <returns></returns>
        public static void save_py(this OptimizerHelper optim, System.IO.Stream stream, bool leaveOpen = false) {
            // Construct our state_dict, without the skip parameters
            var sd = optim.state_dict();
            //PyTorchPickler.PickleStateDict(stream, sd, leaveOpen);
        }


        /// <summary>
        /// Load the optimizer state from a file saved using `torch.save`
        /// </summary>
        /// <param name="location">The file path.</param>
        /// <remarks>
        /// This method only supports loading the newer format used by `torch.save`, using a zip file. 
        /// The order in which the parameters were added to the optimizer must be identical when saving and loading.
        /// </remarks>
        public static void load_py(this OptimizerHelper optim, string location) {
            if (!System.IO.File.Exists(location))
                throw new System.IO.FileNotFoundException(location);

            using var stream = System.IO.File.OpenRead(location);
            optim.load_py(stream);
        }

        /// <summary>
        /// Load the optimizer state from a file saved using `torch.save`
        /// </summary>
        /// <param name="stream">A readable stream instance.</param>        
        /// <param name="leaveOpen">true to leave the stream open after saving the file</param>
        /// <remarks>
        /// This method only supports loading the newer format used by `torch.save`, using a zip file. 
        /// The order in which the parameters were added to the optimizer must be identical when saving and loading.
        /// </remarks>
        public static void load_py(this OptimizerHelper optim, System.IO.Stream stream, bool leaveOpen = false) {
            // Unpickle the state dictionary into memory
            var loadedStateDict = PyTorchUnpickler.UnpickleStateDict(stream, leaveOpen);

            // We will get the state dict from the optimizer, and then set the properties using reflection
            var optimStateDict = optim.state_dict();
            
            // The stateDict should have two keys:
            // 1] "param_groups" => equivalent to Options
            var loadedParamGroups = (ArrayList)loadedStateDict["param_groups"]!;
            // Assign all the fields in the param groups
            for (int iOption = 0; iOption < optimStateDict.Options.Count; iOption++) {
                var reference = (Hashtable)loadedParamGroups[iOption]!;
                AssignFields(optimStateDict.Options[iOption], reference);
                AssignProperties(optimStateDict.Options[iOption], reference);
            }
            // 2] "state" => equivalent to State
            var loadedState = (Hashtable)loadedStateDict["state"]!;
            // Assign all the fields in the state (note: we don't have to have all the values in the state)
            for (int iState = 0; iState < optimStateDict.State.Count; iState++) {
                var reference = (Hashtable?)loadedState[iState];
                if (reference is null || reference.Count == 0) continue;

                AssignFields(optimStateDict.State[iState], reference);
                AssignProperties(optimStateDict.State[iState], reference);
            }
        }

        private static void AssignFields<T>(T obj, Hashtable referenceTable) where T : notnull {
            // Go through all the properties
            foreach (var field in obj.GetType().GetFields(BindingFlags.Instance | BindingFlags.Public | BindingFlags.NonPublic)) {
                object? value = GetValue(field.Name, referenceTable);

                // Set the value!
                // If it's a tensor - first dispose the old tensor, and then set the new value
                if (field.FieldType == typeof(torch.Tensor)) {
                    var orig = (torch.Tensor?)field.GetValue(obj);
                    if (orig is not null) 
                        orig?.Dispose();
                }
                field.SetValue(obj, Convert.ChangeType(value, Nullable.GetUnderlyingType(field.FieldType) ?? field.FieldType));
            }// next property
        }

        private static void AssignProperties<T>(T obj, Hashtable referenceTable) where T : notnull {
            // Go through all the properties
            foreach (var property in obj.GetType().GetProperties(BindingFlags.Instance | BindingFlags.Public | BindingFlags.NonPublic)) {
                object? value = GetValue(property.Name, referenceTable);

                // Set the value!
                // If it's a tensor - first dispose the old tensor, and then set the new value
                if (property.PropertyType == typeof(torch.Tensor)) {
                    var orig = (torch.Tensor?)property.GetValue(obj);
                    if (orig is not null)
                        orig?.Dispose();
                }
                property.SetValue(obj, Convert.ChangeType(value, Nullable.GetUnderlyingType(property.PropertyType) ?? property.PropertyType));
            }// next property
        }

        private static object? GetValue(string name, Hashtable referenceTable) {
            // Special handling for betas/eta/step_sizes/step:
            return name switch {
                "LearningRate" or "InitialLearningRate" => referenceTable["lr"],
                "beta1" or "beta2" => ((object[])referenceTable["betas"]!)[name == "beta1" ? 0 : 1],
                "etaminus" or "etaplus" => ((object[])referenceTable["etas"]!)[name == "etaminus" ? 0 : 1],
                "min_step" or "max_step" => ((object[])referenceTable["step_sizes"]!)[name == "min_step" ? 0 : 1],
                "step" => GetValueFromTensor((torch.Tensor)referenceTable["step"]!),
                _ => referenceTable[name!]
            };
        }

        private static object GetValueFromTensor(Tensor tensor) {
            // Stored as a tensor, so return it as a float and dispose the tensor
            var value = tensor.ToSingle();
            tensor.Dispose();
            return value;
        }
    }
}