using Google.Protobuf.WellKnownTypes;
using SkiaSharp;
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
            using var d = torch.NewDisposeScope(); // Create a new dispose scope for any tensors we create

            // Get our state_dict from our optimizer
            var sd = optim.state_dict();

            // sd.Options -> ArrayList with all the properties
            var optionsList = new ArrayList();
            for (int iOption = 0; iOption < sd.Options.Count; iOption++) {
                var tgtOption = new Dictionary<object, object>();
                OptimizerUtils.AssignFieldsAndPropsToTargetTable(sd.Options[iOption], tgtOption);
                // Add the params variable, which is created separately
                tgtOption["params"] = sd.StateIndexRef[iOption];

                optionsList.Add(tgtOption);
            }

            // sd.State -> IDictionary with the key being the index
            var stateTable = new Dictionary<object, object>();
            for (int iState = 0; iState < sd.State.Count; iState++) {
                var tgtState = new Dictionary<object, object>();
                OptimizerUtils.AssignFieldsAndPropsToTargetTable(sd.State[iState], tgtState);
                
                stateTable[iState] = tgtState;
            }

            // Add it to the pickle format
            var pickleSd = new Dictionary<object, object> {
                ["param_groups"] = optionsList,
                ["state"] = stateTable
            };

            PyTorchPickler.PickleStateDict(stream, pickleSd, leaveOpen);
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
            // 1] "param_groups" => equivalent to Options (should have the name of options)
            var loadedParamGroups = (ArrayList)loadedStateDict["param_groups"]!;
            if (loadedParamGroups.Count != optimStateDict.Options.Count)
                throw new ArgumentException("Identified a mismatch between between the number parameter groups in the loaded state dict and the current one. Are you sure you added the same number of parameter groups?");

            // Store a mapping between state index in the TorchSharp model to state index in the PyTorch model
            var stateIndexToKeyAndOptions = new Dictionary<int, (int, OptimizerOptions)>();

            // Assign all the fields in the param groups
            for (int iOption = 0; iOption < optimStateDict.Options.Count; iOption++) {
                var reference = (IDictionary)loadedParamGroups[iOption]!;
                OptimizerUtils.AssignFieldsAndPropsFromReferenceTable(optimStateDict.Options[iOption], reference);

                // Map the indicies stored in StateIndexRef to the indicies stored in "params"
                // `referenceIdxs` is the list of keys in the loaded state object. 
                var referenceIdxs = (ArrayList)reference["params"]!;
                // `targetIdxs` is the list of state indexes mapping to this group optoins
                var targetIdxs = optimStateDict.StateIndexRef[iOption];

                // Both `referenceIdxs` and `targetIdxs` should have the same number of parameters (confirm this)
                if (targetIdxs.Count != referenceIdxs.Count)
                    throw new ArgumentException("Identified a mismatch between between the parameter groups in the loaded state dict and the current one. Are you sure you added all the parameter groups the same way?");

                for (int iState = 0; iState < targetIdxs.Count; iState++)
                    stateIndexToKeyAndOptions[targetIdxs[iState]] = ((int)referenceIdxs[iState]!, optimStateDict.Options[iOption]);
            }

            // 2] "state" => equivalent to State
            var loadedState = (IDictionary)loadedStateDict["state"]!;
            // Assign all the fields in the state (note: we don't have to have all the values in the state)
            for (int iState = 0; iState < optimStateDict.State.Count; iState++) {
                // Get the key and the options
                var (stateKey, stateParamOptions) = stateIndexToKeyAndOptions[iState];
                // Retrieve the reference value from the loaded satte
                var reference = (IDictionary?)loadedState[stateKey];

                // If it doesn't exist - that means that the state was never initialized. Reinitialize it with the
                // new parameter groups.
                if (reference is null || reference.Count == 0) {
                    optimStateDict.State[iState].Initialize(stateParamOptions);
                    continue;
                }

                // Assign all the fields from the reference into our state.
                OptimizerUtils.AssignFieldsAndPropsFromReferenceTable(optimStateDict.State[iState], reference);
            }
        }

    }
}