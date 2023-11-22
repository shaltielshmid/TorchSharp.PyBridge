using NUnit.Framework;
using System.IO.Compression;
using TorchSharp.PyBridge.Tests;
using static TorchSharp.torch.nn;

namespace TorchSharp.PyBridge.Tests {

    public class TestLoadSaveModules {

        [Test]
        public void TestPythonModuleLoad() {
            // We already saved a python module using `torch.save` to the file `pyload_test.model`
            // Load in that model and make sure that the results are the same
            var model = Sequential(("lin1", Linear(5, 1, hasBias: false)), ("lin2", Linear(1, 2, hasBias: false)));
            model.load_py("pickled_modules/module_load.pth");

            // The weights are all ones, so make sure that if we give it an array of ones we get
            // back as the result [5,5]
            var res = model.forward(torch.tensor(new[] { 1, 1, 1, 1, 1 }).@float());
            Assert.Multiple(() => {
                Assert.That(res[0].ToSingle(), Is.EqualTo(5));
                Assert.That(res[1].ToSingle(), Is.EqualTo(5));
            });
        }

        [Test]
        public void TestPythonModuleSave() {
            // We have a saved pytorch state_dict with all the weights being 2's.
            // Therefore, create that model and save it, and make sure the bytes are equal.

            // Create our model and set the weights
            var model = Sequential(("lin1", Linear(5, 1, hasBias: false)), ("lin2", Linear(1, 2, hasBias: false)));
            model.state_dict()["lin1.weight"].bytes = torch.full(1, 5, 2, torch.ScalarType.Float32).bytes;
            model.state_dict()["lin2.weight"].bytes = torch.full(2, 1, 2, torch.ScalarType.Float32).bytes;

            // Save the model to a memory stream
            var ms = new MemoryStream();
            model.save_py(ms, leaveOpen: true);
            ms.Seek(0, SeekOrigin.Begin);

            // Compare the bytes to pyload_test
            Assert.That(SaveUtils.CompareSavedModules(ms, File.OpenRead("pickled_modules/module_save.bin")), Is.True);
        }




    }

    
}