using NUnit.Framework;
using System.Diagnostics;
using System.IO.Compression;
using TorchSharp.PyBridge.Tests;
using static TorchSharp.torch.nn;

namespace TorchSharp.PyBridge.Tests {

    public class TestLoadSaveModules {

        [Test]
        public void TestPythonModuleLoad() {
            // We already saved a python module using `torch.save` to the file `module_load.pth`
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
            // Create a sequential model and set the values to be absolute numbers, save the file, and make sure
            // loading the module in pytorch gives us the expected numbers
            var model = Sequential(("lin1", Linear(5, 1, hasBias: false)), ("lin2", Linear(1, 2, hasBias: false)));
            model.state_dict()["lin1.weight"].bytes = torch.full(1, 5, 2, torch.ScalarType.Float32).bytes;
            model.state_dict()["lin2.weight"].bytes = torch.full(2, 1, 2, torch.ScalarType.Float32).bytes;

            // Create a temporary filename to test
            var tempFile = Guid.NewGuid().ToString() + ".pth";
            try {
                // Save the module
                model.save_py(tempFile);

                // Run the python
                var p = new Process();
                p.StartInfo.FileName = "python";
                p.StartInfo.Arguments = "test_torch_load.py " + tempFile;
                p.StartInfo.UseShellExecute = false;
                p.StartInfo.RedirectStandardOutput = true;

                p.Start();
                p.WaitForExit(10);
                // Read the output
                var output = p.StandardOutput.ReadToEnd().Trim();
                Assert.That(output, Is.EqualTo("tensor([20., 20.])"));
            } finally {
                if (File.Exists(tempFile))
                    File.Delete(tempFile);
            }
        }


        [Test]
        public void TestSafetensorsModuleLoad() {
            // We already saved a safetensors module using `safetensors.torch.save_file` to the file `module_load.safetensors`
            // Load in that model and make sure that the results are the same
            var model = Sequential(("lin1", Linear(5, 2, hasBias: false)), ("lin2", Linear(2, 2, hasBias: false)));
            model.load_safetensors("pickled_modules/module_load.safetensors");

            // The weights are ones for lin1 and twos for lin2. Therefore, for the input of (11, 11, 11, 11, 11) we should get
            // back the result of [220,200]
            var res = model.forward(torch.tensor(new[] { 11, 11, 11, 11, 11 }).@float());
            Assert.Multiple(() => {
                Assert.That(res[0].ToSingle(), Is.EqualTo(220));
                Assert.That(res[1].ToSingle(), Is.EqualTo(220));
            });
        }

        [Test]
        public void TestSafetensorsModuleSave() {
            // Create a sequential model and set the values to be absolute numbers, save the file, and make sure
            // loading the safetensors in python gives us the expected numbers
            var model = Sequential(("lin1", Linear(5, 1, hasBias: false)), ("lin2", Linear(1, 2, hasBias: false)));
            model.state_dict()["lin1.weight"].bytes = torch.full(1, 5, 2, torch.ScalarType.Float32).bytes;
            model.state_dict()["lin2.weight"].bytes = torch.full(2, 1, 2, torch.ScalarType.Float32).bytes;

            // Create a temporary filename to test
            var tempFile = Guid.NewGuid().ToString() + ".safetensors";
            try {
                // Save the module
                model.save_safetensors(tempFile);

                // Run the python
                var p = new Process();
                p.StartInfo.FileName = "python";
                p.StartInfo.Arguments = "test_safetensors_load.py " + tempFile;
                p.StartInfo.UseShellExecute = false;
                p.StartInfo.RedirectStandardOutput = true;

                p.Start();
                p.WaitForExit(10);
                // Read the output
                var output = p.StandardOutput.ReadToEnd().Trim();
                Assert.That(output, Is.EqualTo("tensor([20., 20.])"));
            }
            finally {
                if (File.Exists(tempFile))
                    File.Delete(tempFile);
            }


        }

        [Test]
        public void TestLoadBatchNorm2D_Bug13() {
            var model = BatchNorm2d(5);
            // Run a few inputs through, to increment the `num_batches_tracked` field
            for (int i = 0; i < 5; i++) {
                using var d = torch.NewDisposeScope();
                model.forward(torch.rand(new[] { 5L, 5L, 5L, 5L }));
            }

            Assert.That(model.num_batches_tracked.item<long>() == 5);

            // Save to a file, and try reloading
            using var stream = new MemoryStream();
            model.save_py(stream, leaveOpen: true);
            stream.Position = 0;

            // Create the new model and load it
            var model2 = BatchNorm2d(5);
            model2.load_py(stream);

            Assert.That(model2.num_batches_tracked.item<long>() == 5);
        }
    }

    
}