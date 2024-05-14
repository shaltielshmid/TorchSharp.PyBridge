using NUnit.Framework;
using System.Diagnostics;
using System.IO.Compression;
using TorchSharp.PyBridge.Tests;
using static Tensorboard.ApiDef.Types;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace TorchSharp.PyBridge.Tests {

    public class TestLoadSaveTensors {
        [Test]
        public void TestPythonTensorsLoad() {
            // We already saved a python dictionary using `torch.save` to the file `tensors.pth`
            var sd = PyTorchUnpickler.UnpickleStateDict("pickled_tensors/tensors.pth");
            // Confirm keys
            Assert.Multiple(() => {
                Assert.That(sd.ContainsKey("arr"), Is.True);
                Assert.That(sd.ContainsKey("arr_2d"), Is.True);
            });
            var arr = (torch.Tensor)sd["arr"]!;
            var arr2D = (torch.Tensor)sd["arr_2d"]!;

            // arr = torch.tensor([1, 2, 3, 4, 5, 6])
            // arr_2d = arr.clone().reshape(2, 3)
            // Confirm type
            Assert.Multiple(() => {
                Assert.That(arr.dtype, Is.EqualTo(ScalarType.Int64));
                Assert.That(arr2D.dtype, Is.EqualTo(ScalarType.Int64));
            });
            // Confirm shape
            Assert.Multiple(() => {
                Assert.That(arr.shape, Is.EquivalentTo(new[] { 6L }));
                Assert.That(arr2D.shape, Is.EquivalentTo(new[] { 2L, 3L }));
            });

            // Confirm content
            Assert.Multiple(() => {
                Assert.That(arr.data<long>().ToArray(), Is.EquivalentTo(new long[] { 1, 2, 3, 4, 5, 6 }));
                Assert.That(arr2D.data<long>().ToArray(), Is.EquivalentTo(new long[] { 1, 2, 3, 4, 5, 6}));
            });
        }

        [Test]
        public void TestPythonTensorsSave() {

            var arr = torch.tensor(new long[] { 1, 2, 3, 4, 5, 6 });
            var arr2D = arr.clone().reshape(2, 3);
            var dict = new Dictionary<string, Tensor>() {
                { "arr", arr },
                { "arr_2d", arr2D }
            };

            // Create a temporary filename to test
            var tempFile = Guid.NewGuid().ToString() + ".pth";
            try {
                // Save the tensors
                PyTorchPickler.PickleStateDict(tempFile, dict);
                
                // Run the python
                var p = new Process();
                p.StartInfo.FileName = "python";
                p.StartInfo.Arguments = "test_torch_tensor_load.py " + tempFile;
                p.StartInfo.UseShellExecute = false;
                p.StartInfo.RedirectStandardOutput = true;

                p.Start();
                p.WaitForExit(10);
                // Read the output
                var output = p.StandardOutput.ReadToEnd().Trim();
                Assert.That(output, Is.EqualTo(
                        "arr" + Environment.NewLine + 
                        "tensor([1, 2, 3, 4, 5, 6])" + Environment.NewLine +
                        "arr_2d" + Environment.NewLine +
                        "tensor([[1, 2, 3]," + Environment.NewLine + "        [4, 5, 6]])")); 
            } finally {
                if (File.Exists(tempFile))
                    File.Delete(tempFile);
            }
        }


        [Test]
        public void TestSafetensorsTensorLoad() {
            // We already saved a the safetensors tensors using `save_file` to the file `tensors.safetensors`
            var sd = Safetensors.LoadStateDict("pickled_tensors/tensors.safetensors");
            // Confirm keys
            Assert.Multiple(() => {
                Assert.That(sd.ContainsKey("arr"), Is.True);
                Assert.That(sd.ContainsKey("arr_2d"), Is.True);
            });
            var arr = sd["arr"]!;
            var arr2D = sd["arr_2d"]!;

            // arr = torch.tensor([1, 2, 3, 4, 5, 6])
            // arr_2d = arr.clone().reshape(2, 3)
            // Confirm type
            Assert.Multiple(() => {
                Assert.That(arr.dtype, Is.EqualTo(ScalarType.Int64));
                Assert.That(arr2D.dtype, Is.EqualTo(ScalarType.Int64));
            });
            // Confirm shape
            Assert.Multiple(() => {
                Assert.That(arr.shape, Is.EquivalentTo(new[] { 6L }));
                Assert.That(arr2D.shape, Is.EquivalentTo(new[] { 2L, 3L }));
            });

            // Confirm content
            Assert.Multiple(() => {
                Assert.That(arr.data<long>().ToArray(), Is.EquivalentTo(new long[] { 1, 2, 3, 4, 5, 6 }));
                Assert.That(arr2D.data<long>().ToArray(), Is.EquivalentTo(new long[] { 1, 2, 3, 4, 5, 6 }));
            });
        }

        [Test]
        public void TestSafetensorsModuleSave() {

            var arr = torch.tensor(new long[] { 1, 2, 3, 4, 5, 6 });
            var arr2D = arr.clone().reshape(2, 3);
            var dict = new Dictionary<string, Tensor>() {
                { "arr", arr },
                { "arr_2d", arr2D }
            };

            // Create a temporary filename to test
            var tempFile = Guid.NewGuid().ToString() + ".pth";
            try {
                // Save the tensors
                Safetensors.SaveStateDict(tempFile, dict);

                // Run the python
                var p = new Process();
                p.StartInfo.FileName = "python";
                p.StartInfo.Arguments = "test_safetensors_tensor_load.py " + tempFile;
                p.StartInfo.UseShellExecute = false;
                p.StartInfo.RedirectStandardOutput = true;

                p.Start();
                p.WaitForExit(10);
                // Read the output
                var output = p.StandardOutput.ReadToEnd().Trim();
                Assert.That(output, Is.EqualTo(
                        "arr" + Environment.NewLine +
                        "tensor([1, 2, 3, 4, 5, 6])" + Environment.NewLine +
                        "arr_2d" + Environment.NewLine +
                        "tensor([[1, 2, 3]," + Environment.NewLine + "        [4, 5, 6]])"));
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