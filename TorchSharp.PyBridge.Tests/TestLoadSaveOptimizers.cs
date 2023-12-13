using NUnit.Framework;
using System.IO.Compression;
using TorchSharp.Modules;
using TorchSharp.PyBridge.Tests;
using static TorchSharp.torch.nn;

namespace TorchSharp.PyBridge.Tests {
    public class TestLoadSaveOptimizers {

        private void TestSaveOptim<T>(Func<IEnumerable<Parameter>, T> func, bool withLoss = false) where T : OptimizerHelper {
            // Set the manual seed so that the randoms don't change between runs
            // and our tests will succeed
            torch.manual_seed(423812);

            var l1 = Linear(10, 10, true);
            var l2 = Linear(10, 10, true);
            var seq = Sequential(l1, l2);

            torch.manual_seed(423812);
            var optim = func(seq.parameters());

            // Force the buffers that are only created after loss to be created.
            if (withLoss) {
                using (var d = torch.NewDisposeScope()) {
                    optim.zero_grad();
                    var x = torch.randn(new[] { 64L, 10L });
                    var y = torch.randn(new[] { 64L, 10L });
                    torch.nn.functional.mse_loss(seq.call(x), y).backward();
                    optim.step();
                }
            }

            // Save that optim to memory
            using var ms = new MemoryStream();
            optim.save_py(ms, leaveOpen: true);
            ms.Position = 0;

            // Create a new optimizer, and load in the values, and make sure they are the same
            torch.manual_seed(423812);
            var optim2 = func(seq.parameters());
            optim2.load_py(ms, leaveOpen: true);

            // Save the second optim to a memory stream, and make sure they are identical
            using var ms2 = new MemoryStream();
            optim2.save_py(ms2, leaveOpen: true);

            // Compare the bytes to pyload_test
            ms.Position = 0; ms2.Position = 0;
            Assert.That(SaveUtils.CompareSavedModules(ms, ms2));
        }

        [Test]
        public void TestSaveRprop() {
            TestSaveOptim(p => torch.optim.Rprop(p));
        }

        [Test]
        public void TestSaveRpropWithLoss() {
            TestSaveOptim(p => torch.optim.Rprop(p), true);
        }

        [Test]
        public void TestSaveSGD() {
            TestSaveOptim(p => torch.optim.SGD(p, 0.01));
        }

        [Test]
        public void TestSaveSGDWithLoss() {
            TestSaveOptim(p => torch.optim.SGD(p, 0.01), true);
        }

        [Test]
        public void TestSaveASGD() {
            TestSaveOptim(p => torch.optim.ASGD(p));
        }

        [Test]
        public void TestSaveASGDWithLoss() {
            TestSaveOptim(p => torch.optim.ASGD(p), true);
        }

        [Test]
        public void TestSaveRMSProp() {
            TestSaveOptim(p => torch.optim.RMSProp(p));
        }

        [Test]
        public void TestSaveRMSPropWithLoss() {
            TestSaveOptim(p => torch.optim.RMSProp(p), true);
        }

        [Test]
        public void TestSaveRAdam() {
            TestSaveOptim(p => torch.optim.RAdam(p));
        }

        [Test]
        public void TestSaveRAdamWithLoss() {
            TestSaveOptim(p => torch.optim.RAdam(p), true);
        }

        [Test]
        public void TestSaveNAdam() {
            TestSaveOptim(p => torch.optim.NAdam(p));
        }

        [Test]
        public void TestSaveNAdamWithLoss() {
            TestSaveOptim(p => torch.optim.NAdam(p), true);
        }

        [Test]
        public void TestSaveAdam() {
            TestSaveOptim(p => torch.optim.Adam(p));
        }

        [Test]
        public void TestSaveAdamWithLoss() {
            TestSaveOptim(p => torch.optim.Adam(p), true);
        }

        [Test]
        public void TestSaveAdamW() {
            TestSaveOptim(p => torch.optim.AdamW(p));
        }

        [Test]
        public void TestSaveAdamWWithLoss() {
            TestSaveOptim(p => torch.optim.AdamW(p), true);
        }

        [Test]
        public void TestSaveAdamax() {
            TestSaveOptim(p => torch.optim.Adamax(p));
        }

        [Test]
        public void TestSaveAdamaxWithLoss() {
            TestSaveOptim(p => torch.optim.Adamax(p), true);
        }

        [Test]
        public void TestSaveAdagrad() {
            TestSaveOptim(p => torch.optim.Adagrad(p));
        }

        [Test]
        public void TestSaveAdagradWithLoss() {
            TestSaveOptim(p => torch.optim.Adagrad(p), true);
        }

        [Test]
        public void TestSaveAdadelta() {
            TestSaveOptim(p => torch.optim.Adadelta(p));
        }

        [Test]
        public void TestSaveAdadeltaWithLoss() {
            TestSaveOptim(p => torch.optim.Adadelta(p), true);
        }


        [Test]
        public void TestLoadSGD() {
            var lin = torch.nn.Linear(10, 10);

            double learning_rate = 0.00004f;
            var optimizer = torch.optim.SGD(lin.parameters(), learning_rate);

            optimizer.load_py("pickled_optimizers/sgd_load.pth");

            var sd = optimizer.state_dict();
            Assert.Multiple(() => {
                Assert.That(sd.Options, Has.One.Items);
                Assert.That(sd.State, Has.Exactly(2).Items);
            });

            foreach (var opts in sd.Options) {
                var options = opts as Modules.SGD.Options;
                Assert.That(options!.momentum, Is.EqualTo(0.1));
                Assert.That(options!.LearningRate, Is.Not.EqualTo(learning_rate));
            }

            foreach (var st in sd.State) {
                var state = st as Modules.SGD.State;
                Assert.NotNull(state!.momentum_buffer);
            }
        }

        [Test]
        public void TestLoadASGD() {
            var lin = torch.nn.Linear(10, 10);

            double learning_rate = 0.004f;
            var optimizer = torch.optim.ASGD(lin.parameters(), learning_rate);

            optimizer.load_py("pickled_optimizers/asgd_load.pth");

            var sd = optimizer.state_dict();
            Assert.Multiple(() => {
                Assert.That(sd.Options, Has.One.Items);
                Assert.That(sd.State, Has.Exactly(2).Items);
            });
            foreach (var opts in sd.Options) {
                var options = opts as Modules.ASGD.Options;
                Assert.Multiple(() => {
                    Assert.That(options!.alpha, Is.EqualTo(0.65));
                    Assert.That(options!.lambd, Is.EqualTo(1e-3));
                    Assert.That(options!.t0, Is.EqualTo(1e5));
                    Assert.That(options!.LearningRate, Is.Not.EqualTo(learning_rate));
                });
            }

            foreach (var st in sd.State) {
                var state = st as Modules.ASGD.State;
                Assert.That(state!.step, Is.EqualTo(1));
                Assert.NotNull(state!.ax);
            }
        }

        [Test]
        public void TestLoadRMSprop() {
            var lin1 = torch.nn.Linear(10, 10);
            var lin2 = torch.nn.Linear(10, 10);

            var seq = Sequential(("lin1", lin1), ("lin2", lin2));

            var pgs = new RMSProp.ParamGroup[] {
                new () { Parameters = lin1.parameters(), Options = new() { LearningRate = 0.00005 } },
                new (lin2.parameters(), lr: 0.00003, centered: false, momentum: 0.25)
            };

            double learning_rate = 0.00004f;
            var optimizer = torch.optim.RMSProp(pgs, learning_rate);

            optimizer.load_py("pickled_optimizers/rmsprop_load.pth");

            var sd = optimizer.state_dict();
            Assert.Multiple(() => {
                Assert.That(sd.Options, Has.Count.EqualTo(2));
                Assert.That(sd.State, Has.Count.EqualTo(4));
            });

            var options = (sd.Options[0] as Modules.RMSProp.Options)!;
            Assert.Multiple(() => {
                Assert.That(options.momentum, Is.EqualTo(0.1));
                Assert.That(options.LearningRate, Is.EqualTo(0.001));
                Assert.That(options.centered, Is.False);
            });

            options = (sd.Options[1] as Modules.RMSProp.Options)!;
            Assert.Multiple(() => {
                Assert.That(options.momentum, Is.EqualTo(0));
                Assert.That(options.LearningRate, Is.EqualTo(0.01));
                Assert.That(options.centered, Is.True);
            });

            var state = (sd.State[0] as Modules.RMSProp.State)!;
            Assert.Multiple(() => {
                Assert.That(state.step, Is.EqualTo(1));
                Assert.That(state.square_avg, Is.Not.Null);
                Assert.That(state.momentum_buffer, Is.Not.Null);
                Assert.That(state.grad_avg, Is.Null);
            });

            state = (sd.State[1] as Modules.RMSProp.State)!;
            Assert.Multiple(() => {
                Assert.That(state.step, Is.EqualTo(1));
                Assert.That(state.square_avg, Is.Not.Null);
                Assert.That(state.momentum_buffer, Is.Not.Null);
                Assert.That(state.grad_avg, Is.Null);
            });

            state = (sd.State[2] as Modules.RMSProp.State)!;
            Assert.Multiple(() => {
                Assert.That(state.step, Is.EqualTo(1));
                Assert.That(state.square_avg, Is.Not.Null);
                Assert.That(state.momentum_buffer, Is.Null);
                Assert.That(state.grad_avg, Is.Not.Null);
            });

            state = (sd.State[3] as Modules.RMSProp.State)!;
            Assert.Multiple(() => {
                Assert.That(state.step, Is.EqualTo(1));
                Assert.That(state.square_avg, Is.Not.Null);
                Assert.That(state.momentum_buffer, Is.Null);
                Assert.That(state.grad_avg, Is.Not.Null);
            });
        }

        [Test]
        public void TestLoadRprop() {
            var lin1 = torch.nn.Linear(10, 10);
            var lin2 = torch.nn.Linear(10, 10);

            var seq = Sequential(("lin1", lin1), ("lin2", lin2));

            var pgs = new Rprop.ParamGroup[] {
                new () { Parameters = lin1.parameters(), Options = new() { LearningRate = 0.00005 } },
                new (lin2.parameters(), lr: 0.00003, maximize: false)
            };

            double learning_rate = 0.00004f;
            var optimizer = torch.optim.Rprop(pgs, learning_rate);

            optimizer.load_py("pickled_optimizers/rprop_load.pth");

            var sd = optimizer.state_dict();
            Assert.Multiple(() => {
                Assert.That(sd.Options, Has.Count.EqualTo(2));
                Assert.That(sd.State, Has.Count.EqualTo(4));
            });

            var options = (sd.Options[0] as Modules.Rprop.Options)!;
            Assert.Multiple(() => {
                Assert.That(options.etaminus, Is.EqualTo(0.35));
                Assert.That(options.etaplus, Is.EqualTo(1.5));
                Assert.That(options.min_step, Is.EqualTo(1e-5));
                Assert.That(options.max_step, Is.EqualTo(5));
                Assert.That(options.LearningRate, Is.EqualTo(0.001));
                Assert.That(options.maximize, Is.False);
            });

            options = (sd.Options[1] as Modules.Rprop.Options)!;
            Assert.Multiple(() => {
                Assert.That(options.etaminus, Is.EqualTo(0.45));
                Assert.That(options.etaplus, Is.EqualTo(1.5));
                Assert.That(options.min_step, Is.EqualTo(1e-5));
                Assert.That(options.max_step, Is.EqualTo(5));
                Assert.That(options.LearningRate, Is.EqualTo(0.01));
                Assert.That(options.maximize, Is.True);
            });

            foreach (var st in sd.State) {
                var state = (sd.State[0] as Modules.Rprop.State)!;
                Assert.Multiple(() => {
                    Assert.That(state.step, Is.EqualTo(1));
                    Assert.That(state.prev, Is.Not.Null);
                    Assert.That(state.step_size, Is.Not.Null);
                });
            }
        }

        [Test]
        public void TestLoadAdam() {
            var lin1 = torch.nn.Linear(10, 10);
            var lin2 = torch.nn.Linear(10, 10);

            var pgs = new Adam.ParamGroup[] {
                new () { Parameters = lin1.parameters(), Options = new() { LearningRate = 0.00005 } },
                new (lin2.parameters(), lr: 0.00003, amsgrad: false, beta1: 0.25)
            };

            double learning_rate = 0.00004f;
            var optimizer = torch.optim.Adam(pgs, learning_rate);

            optimizer.load_py("pickled_optimizers/adam_load.pth");

            var sd = optimizer.state_dict();
            Assert.Multiple(() => {
                Assert.That(sd.Options, Has.Count.EqualTo(2));
                Assert.That(sd.State, Has.Count.EqualTo(4));
            });

            var options = (sd.Options[0] as Modules.Adam.Options)!;
            Assert.Multiple(() => {
                Assert.That(options.beta1, Is.EqualTo(0.8));
                Assert.That(options.beta2, Is.EqualTo(0.9));
                Assert.That(options.LearningRate, Is.EqualTo(0.001));
                Assert.That(options.amsgrad, Is.False);
            });

            options = (sd.Options[1] as Modules.Adam.Options)!;
            Assert.Multiple(() => {
                Assert.That(options.beta1, Is.EqualTo(0.7));
                Assert.That(options.beta2, Is.EqualTo(0.79));
                Assert.That(options.LearningRate, Is.EqualTo(0.01));
                Assert.That(options!.amsgrad, Is.True);
            });

            var state = (sd.State[0] as Modules.Adam.State)!;
            Assert.Multiple(() => {
                Assert.That(state.step, Is.EqualTo(1));
                Assert.That(state.exp_avg, Is.Not.Null);
                Assert.That(state.exp_avg_sq, Is.Not.Null);
                Assert.That(state.max_exp_avg_sq, Is.Null);
            });

            state = (sd.State[1] as Modules.Adam.State)!;
            Assert.Multiple(() => {
                Assert.That(state.step, Is.EqualTo(1));
                Assert.That(state.exp_avg, Is.Not.Null);
                Assert.That(state.exp_avg_sq, Is.Not.Null);
                Assert.That(state.max_exp_avg_sq, Is.Null);
            });

            state = (sd.State[2] as Modules.Adam.State)!;
            Assert.Multiple(() => {
                Assert.That(state.step, Is.EqualTo(1));
                Assert.That(state.exp_avg, Is.Not.Null);
                Assert.That(state.exp_avg_sq, Is.Not.Null);
                Assert.That(state.max_exp_avg_sq, Is.Not.Null);
            });

            state = (sd.State[3] as Modules.Adam.State)!;
            Assert.Multiple(() => {
                Assert.That(state.step, Is.EqualTo(1));
                Assert.That(state.exp_avg, Is.Not.Null);
                Assert.That(state.exp_avg_sq, Is.Not.Null);
                Assert.That(state.max_exp_avg_sq, Is.Not.Null);
            });
        }

        [Test]
        public void TestLoadAdamW() {
            var lin1 = torch.nn.Linear(10, 10);
            var lin2 = torch.nn.Linear(10, 10);

            var pgs = new AdamW.ParamGroup[] {
                new () { Parameters = lin1.parameters(), Options = new() { LearningRate = 0.00005 } },
                new (lin2.parameters(), lr: 0.00003, amsgrad: false, beta1: 0.25)
            };

            double learning_rate = 0.00004f;
            var optimizer = torch.optim.AdamW(pgs, learning_rate);

            optimizer.load_py("pickled_optimizers/adamw_load.pth");

            var sd = optimizer.state_dict();
            Assert.Multiple(() => {
                Assert.That(sd.Options, Has.Count.EqualTo(2));
                Assert.That(sd.State, Has.Count.EqualTo(4));
            });

            var options = (sd.Options[0] as Modules.AdamW.Options)!;
            Assert.Multiple(() => {
                Assert.That(options.beta1, Is.EqualTo(0.8));
                Assert.That(options.beta2, Is.EqualTo(0.9));
                Assert.That(options.LearningRate, Is.EqualTo(0.001));
                Assert.That(options.amsgrad, Is.False);
            });

            options = (sd.Options[1] as Modules.AdamW.Options)!;
            Assert.Multiple(() => {
                Assert.That(options.beta1, Is.EqualTo(0.7));
                Assert.That(options.beta2, Is.EqualTo(0.79));
                Assert.That(options.LearningRate, Is.EqualTo(0.01));
                Assert.That(options.amsgrad, Is.True);
            });


            var state = (sd.State[0] as Modules.AdamW.State)!;
            Assert.Multiple(() => {
                Assert.That(state.step, Is.EqualTo(1));
                Assert.That(state.exp_avg, Is.Not.Null);
                Assert.That(state.exp_avg_sq, Is.Not.Null);
                Assert.That(state.max_exp_avg_sq, Is.Null);
            });

            state = (sd.State[1] as Modules.AdamW.State)!;
            Assert.Multiple(() => {
                Assert.That(state.step, Is.EqualTo(1));
                Assert.That(state.exp_avg, Is.Not.Null);
                Assert.That(state.exp_avg_sq, Is.Not.Null);
                Assert.That(state.max_exp_avg_sq, Is.Null);
            });

            state = (sd.State[2] as Modules.AdamW.State)!;
            Assert.Multiple(() => {
                Assert.That(state.step, Is.EqualTo(1));
                Assert.That(state.exp_avg, Is.Not.Null);
                Assert.That(state.exp_avg_sq, Is.Not.Null);
                Assert.That(state.max_exp_avg_sq, Is.Not.Null);
            });

            state = (sd.State[3] as Modules.AdamW.State)!;
            Assert.Multiple(() => {
                Assert.That(state.step, Is.EqualTo(1));
                Assert.That(state.exp_avg, Is.Not.Null);
                Assert.That(state.exp_avg_sq, Is.Not.Null);
                Assert.That(state.max_exp_avg_sq, Is.Not.Null);
            });
        }

        [Test]
        public void TestLoadNAdam() {
            var lin1 = torch.nn.Linear(10, 10);
            var lin2 = torch.nn.Linear(10, 10);

            var pgs = new NAdam.ParamGroup[] {
                new () { Parameters = lin1.parameters(), Options = new() { LearningRate = 0.00005 } },
                new (lin2.parameters(), lr: 0.00003, beta1: 0.25, weight_decay: 0.1)
            };

            double learning_rate = 0.00004f;
            var optimizer = torch.optim.NAdam(pgs, learning_rate);

            optimizer.load_py("pickled_optimizers/nadam_load.pth");

            var sd = optimizer.state_dict();
            Assert.Multiple(() => {
                Assert.That(sd.Options, Has.Count.EqualTo(2));
                Assert.That(sd.State, Has.Count.EqualTo(4));
            });

            var options = (sd.Options[0] as Modules.NAdam.Options)!;
            Assert.Multiple(() => {
                Assert.That(options.beta1, Is.EqualTo(0.8));
                Assert.That(options.beta2, Is.EqualTo(0.9));
                Assert.That(options.LearningRate, Is.EqualTo(0.001));
                Assert.That(options.weight_decay, Is.EqualTo(0));
            });

            options = (sd.Options[1] as Modules.NAdam.Options)!;
            Assert.Multiple(() => {
                Assert.That(options.beta1, Is.EqualTo(0.7));
                Assert.That(options.beta2, Is.EqualTo(0.79));
                Assert.That(options.LearningRate, Is.EqualTo(0.01));
                Assert.That(options.weight_decay, Is.EqualTo(0.3));
            });

            foreach (var st in sd.State) {
                var state = (st as Modules.NAdam.State)!;
                Assert.Multiple(() => {
                    Assert.That(state.step, Is.EqualTo(1));
                    Assert.That(state.exp_avg, Is.Not.Null);
                    Assert.That(state.exp_avg_sq, Is.Not.Null);
                });
            }
        }

        [Test]
        public void TestLoadingRAdam() {
            var lin1 = torch.nn.Linear(10, 10);
            var lin2 = torch.nn.Linear(10, 10);

            var pgs = new RAdam.ParamGroup[] {
                new () { Parameters = lin1.parameters(), Options = new() { LearningRate = 0.00005 } },
                new (lin2.parameters(), lr: 0.00003, beta1: 0.25)
            };

            double learning_rate = 0.00004f;
            var optimizer = torch.optim.RAdam(pgs, learning_rate);

            optimizer.load_py("pickled_optimizers/radam_load.pth");

            var sd = optimizer.state_dict();
            Assert.Multiple(() => {
                Assert.That(sd.Options, Has.Count.EqualTo(2));
                Assert.That(sd.State, Has.Count.EqualTo(4));
            });

            var options = (sd.Options[0] as Modules.RAdam.Options)!;
            Assert.Multiple(() => {
                Assert.That(options.beta1, Is.EqualTo(0.8));
                Assert.That(options.beta2, Is.EqualTo(0.9));
                Assert.That(options.LearningRate, Is.EqualTo(0.001));
                Assert.That(options.weight_decay, Is.EqualTo(0));
            });

            options = (sd.Options[1] as Modules.RAdam.Options)!;
            Assert.Multiple(() => {
                Assert.That(options.beta1, Is.EqualTo(0.7));
                Assert.That(options.beta2, Is.EqualTo(0.79));
                Assert.That(options.LearningRate, Is.EqualTo(0.01));
                Assert.That(options.weight_decay, Is.EqualTo(0.3));
            });

            foreach (var st in sd.State) {
                var state = (st as Modules.RAdam.State)!;
                Assert.Multiple(() => {
                    Assert.That(state.step, Is.EqualTo(1));
                    Assert.That(state.exp_avg, Is.Not.Null);
                    Assert.That(state.exp_avg_sq, Is.Not.Null);
                });
            }
        }

        [Test]
        public void TestLoadAdadelta() {
            var lin1 = torch.nn.Linear(10, 10);
            var lin2 = torch.nn.Linear(10, 10);

            var pgs = new Adadelta.ParamGroup[] {
                new () { Parameters = lin1.parameters(), Options = new() { LearningRate = 0.00005 } },
                new (lin2.parameters(), lr: 0.00003, maximize: false, rho: 0.25)
            };

            double learning_rate = 0.00004f;
            var optimizer = torch.optim.Adadelta(pgs, learning_rate);

            optimizer.load_py("pickled_optimizers/adadelta_load.pth");

            var sd = optimizer.state_dict();
            Assert.Multiple(() => {
                Assert.That(sd.Options, Has.Count.EqualTo(2));
                Assert.That(sd.State, Has.Count.EqualTo(4));
            });

            var options = (sd.Options[0] as Modules.Adadelta.Options)!;
            Assert.Multiple(() => {
                Assert.That(options.rho, Is.EqualTo(0.85));
                Assert.That(options.weight_decay, Is.EqualTo(0.3));
                Assert.That(options.LearningRate, Is.EqualTo(0.001));
                Assert.That(options.maximize, Is.False);
            });

            options = (sd.Options[1] as Modules.Adadelta.Options)!;
            Assert.Multiple(() => {
                Assert.That(options.rho, Is.EqualTo(0.79));
                Assert.That(options.weight_decay, Is.EqualTo(0.3));
                Assert.That(options.LearningRate, Is.EqualTo(0.01));
                Assert.That(options.maximize, Is.True);
            });


            foreach (var st in sd.State) {
                var state = (st as Modules.Adadelta.State)!;
                Assert.Multiple(() => {
                    Assert.That(state.step, Is.EqualTo(1));
                    Assert.That(state.square_avg, Is.Not.Null);
                    Assert.That(state.acc_delta, Is.Not.Null);
                });
            }
        }

        [Test]
        public void TestLoadAdagrad() {
            var lin = torch.nn.Linear(10, 10);

            double learning_rate = 0.00004f;
            var optimizer = torch.optim.Adagrad(lin.parameters(), learning_rate);

            optimizer.load_py("pickled_optimizers/adagrad_load.pth");

            var sd = optimizer.state_dict();
            Assert.Multiple(() => {
                Assert.That(sd.Options, Has.One.Items);
                Assert.That(sd.State, Has.Count.EqualTo(2));
            });

            foreach (var opts in sd.Options) {
                var options = (opts as Modules.Adagrad.Options)!;
                Assert.Multiple(() => {
                    Assert.That(options.lr_decay, Is.EqualTo(0.85));
                    Assert.That(options.weight_decay, Is.EqualTo(0.3));
                    Assert.That(options.LearningRate, Is.Not.EqualTo(learning_rate));
                });
            }

            foreach (var st in sd.State) {
                var state = (st as Modules.Adagrad.State)!;
                Assert.Multiple(() => {
                    Assert.That(state.step, Is.EqualTo(1));
                    Assert.That(state.sum, Is.Not.Null);
                });
            }
        }

        [Test]
        public void TestLoadAdamax() {
            var lin1 = torch.nn.Linear(10, 10);
            var lin2 = torch.nn.Linear(10, 10);

            var pgs = new Adamax.ParamGroup[] {
                new () { Parameters = lin1.parameters(), Options = new() { LearningRate = 0.00005 } },
                new (lin2.parameters(), lr: 0.00003, weight_decay: 0.25, beta1: 0.25)
            };

            double learning_rate = 0.00004f;
            var optimizer = torch.optim.Adamax(pgs, learning_rate);

            optimizer.load_py("pickled_optimizers/adamax_load.pth");

            var sd = optimizer.state_dict();
            Assert.Multiple(() => {
                Assert.That(sd.Options, Has.Count.EqualTo(2));
                Assert.That(sd.State, Has.Count.EqualTo(4));
            });

            var options = (sd.Options[0] as Modules.Adamax.Options)!;
            Assert.Multiple(() => {
                Assert.That(options.beta1, Is.EqualTo(0.8));
                Assert.That(options.beta2, Is.EqualTo(0.9));
                Assert.That(options.LearningRate, Is.EqualTo(0.001));
                Assert.That(options.weight_decay, Is.EqualTo(0));
            });

            options = (sd.Options[1] as Modules.Adamax.Options)!;
            Assert.Multiple(() => {
                Assert.That(options.beta1, Is.EqualTo(0.7));
                Assert.That(options.beta2, Is.EqualTo(0.79));
                Assert.That(options.LearningRate, Is.EqualTo(0.01));
                Assert.That(options.weight_decay, Is.EqualTo(0.3));
            });

            foreach (var state in sd.State.Cast<Adamax.State>()) {
                Assert.Multiple(() => {
                    Assert.That(state.step, Is.EqualTo(1));
                    Assert.That(state.exp_avg, Is.Not.Null);
                    Assert.That(state.exp_inf, Is.Not.Null);
                });
            }
        }
        [Test]
        public void TestLoadAdamEmptyState() {
            var lin1 = torch.nn.Linear(10, 10);
            var lin2 = torch.nn.Linear(10, 10);
            var lin3 = torch.nn.Linear(10, 10);

            var pgs = new Adam.ParamGroup[] {
                new () { Parameters = lin1.parameters(), Options = new() { LearningRate = 0.00005 } },
                new (lin2.parameters(), lr: 0.00003, amsgrad: false, beta1: 0.25),
                new (lin3.parameters(), lr: 0.00003, amsgrad: false, beta1: 0.9, beta2: 9.99)
            };

            double learning_rate = 0.00004f;
            var optimizer = torch.optim.Adam(pgs, learning_rate);

            // Calculate loss for lin3, so that it steps through
            torch.nn.functional.mse_loss(lin3.call(torch.rand(10)), torch.rand(10)).backward();
            optimizer.step();

            optimizer.load_py("pickled_optimizers/adam_emptystate_load.pth");

            var sd = optimizer.state_dict();
            Assert.Multiple(() => {
                Assert.That(sd.Options, Has.Count.EqualTo(3));
                Assert.That(sd.State, Has.Count.EqualTo(6));
            });

            var options = (sd.Options[0] as Modules.Adam.Options)!;
            Assert.Multiple(() => {
                Assert.That(options.beta1, Is.EqualTo(0.8));
                Assert.That(options.beta2, Is.EqualTo(0.9));
                Assert.That(options.LearningRate, Is.EqualTo(0.001));
                Assert.That(options.amsgrad, Is.False);
            });

            options = (sd.Options[1] as Modules.Adam.Options)!;
            Assert.Multiple(() => {
                Assert.That(options.beta1, Is.EqualTo(0.7));
                Assert.That(options.beta2, Is.EqualTo(0.79));
                Assert.That(options.LearningRate, Is.EqualTo(0.01));
                Assert.That(options!.amsgrad, Is.True);
            });

            options = (sd.Options[2] as Modules.Adam.Options)!;
            Assert.Multiple(() => {
                Assert.That(options.beta1, Is.EqualTo(0.6));
                Assert.That(options.beta2, Is.EqualTo(0.69));
                Assert.That(options.LearningRate, Is.EqualTo(0.01));
                Assert.That(options!.amsgrad, Is.True);
            });

            var state = (sd.State[0] as Modules.Adam.State)!;
            Assert.Multiple(() => {
                Assert.That(state.step, Is.EqualTo(1));
                Assert.That(state.exp_avg, Is.Not.Null);
                Assert.That(state.exp_avg_sq, Is.Not.Null);
                Assert.That(state.max_exp_avg_sq, Is.Null);
            });

            state = (sd.State[1] as Modules.Adam.State)!;
            Assert.Multiple(() => {
                Assert.That(state.step, Is.EqualTo(1));
                Assert.That(state.exp_avg, Is.Not.Null);
                Assert.That(state.exp_avg_sq, Is.Not.Null);
                Assert.That(state.max_exp_avg_sq, Is.Null);
            });

            state = (sd.State[2] as Modules.Adam.State)!;
            Assert.Multiple(() => {
                Assert.That(state.step, Is.EqualTo(1));
                Assert.That(state.exp_avg, Is.Not.Null);
                Assert.That(state.exp_avg_sq, Is.Not.Null);
                Assert.That(state.max_exp_avg_sq, Is.Not.Null);
            });

            state = (sd.State[3] as Modules.Adam.State)!;
            Assert.Multiple(() => {
                Assert.That(state.step, Is.EqualTo(1));
                Assert.That(state.exp_avg, Is.Not.Null);
                Assert.That(state.exp_avg_sq, Is.Not.Null);
                Assert.That(state.max_exp_avg_sq, Is.Not.Null);
            });

            // Make sure lin3 was reset to the defaults
            state = (sd.State[4] as Modules.Adam.State)!;
            Assert.Multiple(() => {
                Assert.That(state.step, Is.EqualTo(0));
                Assert.That(torch.count_nonzero(state.exp_avg).ToInt32(), Is.EqualTo(0));
                Assert.That(torch.count_nonzero(state.exp_avg_sq).ToInt32(), Is.EqualTo(0));
                Assert.That(state.max_exp_avg_sq, Is.Not.Null);
            });

            state = (sd.State[5] as Modules.Adam.State)!;
            Assert.Multiple(() => {
                Assert.That(state.step, Is.EqualTo(0));
                Assert.That(torch.count_nonzero(state.exp_avg).ToInt32(), Is.EqualTo(0));
                Assert.That(torch.count_nonzero(state.exp_avg_sq).ToInt32(), Is.EqualTo(0));
                Assert.That(state.max_exp_avg_sq, Is.Not.Null);
            });
        }
    }
}
