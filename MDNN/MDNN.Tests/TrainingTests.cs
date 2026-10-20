using System;
  using Xunit;
  using MDNN;
  using MDNN.Layers;
  using MDNN.Activations;
  using MDNN.Optimizers;
  using MDNN.Losses;
  using System.Linq;

  namespace MDNN.Tests;

  public class SmokeTrainTest:GradientCheckTestBase
  {
      [Fact]
      public void Smoke_linear_regression_learns()
      {
          // y = 2x + 1
          double[][] X = { new[]{-1.0}, new[]{0.0}, new[]{1.0}, new[]{2.0} };
          double[][] Y = { new[]{-1.0}, new[]{1.0}, new[]{3.0}, new[]{5.0} };

          var model = new NeuralNetwork(new Dense(1, new Linear()), new Adam(0.05), new MSE());

          for (int epoch = 0; epoch < 500; epoch++)
          for (int n = 0; n < X.Length; n++)
          {
              model.Train.Fit(new Tensor(X[n]), new Tensor(Y[n]));
              model.Train.UpdateParams();
          }

          var loss = model.Loss;
          double mse = 0;
          for (int n = 0; n < X.Length; n++)
              mse += loss.CalculateAndGetLoss(model.GetResults(new Tensor(X[n])).Data, Y[n]);
          Assert.True(mse / X.Length < 1e-3, $"MSE = {mse / X.Length}");
      }

      [Fact]
      public void Smoke_XOR_learns()
      {
          double[][] X = { new[]{0.0,0.0}, new[]{0.0,1.0}, new[]{1.0,0.0}, new[]{1.0,1.0} };
          double[][] Y = { new[]{0.0},     new[]{1.0},     new[]{1.0},     new[]{0.0} };

          var model = new NeuralNetwork(new Dense(1, new Sigmoid()), new Adam(0.05), new MSE());
          model.Layers.Add(new Dense(8, new Tanh()));  

          for (int epoch = 0; epoch < 3000; epoch++)
          for (int n = 0; n < 4; n++)
          {
              model.Train.Fit(new Tensor(X[n]), new Tensor(Y[n]));
              model.Train.UpdateParams();
          }

          // more robust than a strict loss threshold: the rounded outputs must match
          for (int n = 0; n < 4; n++)
              Assert.Equal(Y[n][0], Math.Round(model.GetResults(new Tensor(X[n])).Data[0]));
      }
      
      [Fact]
      public void SimpleTrainLoop_small_epochs_does_not_throw()
      {
          double[][] X = { new[]{0.0}, new[]{1.0} };
          double[][] Y = { new[]{0.0}, new[]{1.0} };

          var model = new NeuralNetwork(new Dense(1, new Linear()), new SGD(0.01), new MSE());
          model.Train.AutoSave = false;      // don't write a .json to disk

          model.Train.SimpleTrainLoop(X, Y, 10);        // 10 < 100 → threw DivideByZeroException before the fix
      }
      
      [Fact]
      public void Dense_init_symmetric_and_fanin_scaled()
      {
          int fanIn = 1000;                                   // large sample
          double[] w = new Neuron(fanIn, new ReLU()).Weights;

          Assert.Contains(w, x => x < 0);                     // symmetry: there MUST be negatives
          Assert.Contains(w, x => x > 0);                     //           and positives

          double var = w.Select(x => x * x).Average();        // mean≈0 → var ≈ E[x²]
          double heVar = 2.0 / fanIn;
          Assert.InRange(var, heVar * 0.5, heVar * 1.5);      // He scale roughly matches
      }
  }

  public class OptimizerTest:GradientCheckTestBase
  {
      [Fact]
      public void SGD_update()
          => Assert.Equal(0.8, new SGD(0.1).Update(1.0, 2.0,0), 12);   // 1 - 0.1·2
      
      [Fact]
      public void Momentum_accumulates_velocity()
      {
          var m = new Momentum(0.1, 0.9);      // L=0.1, γ=0.9
          double w = m.Update(1.0, 2.0,0);       // v=0.2 → w=0.8
          Assert.Equal(0.8, w, 12);
          w = m.Update(w, 2.0,0);                // v=0.9·0.2+0.1·2=0.38 → w=0.42
          Assert.Equal(0.42, w, 12);           // verifies that momentum accumulates
      }
      
      [Fact]
      public void Adam_first_step_is_bias_corrected()
      {
          var adam = new Adam(0.05);
          double w = adam.Update(1.0, 2.0, 0);
          // with bias correction the first step ≈ L·sign(g) regardless of |g| → 1 - 0.05 = 0.95
          Assert.Equal(0.95, w, 6);
      }

      [Fact]
      public void Adam_bias_correction_follows_each_parameters_own_step()
      {
          // The 1 − β^t correction is cached for the last t (during training all weights of
          // a neuron share the same t). Here t alternates — parameter 0 is at step 3,
          // parameter 1 at step 1 — and the cache must not mix them up: parameter 1 must get
          // the FIRST-step correction.
          //
          // Expected values are computed independently from the formula (not by a second
          // Adam instance — it would have the same bug), and the gradient varies: with a
          // constant gradient the corrected step is always the same and a wrong correction
          // would go unnoticed.
          const double lr = 0.05, b1 = 0.9, b2 = 0.999;
          double Expected(double[] grads)
          {
              double m = 0, v = 0;
              foreach (double g in grads) { m = b1 * m + (1 - b1) * g; v = b2 * v + (1 - b2) * g * g; }
              int t = grads.Length;
              return 1.0 - lr * (m / (1 - Math.Pow(b1, t))) / (Math.Sqrt(v / (1 - Math.Pow(b2, t))) + 1e-8);
          }

          var adam = new Adam(lr);
          adam.Update(1.0, 1.0, 0);
          adam.Update(1.0, 3.0, 0);
          adam.Update(1.0, -2.0, 0);

          double first = adam.Update(1.0, 5.0, 1);    // parameter 1, step 1
          double fourth = adam.Update(1.0, 4.0, 0);   // parameter 0, step 4

          Assert.Equal(Expected(new[] { 5.0 }), first, 12);
          Assert.Equal(Expected(new[] { 1.0, 3.0, -2.0, 4.0 }), fourth, 12);
      }
  }

  public class IndependenceTests : GradientCheckTestBase
  {
      [Fact]
      public void Two_models_built_interleaved_are_independent()
      {
          // Build TWO models interleaved, with different architecture and loss.
          // (Previously a shared static layersList made B infer its sizes from A; now it is per instance.)
          var a = new NeuralNetwork(new Dense(1, new Linear()), new SGD(0.01), new MSE());
          var b = new NeuralNetwork(new Dense(3, new Softmax()), new Adam(0.01), new CrossEntropy());
          a.Layers.Add(new Dense(4, new Tanh()));   // A: 4 hidden neurons, SGD
          b.Layers.Add(new Dense(6, new Tanh()));   // B: 6 hidden neurons, Adam

          Assert.IsType<MSE>(a.Loss);               // configurations are independent
          Assert.IsType<CrossEntropy>(b.Loss);

          a.Train.Fit(new Tensor(new double[] { 0.5 }), new Tensor(new double[] { 1.0 }));
          b.Train.Fit(new Tensor(new double[] { 0.5, 0.3, 0.2 }), new Tensor(new double[] { 0.0, 1.0, 0.0 }));

          // per-model optimizer: A has SGD, B has Adam (previously a shared static → they would clash)
          Assert.IsType<SGD>(((Dense)a.Layers.Layers[0]).Neurons[0].Optimizer);
          Assert.IsType<Adam>(((Dense)b.Layers.Layers[0]).Neurons[0].Optimizer);

          // after training both: each keeps its own architecture and loss (no interference)
          Assert.Single(a.GetResults(new Tensor(new double[] { 0.5 })).Data);
          Assert.Equal(3, b.GetResults(new Tensor(new double[] { 0.5, 0.3, 0.2 })).Data.Length);
          Assert.IsType<MSE>(a.Loss);
          Assert.IsType<CrossEntropy>(b.Loss);
      }
  }

  public class EmptyMiniBatchTests : GradientCheckTestBase
  {
      // Neuron.UpdateWeightsAndBias divided by mini_batch_size, which was 0 → 0/0 = NaN,
      // silently spreading through the whole model.
      [Fact]
      public void UpdateParams_without_backprop_does_not_produce_NaN()
      {
          var model = new NeuralNetwork(new Dense(2, new Linear()), new SGD(0.01), new MSE());
          model.Layers.SetInputShape(new int[] { 2 });

          var x = new Tensor(new double[] { 1, 2 });
          model.GetResults(x);
          model.Train.UpdateParams();                             // no backprop before this

          Assert.All(model.GetResults(x).Data, v => Assert.False(double.IsNaN(v), "output is NaN"));
      }
  }

  public class RepeatedTrainLoopTests : GradientCheckTestBase
  {
      // ConsoleController held a static _time / _lastEpochInfo. On the second TrainLoop
      // pastEpochs came out as 0 → (subTime * n) / 0 → OverflowException.
      // Running several models in one process must work (e.g. an AutoML runner).
      [Fact]
      public void Two_train_loops_in_one_process_do_not_throw()
      {
          for (int run = 1; run <= 2; run++)
          {
              var model = new NeuralNetwork(new Dense(1, new Linear()), new SGD(0.05), new MSE());
              model.Train.ShowLossChart = false;
              model.Train.ShowModelInfo = false;
              model.Train.AutoSave = false;
              model.Train.TestAfterTraining = false;
              model.Train.ConsoleReportCount = 1;        // one report per run

              double[][] X = new double[20][];
              double[][] Y = new double[20][];
              for (int i = 0; i < 20; i++)
              {
                  X[i] = new double[] { i * 0.05 };
                  Y[i] = new double[] { i * 0.10 };
              }

              var ex = Record.Exception(() => model.Train.TrainLoop(X, Y, 50, 4));
              Assert.True(ex == null, $"run {run} failed:{ex?.GetType().Name}: {ex?.Message}");
          }
      }

      // The clamp in Train.PreparationForTrainLoop used to write into the PUBLIC field,
      // permanently overwriting the user's setting.
      [Fact]
      public void NumberOfShowEpochInConsole_survives_training()
      {
          var model = new NeuralNetwork(new Dense(1, new Linear()), new SGD(0.05), new MSE());
          model.Train.ShowLossChart = false;
          model.Train.ShowModelInfo = false;
          model.Train.AutoSave = false;
          model.Train.TestAfterTraining = false;

          uint before = model.Train.ConsoleReportCount;   // default 100

          double[][] X = new double[20][];
          double[][] Y = new double[20][];
          for (int i = 0; i < 20; i++)
          {
              X[i] = new double[] { i * 0.05 };
              Y[i] = new double[] { i * 0.10 };
          }
          model.Train.TrainLoop(X, Y, 5);                         // 5 epochs < 100 → clamp

          Assert.Equal(before, model.Train.ConsoleReportCount);
      }
  }

  // ==========================================================================================
  //  Divergence and early stopping
  // ==========================================================================================

  public class DivergenceTests : GradientCheckTestBase
  {
      // On a NaN loss the library used to call Environment.Exit(0) — killing the host
      // process, with exit code 0 = "success" no less. The caller had no chance to react.
      // (This test could not even be written against the old version: Exit would kill the test runner.)
      private static (double[][] X, double[][] Y) Data()
      {
          var X = new double[20][];
          var Y = new double[20][];
          for (int i = 0; i < 20; i++)
          {
              X[i] = new double[] { i * 0.5 + 1 };
              Y[i] = new double[] { i * 1.5 + 1 };
          }
          return (X, Y);
      }

      private static NeuralNetwork DivergingModel()
      {
          // absurd learning rate → weights shoot to ±∞ and then NaN
          var model = new NeuralNetwork(new Dense(1, new Linear()), new SGD(1e12), new MSE());
          model.Train.ShowLossChart = false;
          model.Train.ShowModelInfo = false;
          model.Train.AutoSave = false;
          model.Train.TestAfterTraining = false;
          return model;
      }

      [Fact]
      public void TrainLoop_on_divergence_throws_instead_of_killing_process()
      {
          var (X, Y) = Data();
          var model = DivergingModel();

          var ex = Assert.Throws<TrainingDivergedException>(() => model.Train.TrainLoop(X, Y, 50, 4));

          Assert.False(double.IsFinite(ex.Loss), "the exception should carry the invalid loss");
          Assert.Contains("learning rate", ex.Message);   // the message should suggest a fix
      }

      [Fact]
      public void SimpleTrainLoop_on_divergence_throws_too()
      {
          // previously this was just Console.WriteLine + return → the caller could not tell
          // "training finished" from "it blew up in the first epoch"
          var (X, Y) = Data();
          var model = DivergingModel();

          Assert.Throws<TrainingDivergedException>(() => model.Train.SimpleTrainLoop(X, Y, 50, 4));
      }

      [Fact]
      public void Divergence_is_catchable_and_process_survives()
      {
          // exactly the AutoML-runner scenario: a candidate diverged, catch it and move on
          var (X, Y) = Data();
          bool caught = false;

          try { DivergingModel().Train.TrainLoop(X, Y, 50, 4); }
          catch (TrainingDivergedException) { caught = true; }

          Assert.True(caught);

          // after catching it, another model must train normally
          var healthy = new NeuralNetwork(new Dense(1, new Linear()), new SGD(0.01), new MSE());
          healthy.Train.ShowLossChart = false;
          healthy.Train.ShowModelInfo = false;
          healthy.Train.AutoSave = false;
          healthy.Train.TestAfterTraining = false;
          healthy.Train.TrainLoop(X, Y, 10, 2);

          Assert.True(double.IsFinite(healthy.GetResults(new Tensor(new double[] { 1.0 })).Data[0]));
      }
  }

  public class EarlyStoppingTests : GradientCheckTestBase
  {
      [Fact]
      public void EarlyStopping_stops_before_total_epochs()
      {
          // y = x regression; minDelta is set absurdly high → after the first report (an
          // improvement over MaxValue) NOTHING counts as an improvement → stops after `patience` reports.
          // Without early stopping the loop would run to totalEpoch (100).
          var model = new NeuralNetwork(new Dense(1, new Linear()), new SGD(0.01), new MSE());

          double[][] X = new double[20][];
          double[][] Y = new double[20][];
          for (int i = 0; i < 20; i++)
          {
              X[i] = new double[] { i * 0.1 };
              Y[i] = new double[] { i * 0.1 };
          }

          model.Train.ShowLossChart = false;      // no chart popping up in tests
          model.Train.ShowModelInfo = false;
          model.Train.AutoSave = false;           // no writing to disk
          model.Train.TestAfterTraining = false;
          model.Train.ConsoleReportCount = 100;      // report every epoch (totalEpoch=100)

          model.Train.EarlyStoppingEnabled = true;
          model.Train.EarlyStoppingPatience = 3;
          model.Train.EarlyStoppingMinDelta = 1e9;           // nothing will count as an improvement

          model.Train.TrainLoop(X, Y, 100, 1);

          // 1 report with an improvement (over MaxValue) + 3 without → stop well below 100 epochs
          Assert.True(model.Train.CurrentEpoch < 20,
              $"CurrentEpoch = {model.Train.CurrentEpoch}, early stopping should have stopped early");
      }

      // Restoring the best model at the end of TrainLoop must not depend on the final test.
      // It used to be inside `if (TestAfterTraining)`, so with the test disabled the model
      // kept the last epoch. Noisy data + a large learning rate → valid loss starts rising
      // after a few epochs and early stopping ends on a worse model than the best one.
      [Theory]
      [InlineData(true)]
      [InlineData(false)]
      public void TrainLoop_leaves_the_best_model_regardless_of_final_test(bool testAfterTraining)
      {
          var rnd = new Random(1);
          double[][] X = Enumerable.Range(0, 60).Select(_ => new[] { rnd.NextDouble(), rnd.NextDouble() }).ToArray();
          double[][] Y = X.Select(v => new[] { v[0] + v[1] + rnd.NextDouble() }).ToArray();

          var model = new NeuralNetwork(new Dense(1, new Linear()), new Adam(0.05), new MSE(), seed: 7);
          model.Layers.Add(new Dense(32, new ReLU()));
          model.Train.ShowModelInfo = false;
          model.Train.AutoSave = false;
          model.Train.TestAfterTraining = testAfterTraining;
          model.Train.EarlyStoppingEnabled = true;
          model.Train.EarlyStoppingPatience = 3;
          model.Train.ConsoleReportCount = 1000;   // report every epoch

          model.Train.TrainLoop(X, Y, 300, 4);

          Tensor validX = model.Train.ValidInputs!;
          Tensor validY = model.Train.ValidTargets!;
          double sum = 0;
          for (int i = 0; i < validX.Shape[0]; i++)
          {
              sum += new MSE().CalculateAndGetLoss(
                  model.GetResults(validX.GetSubTensor([i])).Data, validY.GetSubTensor([i]).Data);
          }

          // test precondition: the last epoch really is NOT the best one
          Assert.NotEqual(model.Train.BestEpoch, model.Train.CurrentEpoch);
          Assert.Equal(model.Train.LowestValidLoss, sum / validX.Shape[0], 12);
      }
  }

  // ==========================================================================================
  //  Conventional epochs (a full pass in batches instead of random draws with replacement)
  // ==========================================================================================

  public class ConventionalEpochTests : GradientCheckTestBase
  {
      private static (double[][] X, double[][] Y) Data(int n)
      {
          var X = new double[n][];
          var Y = new double[n][];
          for (int i = 0; i < n; i++)
          {
              X[i] = new double[] { i * 0.07 - 0.5 };
              Y[i] = new double[] { i * 0.11 - 0.3 };
          }
          return (X, Y);
      }

      private static NeuralNetwork Quiet(int? seed = null)
      {
          var m = new NeuralNetwork(new Dense(1, new Linear()), new SGD(0.01), new MSE(), seed);
          m.Train.ShowLossChart = false;
          m.Train.ShowModelInfo = false;
          m.Train.AutoSave = false;
          m.Train.TestAfterTraining = false;
          return m;
      }

      [Theory]
      [InlineData(1)]     // 14 samples / batch 1 → 14 steps per epoch
      [InlineData(4)]     // 14 / 4 = 3.5         → 4 steps (last batch partial)
      [InlineData(7)]     // 14 / 7               → 2 steps
      [InlineData(14)]    // whole set in one batch → 1 step
      [InlineData(50)]    // batch larger than set  → still 1 step
      public void One_epoch_runs_one_optimizer_step_per_batch(int batchSize)
      {
          var (X, Y) = Data(20);
          var model = Quiet();

          model.Train.TrainLoop(X, Y, 1, (uint)batchSize);

          int trainCount = model.Train.TrainInputs!.Shape[0];      // 0.7 of 20 = 14
          ulong expected = (ulong)((trainCount + batchSize - 1) / batchSize);

          Assert.Equal(14, trainCount);
          Assert.Equal(expected, model.Train.OptimizerSteps);
      }

      [Fact]
      public void Epoch_counter_counts_epochs_not_optimizer_steps()
      {
          // UpdateParams() used to increment the epoch counter, and it came out the same because
          // each "epoch" was a single step. Now the two must differ.
          var (X, Y) = Data(20);
          var model = Quiet();

          model.Train.TrainLoop(X, Y, 3, 1);

          Assert.Equal(3u, model.Train.CurrentEpoch);          // 3 passes over the data
          Assert.Equal(42ul, model.Train.OptimizerSteps);      // 3 × 14 batches
      }

      [Fact]
      public void Manual_training_counts_steps_but_no_epochs()
      {
          // Manual Fit + UpdateParams has no epochs — reporting them as epochs would be a lie.
          var model = Quiet();
          model.Layers.SetInputShape(new int[] { 1 });

          for (int i = 0; i < 5; i++)
          {
              model.Train.Fit(new Tensor(new double[] { 1 }), new Tensor(new double[] { 0.5 }));
              model.Train.UpdateParams();
          }

          Assert.Equal(0u, model.Train.CurrentEpoch);
          Assert.Equal(5ul, model.Train.OptimizerSteps);
      }

      // The core property: in one epoch the model must see EVERY training sample EXACTLY ONCE.
      // Samples used to be drawn with replacement, so some were never seen and others several times.
      //
      // Proof without computing gradients by hand: one epoch with batch = whole train set
      // (i.e. one optimizer step) must equal a manual Fit over all samples exactly once
      // + one UpdateParams. Gradients are SUMMED within a batch, so the order inside the
      // batch does not matter. If the loop skipped or duplicated a sample, the sum would
      // not match.
      [Fact]
      public void One_epoch_equals_seeing_every_sample_exactly_once()
      {
          var (X, Y) = Data(20);

          // reference model: 0 epochs → only splits the dataset, weights stay at their initial values
          var reference = Quiet(seed: 5);
          reference.Train.TrainLoop(X, Y, 0);

          Tensor trainX = reference.Train.TrainInputs!;
          Tensor trainY = reference.Train.TrainTargets!;
          int n = trainX.Shape[0];

          // model under test: one epoch, the whole train set in one batch
          var underTest = Quiet(seed: 5);
          underTest.Train.TrainLoop(X, Y, 1, (uint)n);

          // reference: every sample exactly once by hand, then one step
          for (int i = 0; i < n; i++)
          {
              reference.Train.Fit(trainX.GetSubTensor([i]), trainY.GetSubTensor([i]));
          }
          reference.Train.UpdateParams();

          double[] expected = ((Dense)reference.Layers.Layers[0]).Neurons[0].Weights;
          double[] actual = ((Dense)underTest.Layers.Layers[0]).Neurons[0].Weights;

          // not bit-identical: floating-point addition is not associative and the batch is
          // traversed in shuffled order, so the last bits may differ
          Assert.Equal(expected.Length, actual.Length);
          for (int i = 0; i < expected.Length; i++)
          {
              Assert.True(Math.Abs(expected[i] - actual[i]) < 1e-12,
                  $"weight {i}: expected {expected[i]}, got {actual[i]}");
          }
      }

      [Fact]
      public void Zero_batch_size_is_rejected()
      {
          // zero used to silently skip training; with batched epochs "start += 0" would loop forever
          var (X, Y) = Data(20);

          Assert.Throws<ArgumentException>(() => Quiet().Train.TrainLoop(X, Y, 5, 0));
      }

      [Fact]
      public void Epochs_are_reproducible_with_a_seed()
      {
          // per-epoch shuffling draws from Context.Random → with a seed the result must be identical
          var (X, Y) = Data(20);
          var probe = new Tensor(new double[] { 0.25 });

          double Run()
          {
              var m = Quiet(seed: 11);
              m.Train.TrainLoop(X, Y, 5, 3);
              return m.GetResults(probe).Data[0];
          }

          Assert.Equal(Run(), Run());
      }
  }

  // ==========================================================================================
  //  Bugs found while going through the examples (2026-08-11) — both were old bugs, not regressions
  // ==========================================================================================

  public class SmallDatasetTests : GradientCheckTestBase
  {
      // The split computed (int)(n * 0.15), which is ZERO for n < 7 → Slice(x, 0) threw
      // "Invalid slice range!". So any dataset smaller than 7 samples crashed TrainLoop.
      // It surfaced while verifying the README example, which has 4 samples.
      [Theory]
      [InlineData(3)]
      [InlineData(4)]
      [InlineData(5)]
      [InlineData(6)]
      [InlineData(7)]
      [InlineData(20)]
      public void Tiny_dataset_still_splits_into_three_nonempty_sets(int sampleCount)
      {
          var model = new NeuralNetwork(new Dense(1, new Linear()), new SGD(0.01), new MSE());

          var xs = new double[sampleCount, 1];
          var ys = new double[sampleCount, 1];
          for (int i = 0; i < sampleCount; i++) { xs[i, 0] = i; ys[i, 0] = i * 2; }

          model.Train.SplitDatasets(new Tensor(xs), new Tensor(ys));

          Assert.True(model.Train.TrainInputs!.Shape[0] >= 1, "train is empty");
          Assert.True(model.Train.ValidInputs!.Shape[0] >= 1, "valid is empty");
          Assert.True(model.Train.TestInputs!.Shape[0] >= 1, "test is empty");

          Assert.Equal(sampleCount,
              model.Train.TrainInputs.Shape[0]
              + model.Train.ValidInputs.Shape[0]
              + model.Train.TestInputs.Shape[0]);
      }

      [Fact]
      public void Two_samples_are_rejected_with_a_clear_message()
      {
          var model = new NeuralNetwork(new Dense(1, new Linear()), new SGD(0.01), new MSE());
          var xs = new double[2, 1];
          var ys = new double[2, 1];

          var ex = Assert.Throws<ArgumentException>(
              () => model.Train.SplitDatasets(new Tensor(xs), new Tensor(ys)));
          Assert.Contains("SetDatasets", ex.Message);
      }

      [Fact]
      public void Readme_quick_start_with_four_samples_trains()
      {
          // exactly the README example (4 samples) — used to fail with "Invalid slice range!"
          double[][] X = { new double[] { 0 }, new double[] { 1 }, new double[] { 2 }, new double[] { 3 } };
          double[][] Y = { new double[] { 0 }, new double[] { 2 }, new double[] { 4 }, new double[] { 6 } };

          var model = new NeuralNetwork(new Dense(1, new Linear()), new SGD(0.01), new MSE());
          model.Layers.Add(new Dense(8, new ReLU()));
          model.Train.ShowLossChart = false;
          model.Train.ShowModelInfo = false;
          model.Train.AutoSave = false;
          model.Train.TestAfterTraining = false;

          model.Train.TrainLoop(X, Y, 5, 1);

          Assert.Equal(5u, model.Train.CurrentEpoch);
      }
  }

  public class TargetSizeMismatchTests : GradientCheckTestBase
  {
      // When the target had fewer values than the output layer has neurons, an
      // IndexOutOfRangeException was thrown from deep inside the gradient code — the cause was
      // impossible to tell from it. Also found in the examples: the README has `new Dense(3)`
      // and a reader can easily pass a scalar target.
      [Fact]
      public void Mismatched_target_size_gives_a_clear_error()
      {
          var model = new NeuralNetwork(new Dense(3, new Linear()), new SGD(0.01), new MSE());
          model.Layers.SetInputShape(new int[] { 1 });

          var ex = Assert.Throws<ArgumentException>(() =>
              model.Train.Fit(new Tensor(new double[] { 1 }), new Tensor(new double[] { 0 })));

          Assert.Contains("3", ex.Message);
          Assert.Contains("1", ex.Message);
      }
  }

  public class SequenceScoreTests : GradientCheckTestBase
  {
      // In the sequence branch maxScore grew by the number of output NEURONS while score
      // grew by 1 per timestep → the score/maxScore ratio was meaningless. The non-sequence
      // branch had it right.
      [Fact]
      public void Sequential_test_counts_timesteps_not_output_neurons()
      {
          var model = new NeuralNetwork(new Dense(2, new Linear()), new SGD(0.01), new MSE());
          model.Context.SequenceTrain = true;

          // 2 sequences × 3 steps × 2 features, 2 output neurons
          var inputs  = new Tensor(new double[2 * 3 * 2], new int[] { 2, 3, 2 });
          var targets = new Tensor(new double[2 * 3 * 2], new int[] { 2, 3, 2 });

          (int score, int maxScore) = model.Train.Evaluate(inputs, targets, showInConsole: false);

          Assert.Equal(6, maxScore);          // 2 sequences × 3 steps; previously 12 (× 2 neurons)
          Assert.InRange(score, 0, maxScore); // score must not exceed the maximum
      }
  }
