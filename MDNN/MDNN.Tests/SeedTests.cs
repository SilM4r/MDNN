using System;
  using Xunit;
  using MDNN;
  using MDNN.Layers;
  using MDNN.Activations;
  using MDNN.Optimizers;
  using MDNN.Losses;
  using System.Linq;

  namespace MDNN.Tests;

  // ==========================================================================================
  //  Seed / reproducibility
  // ==========================================================================================

  public class SeedReproducibilityTests : GradientCheckTestBase
  {
      // There used to be THREE independent sources of randomness and none could be set:
      // GeneralNeuralNetworkSettings.Random (weights), Train._rnd (sample selection),
      // Random.Shared (ShuffleTensor). The same experiment could not be run twice, which is
      // a prerequisite for fairly comparing two candidates (e.g. in an AutoML runner).
      private static (double[][] X, double[][] Y) Data(int n)
      {
          var X = new double[n][];
          var Y = new double[n][];
          for (int i = 0; i < n; i++)
          {
              X[i] = new double[] { i * 0.05, 1 - i * 0.03 };
              Y[i] = new double[] { i * 0.10 };
          }
          return (X, Y);
      }

      private static NeuralNetwork Build(int? seed)
      {
          var model = new NeuralNetwork(new Dense(1, new Linear()), new SGD(0.01), new MSE(), seed);
          model.Layers.Add(new Dense(4, new ReLU()));
          model.Train.ShowLossChart = false;
          model.Train.ShowModelInfo = false;
          model.Train.AutoSave = false;
          model.Train.TestAfterTraining = false;
          return model;
      }

      [Fact]
      public void Same_seed_gives_identical_initial_weights()
      {
          var a = Build(42);
          var b = Build(42);

          a.GetResults(new Tensor(new double[] { 1, 1 }));   // triggers wiring = weight initialization
          b.GetResults(new Tensor(new double[] { 1, 1 }));

          var na = ((Dense)a.Layers.Layers[0]).Neurons;
          var nb = ((Dense)b.Layers.Layers[0]).Neurons;

          for (int i = 0; i < na.Count; i++)
              Assert.Equal(na[i].Weights, nb[i].Weights);
      }

      [Fact]
      public void Different_seed_gives_different_initial_weights()
      {
          var a = Build(1);
          var b = Build(2);

          a.GetResults(new Tensor(new double[] { 1, 1 }));
          b.GetResults(new Tensor(new double[] { 1, 1 }));

          Assert.NotEqual(
              ((Dense)a.Layers.Layers[0]).Neurons[0].Weights,
              ((Dense)b.Layers.Layers[0]).Neurons[0].Weights);
      }

      [Fact]
      public void Same_seed_gives_identical_result_after_full_training()
      {
          // the strictest variant: the seed must also cover dataset shuffling and sample
          // selection, not just weight initialization
          var (X, Y) = Data(20);
          var probe = new Tensor(new double[] { 0.5, 0.5 });

          var a = Build(7);
          a.Train.TrainLoop(X, Y, 30, 4);
          double outA = a.GetResults(probe).Data[0];

          var b = Build(7);
          b.Train.TrainLoop(X, Y, 30, 4);
          double outB = b.GetResults(probe).Data[0];

          Assert.Equal(outA, outB);
      }

      [Fact]
      public void Same_seed_splits_dataset_identically()
      {
          var (X, Y) = Data(20);

          var a = Build(7);
          a.Train.TrainLoop(X, Y, 5);
          double[] validA = a.Train.ValidInputs!.Data;

          var b = Build(7);
          b.Train.TrainLoop(X, Y, 5);

          Assert.Equal(validA, b.Train.ValidInputs!.Data);   // same shuffle → same split
      }

      [Fact]
      public void Without_seed_behaviour_is_unchanged()
      {
          // backward compatibility: without a seed, values still come from the shared global
          // generator, so two models do NOT have the same weights
          var a = Build(null);
          var b = Build(null);

          a.GetResults(new Tensor(new double[] { 1, 1 }));
          b.GetResults(new Tensor(new double[] { 1, 1 }));

          Assert.NotEqual(
              ((Dense)a.Layers.Layers[0]).Neurons[0].Weights,
              ((Dense)b.Layers.Layers[0]).Neurons[0].Weights);
      }

      [Fact]
      public void Seeded_conv_kernels_are_reproducible()
      {
          NeuralNetwork WithConv(int seed)
          {
              var m = new NeuralNetwork(new Dense(2, new Linear()), new SGD(0.01), new MSE(), seed);
              m.Layers.Add(new Conv(2, 3, new ReLU(), "valid"));
              m.Layers.SetInputShape(new int[] { 6, 6, 1 });
              return m;
          }

          var a = (Conv)WithConv(99).Layers.Layers[0];
          var b = (Conv)WithConv(99).Layers.Layers[0];

          Assert.Equal(a.Kernels[0][0][0], b.Kernels[0][0][0]);
          Assert.Equal(a.Biases, b.Biases);
      }
  }

  public class SeedErgonomicsTests : GradientCheckTestBase
  {
      [Fact]
      public void Seed_can_be_read_back()
      {
          // a caller (e.g. an AutoML runner) needs to record what reproduces a candidate's run
          var model = new NeuralNetwork(new Dense(2, new Linear()), new SGD(0.01), new MSE(), seed: 1234);
          Assert.Equal(1234, model.Context.Seed);
      }

      [Fact]
      public void Seed_is_null_when_not_given()
      {
          var model = new NeuralNetwork(new Dense(2, new Linear()), new SGD(0.01), new MSE());
          Assert.Null(model.Context.Seed);
      }

      [Fact]
      public void Seed_can_be_set_via_context_before_layers_are_built()
      {
          NeuralNetwork Build()
          {
              var m = new NeuralNetwork(new Dense(2, new Linear()), new SGD(0.01), new MSE());
              m.Context.Seed = 42;                       // nothing drawn yet → OK
              m.Layers.Add(new Dense(3, new ReLU()));
              m.GetResults(new Tensor(new double[] { 1, 1 }));
              return m;
          }

          Assert.Equal(
              ((Dense)Build().Layers.Layers[0]).Neurons[0].Weights,
              ((Dense)Build().Layers.Layers[0]).Neurons[0].Weights);
      }

      [Fact]
      public void Seed_can_still_be_set_after_adding_layers()
      {
          // Materialization happens in a single pass, so Add() draws NOTHING; it only wires shapes.
          // The window for setting the seed is therefore much wider than before: anywhere before
          // the first forward pass. (Previously Add() drew the output layer's weights and a seed set
          // afterwards no longer reached it → the model was only PARTIALLY reproducible.)
          NeuralNetwork Build()
          {
              var m = new NeuralNetwork(new Dense(2, new Linear()), new SGD(0.01), new MSE());
              m.Layers.Add(new Dense(3, new ReLU()));
              m.Context.Seed = 42;                        // after Add() but before the forward pass → OK
              m.GetResults(new Tensor(new double[] { 1, 1 }));
              return m;
          }

          Assert.Equal(
              ((Dense)Build().Layers.Layers[0]).Neurons[0].Weights,
              ((Dense)Build().Layers.Layers[0]).Neurons[0].Weights);
      }

      [Fact]
      public void Setting_seed_after_weights_were_drawn_throws()
      {
          // The guard still applies; it just moved to where values are actually drawn: after the first forward.
          var model = new NeuralNetwork(new Dense(2, new Linear()), new SGD(0.01), new MSE());
          model.Layers.Add(new Dense(3, new ReLU()));
          model.GetResults(new Tensor(new double[] { 1, 1 }));   // values are drawn here

          var ex = Assert.Throws<InvalidOperationException>(() => model.Context.Seed = 42);

          Assert.Contains("constructor", ex.Message);   // the message should say how to fix it
      }

      [Fact]
      public void Setting_random_after_use_throws_too()
      {
          var model = new NeuralNetwork(new Dense(2, new Linear()), new SGD(0.01), new MSE());
          model.Layers.Add(new Dense(3, new ReLU()));
          model.GetResults(new Tensor(new double[] { 1, 1 }));

          Assert.Throws<InvalidOperationException>(() => model.Context.Random = new Random(42));
      }

      [Fact]
      public void LoadModel_accepts_seed_for_further_training()
      {
          // weights come from the file, but data shuffling and sample selection during further
          // training must be fixable too
          string path = System.IO.Path.Combine(System.IO.Path.GetTempPath(), "mdnn_seed_" + Guid.NewGuid().ToString("N"));
          try
          {
              var src = new NeuralNetwork(new Dense(1, new Linear()), new SGD(0.01), new MSE(), seed: 1);
              src.GetResults(new Tensor(new double[] { 1, 1 }));
              src.SaveAsJson(path);

              var loaded = NeuralNetwork.LoadModel(path, seed: 99);

              Assert.Equal(99, loaded.Context.Seed);
          }
          finally { try { System.IO.File.Delete(path + ".json"); } catch { } }
      }

      // ------------------------------------------------------------------------------
      // This used to be a KNOWN LIMITATION with Assert.NotEqual: a seed only held for a model
      // assembled the same way, because Build materialized parameters during assembly and the
      // draw order depended on the call history (Add/Insert), not on the resulting architecture.
      //
      // Since materialization was unified into a single pass this no longer holds,
      // so the test is FLIPPED to Assert.Equal, exactly as that comment asked for.
      // ------------------------------------------------------------------------------
      [Fact]
      public void Seed_survives_different_assembly_order()
      {
          NeuralNetwork ViaAddAdd()
          {
              var m = new NeuralNetwork(new Dense(2, new Linear()), new SGD(0.01), new MSE(), seed: 42);
              m.Layers.Add(new Dense(3, new ReLU()));
              m.Layers.Add(new Dense(4, new ReLU()));
              m.GetResults(new Tensor(new double[] { 1, 1 }));
              return m;
          }

          NeuralNetwork ViaAddInsert()
          {
              var m = new NeuralNetwork(new Dense(2, new Linear()), new SGD(0.01), new MSE(), seed: 42);
              m.Layers.Add(new Dense(4, new ReLU()));
              m.Layers.Insert(0, new Dense(3, new ReLU()));
              m.GetResults(new Tensor(new double[] { 1, 1 }));
              return m;
          }

          var a = ViaAddAdd();
          var b = ViaAddInsert();

          // same resulting topology…
          Assert.Equal(a.Layers.Layers.Count, b.Layers.Layers.Count);
          for (int i = 0; i < a.Layers.Layers.Count; i++)
              Assert.Equal(
                  ((Dense)a.Layers.Layers[i]).Neurons.Count,
                  ((Dense)b.Layers.Layers[i]).Neurons.Count);

          // …and now also the same weights, in ALL layers
          for (int i = 0; i < a.Layers.Layers.Count; i++)
              Assert.Equal(
                  ((Dense)a.Layers.Layers[i]).Neurons[0].Weights,
                  ((Dense)b.Layers.Layers[i]).Neurons[0].Weights);
      }
  }
