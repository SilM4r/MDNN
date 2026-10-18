using System;
  using Xunit;
  using My_DNN;
  using My_DNN.Layers;
  using My_DNN.Activation_functions;
  using My_DNN.Layers.classes;
  using My_DNN.Optimizers;
  using My_DNN.Loss_functions;
  using System.Linq;

  namespace MDNN.Tests;

  // ==========================================================================================
  //  Fáze 4 — seed / reprodukovatelnost (odemyká díru 1b-5)
  // ==========================================================================================

  public class SeedReproducibilityTests : GradientCheckTestBase
  {
      // Dřív existovaly TŘI nezávislé zdroje náhody a žádný nešel nastavit:
      // GeneralNeuralNetworkSettings.rnd (váhy), Train._rnd (výběr vzorků),
      // Random.Shared (ShuffleTensor). Stejný experiment tedy nešlo spustit dvakrát —
      // což je předpoklad pro férové porovnání dvou kandidátů v AutoML.
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

      private static My_DNN.MDNN Build(int? seed)
      {
          var model = new My_DNN.MDNN(new Dense(1, new Linear()), new SGD(0.01), new MSE(), seed);
          model.Layers.Add(new Dense(4, new ReLu()));
          model.Train.ShowLossChartInTrainLoop = false;
          model.Train.ShowModelInfoIntrainLoop = false;
          model.Train.AutoSaveInTrainLoop = false;
          model.Train.TestNeuralNetworkAfterTraining = false;
          return model;
      }

      [Fact]
      public void Same_seed_gives_identical_initial_weights()
      {
          var a = Build(42);
          var b = Build(42);

          a.GetResults(new Tensor(new double[] { 1, 1 }));   // spustí wiring = inicializaci vah
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
          // nejtvrdší varianta: seed musí pokrýt i míchání datasetu a výběr vzorků,
          // ne jen inicializaci vah
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
          double[] validA = a.Train.ValidDataInputs!.Data;

          var b = Build(7);
          b.Train.TrainLoop(X, Y, 5);

          Assert.Equal(validA, b.Train.ValidDataInputs!.Data);   // shodné míchání → shodný split
      }

      [Fact]
      public void Without_seed_behaviour_is_unchanged()
      {
          // zpětná kompatibilita: bez seedu se pořád tahá ze sdíleného globálního
          // generátoru, takže dva modely NEmají stejné váhy
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
          My_DNN.MDNN WithConv(int seed)
          {
              var m = new My_DNN.MDNN(new Dense(2, new Linear()), new SGD(0.01), new MSE(), seed);
              m.Layers.Add(new Conv(2, 3, new ReLu(), "valid"));
              m.Layers.SetInputSizeForFirstLayer(new int[] { 6, 6, 1 });
              return m;
          }

          var a = (Conv)WithConv(99).Layers.Layers[0];
          var b = (Conv)WithConv(99).Layers.Layers[0];

          Assert.Equal(a.Kernel[0][0][0], b.Kernel[0][0][0]);
          Assert.Equal(a.Biases, b.Biases);
      }
  }

  public class SeedErgonomicsTests : GradientCheckTestBase
  {
      [Fact]
      public void Seed_can_be_read_back()
      {
          // AutoML si u kandidáta potřebuje poznamenat, čím se běh dá zopakovat
          var model = new My_DNN.MDNN(new Dense(2, new Linear()), new SGD(0.01), new MSE(), seed: 1234);
          Assert.Equal(1234, model.Context.Seed);
      }

      [Fact]
      public void Seed_is_null_when_not_given()
      {
          var model = new My_DNN.MDNN(new Dense(2, new Linear()), new SGD(0.01), new MSE());
          Assert.Null(model.Context.Seed);
      }

      [Fact]
      public void Seed_can_be_set_via_context_before_layers_are_built()
      {
          My_DNN.MDNN Build()
          {
              var m = new My_DNN.MDNN(new Dense(2, new Linear()), new SGD(0.01), new MSE());
              m.Context.Seed = 42;                       // ještě se nic nelosovalo → OK
              m.Layers.Add(new Dense(3, new ReLu()));
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
          // Od sjednocení materializace do jednoho průchodu Add() NELOSUJE — jen zapojí tvary.
          // Okno pro nastavení seedu je proto mnohem širší než dřív: stačí to stihnout před
          // prvním forwardem. (Dřív Add() vylosoval váhy výstupní vrstvy a seed nastavený
          // potom už na ni nedosáhl → model vyšel jen ČÁSTEČNĚ reprodukovatelný.)
          My_DNN.MDNN Build()
          {
              var m = new My_DNN.MDNN(new Dense(2, new Linear()), new SGD(0.01), new MSE());
              m.Layers.Add(new Dense(3, new ReLu()));
              m.Context.Seed = 42;                        // po Add(), ale před forwardem → OK
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
          // Guard pořád platí — jen se posunul tam, kde se opravdu losuje: za první forward.
          var model = new My_DNN.MDNN(new Dense(2, new Linear()), new SGD(0.01), new MSE());
          model.Layers.Add(new Dense(3, new ReLu()));
          model.GetResults(new Tensor(new double[] { 1, 1 }));   // tady se losuje

          var ex = Assert.Throws<InvalidOperationException>(() => model.Context.Seed = 42);

          Assert.Contains("konstruktoru", ex.Message);   // hláška má říct, kudy z toho ven
      }

      [Fact]
      public void Setting_random_after_use_throws_too()
      {
          var model = new My_DNN.MDNN(new Dense(2, new Linear()), new SGD(0.01), new MSE());
          model.Layers.Add(new Dense(3, new ReLu()));
          model.GetResults(new Tensor(new double[] { 1, 1 }));

          Assert.Throws<InvalidOperationException>(() => model.Context.Random = new Random(42));
      }

      [Fact]
      public void LoadModel_accepts_seed_for_further_training()
      {
          // váhy se berou ze souboru, ale míchání dat a výběr vzorků při dotrénování
          // musí jít zafixovat taky
          string path = System.IO.Path.Combine(System.IO.Path.GetTempPath(), "mdnn_seed_" + Guid.NewGuid().ToString("N"));
          try
          {
              var src = new My_DNN.MDNN(new Dense(1, new Linear()), new SGD(0.01), new MSE(), seed: 1);
              src.GetResults(new Tensor(new double[] { 1, 1 }));
              src.SaveAsJson(path);

              var loaded = My_DNN.MDNN.LoadModel(path, seed: 99);

              Assert.Equal(99, loaded.Context.Seed);
          }
          finally { try { System.IO.File.Delete(path + ".json"); } catch { } }
      }

      // ------------------------------------------------------------------------------
      // Dřív tu bylo ZNÁMÉ OMEZENÍ s Assert.NotEqual: seed platil jen pro stejně poskládaný
      // model, protože LayerAdjustment materializovalo parametry už při skládání a pořadí
      // losování bylo funkcí historie volání (Add/Insert), ne výsledné architektury.
      //
      // Po sjednocení materializace do jediného průchodu (seed krok 3) to platit přestalo,
      // takže test je OTOČENÝ na Assert.Equal — přesně jak si ten komentář žádal.
      // ------------------------------------------------------------------------------
      [Fact]
      public void Seed_survives_different_assembly_order()
      {
          My_DNN.MDNN ViaAddAdd()
          {
              var m = new My_DNN.MDNN(new Dense(2, new Linear()), new SGD(0.01), new MSE(), seed: 42);
              m.Layers.Add(new Dense(3, new ReLu()));
              m.Layers.Add(new Dense(4, new ReLu()));
              m.GetResults(new Tensor(new double[] { 1, 1 }));
              return m;
          }

          My_DNN.MDNN ViaAddInsert()
          {
              var m = new My_DNN.MDNN(new Dense(2, new Linear()), new SGD(0.01), new MSE(), seed: 42);
              m.Layers.Add(new Dense(4, new ReLu()));
              m.Layers.Insert(0, new Dense(3, new ReLu()));
              m.GetResults(new Tensor(new double[] { 1, 1 }));
              return m;
          }

          var a = ViaAddAdd();
          var b = ViaAddInsert();

          // stejná výsledná topologie…
          Assert.Equal(a.Layers.Layers.Count, b.Layers.Layers.Count);
          for (int i = 0; i < a.Layers.Layers.Count; i++)
              Assert.Equal(
                  ((Dense)a.Layers.Layers[i]).Neurons.Count,
                  ((Dense)b.Layers.Layers[i]).Neurons.Count);

          // …a nově i stejné váhy, ve VŠECH vrstvách
          for (int i = 0; i < a.Layers.Layers.Count; i++)
              Assert.Equal(
                  ((Dense)a.Layers.Layers[i]).Neurons[0].Weights,
                  ((Dense)b.Layers.Layers[i]).Neurons[0].Weights);
      }
  }
