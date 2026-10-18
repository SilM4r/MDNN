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

  public class SmokeTrainTest:GradientCheckTestBase
  {
      [Fact]
      public void Smoke_linear_regression_learns()
      {
          // y = 2x + 1
          double[][] X = { new[]{-1.0}, new[]{0.0}, new[]{1.0}, new[]{2.0} };
          double[][] Y = { new[]{-1.0}, new[]{1.0}, new[]{3.0}, new[]{5.0} };

          var model = new My_DNN.MDNN(new Dense(1, new Linear()), new Adam(0.05), new MSE());

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

          var model = new My_DNN.MDNN(new Dense(1, new Sigmoid()), new Adam(0.05), new MSE());
          model.Layers.Add(new Dense(8, new Tanh()));  

          for (int epoch = 0; epoch < 3000; epoch++)
          for (int n = 0; n < 4; n++)
          {
              model.Train.Fit(new Tensor(X[n]), new Tensor(Y[n]));
              model.Train.UpdateParams();
          }

          // robustnější než přísný práh loss: zaokrouhlené výstupy musí sedět
          for (int n = 0; n < 4; n++)
              Assert.Equal(Y[n][0], Math.Round(model.GetResults(new Tensor(X[n])).Data[0]));
      }
      
      [Fact]
      public void SimpleTrainLoop_small_epochs_does_not_throw()
      {
          double[][] X = { new[]{0.0}, new[]{1.0} };
          double[][] Y = { new[]{0.0}, new[]{1.0} };

          var model = new My_DNN.MDNN(new Dense(1, new Linear()), new SGD(0.01), new MSE());
          model.Train.AutoSaveInTrainLoop = false;      // ať to nepíše .json na disk

          model.Train.SimpleTrainLoop(X, Y, 10);        // 10 < 100 → před opravou DivideByZeroException
      }
      
      [Fact]
      public void Dense_init_symmetric_and_fanin_scaled()
      {
          int fanIn = 1000;                                   // velký vzorek
          double[] w = new Neuron(fanIn, new ReLu()).Weights;

          Assert.Contains(w, x => x < 0);                     // symetrie: MUSÍ být i záporné
          Assert.Contains(w, x => x > 0);                     //           i kladné

          double var = w.Select(x => x * x).Average();        // mean≈0 → var ≈ E[x²]
          double heVar = 2.0 / fanIn;
          Assert.InRange(var, heVar * 0.5, heVar * 1.5);      // He škála řádově sedí
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
          Assert.Equal(0.42, w, 12);           // ověří, že se momentum akumuluje
      }
      
      [Fact]
      public void Adam_first_step_is_bias_corrected()
      {
          var adam = new Adam(0.05);
          double w = adam.Update(1.0, 2.0, 0);
          // s bias correction: první krok ≈ L·sign(g) nezávisle na velikosti g → 1 - 0.05 = 0.95
          Assert.Equal(0.95, w, 6);
      }

      [Fact]
      public void Adam_bias_correction_follows_each_parameters_own_step()
      {
          // Korekce 1 − β^t se pamatuje pro poslední t (v tréninku mají všechny váhy
          // neuronu stejné t). Tady se t střídá — parametr 0 je ve 3. kroku, parametr 1
          // v 1. — a cache se nesmí splést: parametr 1 musí dostat korekci PRVNÍHO kroku.
          //
          // Očekávané hodnoty se počítají nezávisle přímo ze vzorce (ne druhou instancí
          // Adamu — ta by měla tutéž chybu) a gradient se mění: při konstantním gradientu
          // vyjde korigovaný krok pořád stejně a špatná korekce by se neprojevila.
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

          double first = adam.Update(1.0, 5.0, 1);    // parametr 1, krok 1
          double fourth = adam.Update(1.0, 4.0, 0);   // parametr 0, krok 4

          Assert.Equal(Expected(new[] { 5.0 }), first, 12);
          Assert.Equal(Expected(new[] { 1.0, 3.0, -2.0, 4.0 }), fourth, 12);
      }
  }

  public class IndependenceTests : GradientCheckTestBase
  {
      [Fact]
      public void Two_models_built_interleaved_are_independent()
      {
          // Prokládaně postavíme DVA modely s různou architekturou a loss.
          // (Dřív: sdílený statický layersList → B by odvodil velikosti z A. Teď instanční.)
          var a = new My_DNN.MDNN(new Dense(1, new Linear()), new SGD(0.01), new MSE());
          var b = new My_DNN.MDNN(new Dense(3, new Softmax()), new Adam(0.01), new CrossEntropy());
          a.Layers.Add(new Dense(4, new Tanh()));   // A: skrytá 4 neurony, SGD
          b.Layers.Add(new Dense(6, new Tanh()));   // B: skrytá 6 neuronů, Adam

          Assert.IsType<MSE>(a.Loss);               // konfigurace nezávislá
          Assert.IsType<CrossEntropy>(b.Loss);

          a.Train.Fit(new Tensor(new double[] { 0.5 }), new Tensor(new double[] { 1.0 }));
          b.Train.Fit(new Tensor(new double[] { 0.5, 0.3, 0.2 }), new Tensor(new double[] { 0.0, 1.0, 0.0 }));

          // per-model optimizer: A má SGD, B má Adam (dřív sdílený static → šláply by si)
          Assert.IsType<SGD>(((Dense)a.Layers.Layers[0]).Neurons[0].Optimizer);
          Assert.IsType<Adam>(((Dense)b.Layers.Layers[0]).Neurons[0].Optimizer);

          // po trénování obou: každý si drží svou architekturu i loss (nešlapou si po sobě)
          Assert.Equal(1, a.GetResults(new Tensor(new double[] { 0.5 })).Data.Length);
          Assert.Equal(3, b.GetResults(new Tensor(new double[] { 0.5, 0.3, 0.2 })).Data.Length);
          Assert.IsType<MSE>(a.Loss);
          Assert.IsType<CrossEntropy>(b.Loss);
      }
  }

  public class EmptyMiniBatchTests : GradientCheckTestBase
  {
      // 2c-6: Neuron.Update_weights_bias dělí mini_batch_size, které je 0 → 0/0 = NaN,
      // tiše přes celý model.
      [Fact]
      public void UpdateParams_without_backprop_does_not_produce_NaN()
      {
          var model = new My_DNN.MDNN(new Dense(2, new Linear()), new SGD(0.01), new MSE());
          model.Layers.SetInputSizeForFirstLayer(new int[] { 2 });

          var x = new Tensor(new double[] { 1, 2 });
          model.GetResults(x);
          model.Train.UpdateParams();                             // žádný backprop před tím

          Assert.All(model.GetResults(x).Data, v => Assert.False(double.IsNaN(v), "výstup je NaN"));
      }
  }

  public class RepeatedTrainLoopTests : GradientCheckTestBase
  {
      // 2c-3: ConsoleControler drží static _time / _lastEpochInfo. Při druhém TrainLoopu
      // vyjde pastEpochs == 0 → (subTime * n) / 0 → OverflowException.
      // Jde přímo proti cíli Fáze 3 (víc modelů v procesu = předpoklad AutoML).
      [Fact]
      public void Two_train_loops_in_one_process_do_not_throw()
      {
          for (int run = 1; run <= 2; run++)
          {
              var model = new My_DNN.MDNN(new Dense(1, new Linear()), new SGD(0.05), new MSE());
              model.Train.ShowLossChartInTrainLoop = false;
              model.Train.ShowModelInfoIntrainLoop = false;
              model.Train.AutoSaveInTrainLoop = false;
              model.Train.TestNeuralNetworkAfterTraining = false;
              model.Train.NumberOfShowEpochInConsole = 1;        // jeden report za běh

              double[][] X = new double[20][];
              double[][] Y = new double[20][];
              for (int i = 0; i < 20; i++)
              {
                  X[i] = new double[] { i * 0.05 };
                  Y[i] = new double[] { i * 0.10 };
              }

              var ex = Record.Exception(() => model.Train.TrainLoop(X, Y, 50, 4));
              Assert.True(ex == null, $"běh {run} spadl: {ex?.GetType().Name}: {ex?.Message}");
          }
      }

      // 2c-9: klamp v Train.PreparationForTrainLoop zapisuje do VEŘEJNÉHO pole,
      // takže si uživatelovo nastavení natrvalo přepíše.
      [Fact]
      public void NumberOfShowEpochInConsole_survives_training()
      {
          var model = new My_DNN.MDNN(new Dense(1, new Linear()), new SGD(0.05), new MSE());
          model.Train.ShowLossChartInTrainLoop = false;
          model.Train.ShowModelInfoIntrainLoop = false;
          model.Train.AutoSaveInTrainLoop = false;
          model.Train.TestNeuralNetworkAfterTraining = false;

          uint before = model.Train.NumberOfShowEpochInConsole;   // default 100

          double[][] X = new double[20][];
          double[][] Y = new double[20][];
          for (int i = 0; i < 20; i++)
          {
              X[i] = new double[] { i * 0.05 };
              Y[i] = new double[] { i * 0.10 };
          }
          model.Train.TrainLoop(X, Y, 5);                         // 5 epoch < 100 → klamp

          Assert.Equal(before, model.Train.NumberOfShowEpochInConsole);
      }
  }

  // ==========================================================================================
  //  Fáze 2c — dávka 2 (2c-8, 2c-14)
  // ==========================================================================================

  public class DivergenceTests : GradientCheckTestBase
  {
      // 2c-8: při NaN loss se volalo Environment.Exit(0) — knihovna zabila hostitelský
      // proces, a ještě s kódem 0 = „úspěch". Volající neměl šanci zareagovat.
      // (Tenhle test by se ve staré verzi ani nedal napsat: Exit by shodil test runner.)
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

      private static My_DNN.MDNN DivergingModel()
      {
          // absurdní learning rate → váhy vystřelí do ±∞ a pak NaN
          var model = new My_DNN.MDNN(new Dense(1, new Linear()), new SGD(1e12), new MSE());
          model.Train.ShowLossChartInTrainLoop = false;
          model.Train.ShowModelInfoIntrainLoop = false;
          model.Train.AutoSaveInTrainLoop = false;
          model.Train.TestNeuralNetworkAfterTraining = false;
          return model;
      }

      [Fact]
      public void TrainLoop_on_divergence_throws_instead_of_killing_process()
      {
          var (X, Y) = Data();
          var model = DivergingModel();

          var ex = Assert.Throws<TrainingDivergedException>(() => model.Train.TrainLoop(X, Y, 50, 4));

          Assert.False(double.IsFinite(ex.Loss), "výjimka má nést tu neplatnou loss");
          Assert.Contains("learning rate", ex.Message);   // hláška má radit, co s tím
      }

      [Fact]
      public void SimpleTrainLoop_on_divergence_throws_too()
      {
          // dřív tu byl jen Console.WriteLine + return → volající nepoznal rozdíl mezi
          // „dotrénováno" a „rozpadlo se to v první epoše"
          var (X, Y) = Data();
          var model = DivergingModel();

          Assert.Throws<TrainingDivergedException>(() => model.Train.SimpleTrainLoop(X, Y, 50, 4));
      }

      [Fact]
      public void Divergence_is_catchable_and_process_survives()
      {
          // přesně scénář AutoML runneru: kandidát zdivergoval, chytím to a jedu dál
          var (X, Y) = Data();
          bool caught = false;

          try { DivergingModel().Train.TrainLoop(X, Y, 50, 4); }
          catch (TrainingDivergedException) { caught = true; }

          Assert.True(caught);

          // po chycení musí jít normálně natrénovat další model
          var healthy = new My_DNN.MDNN(new Dense(1, new Linear()), new SGD(0.01), new MSE());
          healthy.Train.ShowLossChartInTrainLoop = false;
          healthy.Train.ShowModelInfoIntrainLoop = false;
          healthy.Train.AutoSaveInTrainLoop = false;
          healthy.Train.TestNeuralNetworkAfterTraining = false;
          healthy.Train.TrainLoop(X, Y, 10, 2);

          Assert.True(double.IsFinite(healthy.GetResults(new Tensor(new double[] { 1.0 })).Data[0]));
      }
  }

  public class EarlyStoppingTests : GradientCheckTestBase
  {
      [Fact]
      public void EarlyStopping_stops_before_total_epochs()
      {
          // y = x regrese; minDelta nastavíme absurdně vysoko → po prvním reportu (zlepšení
          // z MaxValue) se už NIC nepočítá jako zlepšení → po `patience` reportech se zastaví.
          // Bez early stoppingu by loop dojel do totalEpoch (100).
          var model = new My_DNN.MDNN(new Dense(1, new Linear()), new SGD(0.01), new MSE());

          double[][] X = new double[20][];
          double[][] Y = new double[20][];
          for (int i = 0; i < 20; i++)
          {
              X[i] = new double[] { i * 0.1 };
              Y[i] = new double[] { i * 0.1 };
          }

          model.Train.ShowLossChartInTrainLoop = false;      // ať v testu neskáče graf
          model.Train.ShowModelInfoIntrainLoop = false;
          model.Train.AutoSaveInTrainLoop = false;           // žádné psaní na disk
          model.Train.TestNeuralNetworkAfterTraining = false;
          model.Train.NumberOfShowEpochInConsole = 100;      // report každou epochu (totalEpoch=100)

          model.Train.EarlyStoppingEnabled = true;
          model.Train.EarlyStoppingPatience = 3;
          model.Train.EarlyStoppingMinDelta = 1e9;           // nic se nebude počítat jako zlepšení

          model.Train.TrainLoop(X, Y, 100, 1);

          // 1 report se zlepšením (z MaxValue) + 3 bez zlepšení → stop hluboko pod 100 epochami
          Assert.True(model.Train.CurrentEpoch < 20,
              $"CurrentEpoch = {model.Train.CurrentEpoch}, early stopping mělo zastavit brzy");
      }
  }

  // ==========================================================================================
  //  Fáze 4 — konvenční epochy (plný průchod po dávkách místo náhodných tahů s opakováním)
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

      private static My_DNN.MDNN Quiet(int? seed = null)
      {
          var m = new My_DNN.MDNN(new Dense(1, new Linear()), new SGD(0.01), new MSE(), seed);
          m.Train.ShowLossChartInTrainLoop = false;
          m.Train.ShowModelInfoIntrainLoop = false;
          m.Train.AutoSaveInTrainLoop = false;
          m.Train.TestNeuralNetworkAfterTraining = false;
          return m;
      }

      [Theory]
      [InlineData(1)]     // 14 vzorků / dávka 1  → 14 kroků na epochu
      [InlineData(4)]     // 14 / 4 = 3.5         → 4 kroky (poslední dávka neúplná)
      [InlineData(7)]     // 14 / 7               → 2 kroky
      [InlineData(14)]    // celý set v jedné dávce → 1 krok
      [InlineData(50)]    // dávka větší než set   → pořád 1 krok
      public void One_epoch_runs_one_optimizer_step_per_batch(int batchSize)
      {
          var (X, Y) = Data(20);
          var model = Quiet();

          model.Train.TrainLoop(X, Y, 1, (uint)batchSize);

          int trainCount = model.Train.TrainDataInputs!.Shape[0];      // 0.7 z 20 = 14
          ulong expected = (ulong)((trainCount + batchSize - 1) / batchSize);

          Assert.Equal(14, trainCount);
          Assert.Equal(expected, model.Train.OptimizerSteps);
      }

      [Fact]
      public void Epoch_counter_counts_epochs_not_optimizer_steps()
      {
          // Dřív UpdateParams() inkrementoval čítač epoch a vycházelo to nastejno, protože
          // na „epochu" připadal jeden krok. Teď se to musí rozejít.
          var (X, Y) = Data(20);
          var model = Quiet();

          model.Train.TrainLoop(X, Y, 3, 1);

          Assert.Equal(3u, model.Train.CurrentEpoch);          // 3 průchody daty
          Assert.Equal(42ul, model.Train.OptimizerSteps);      // 3 × 14 dávek
      }

      [Fact]
      public void Manual_training_counts_steps_but_no_epochs()
      {
          // Ruční Fit + UpdateParams žádné epochy nemá — hlásit je jako epochy by lhalo.
          var model = Quiet();
          model.Layers.SetInputSizeForFirstLayer(new int[] { 1 });

          for (int i = 0; i < 5; i++)
          {
              model.Train.Fit(new Tensor(new double[] { 1 }), new Tensor(new double[] { 0.5 }));
              model.Train.UpdateParams();
          }

          Assert.Equal(0u, model.Train.CurrentEpoch);
          Assert.Equal(5ul, model.Train.OptimizerSteps);
      }

      // Jádro celé změny: za epochu musí model vidět KAŽDÝ trénovací vzorek PRÁVĚ JEDNOU.
      // Dřív se losovalo s opakováním, takže některé vzorky nikdy a jiné víckrát.
      //
      // Důkaz bez ručního počítání gradientů: jedna epocha s dávkou = celý train set
      // (tedy jeden krok optimizeru) se musí rovnat ručnímu Fit přes všechny vzorky
      // právě jednou + jeden UpdateParams. Gradienty se v dávce SČÍTAJÍ, takže na pořadí
      // uvnitř dávky nezáleží. Kdyby smyčka nějaký vzorek vynechala nebo zdvojila,
      // součet by nesouhlasil.
      [Fact]
      public void One_epoch_equals_seeing_every_sample_exactly_once()
      {
          var (X, Y) = Data(20);

          // referenční model: 0 epoch → jen se rozdělí dataset, váhy zůstanou počáteční
          var reference = Quiet(seed: 5);
          reference.Train.TrainLoop(X, Y, 0);

          Tensor trainX = reference.Train.TrainDataInputs!;
          Tensor trainY = reference.Train.TrainDataCurrentOutput!;
          int n = trainX.Shape[0];

          // model pod testem: jedna epocha, celý train set v jedné dávce
          var underTest = Quiet(seed: 5);
          underTest.Train.TrainLoop(X, Y, 1, (uint)n);

          // reference: ručně každý vzorek právě jednou, pak jeden krok
          for (int i = 0; i < n; i++)
          {
              reference.Train.Fit(trainX.GetTensorValue([i]), trainY.GetTensorValue([i]));
          }
          reference.Train.UpdateParams();

          double[] expected = ((Dense)reference.Layers.Layers[0]).Neurons[0].Weights;
          double[] actual = ((Dense)underTest.Layers.Layers[0]).Neurons[0].Weights;

          // ne bit-identické: sčítání floatů není asociativní a dávka se prochází
          // v zamíchaném pořadí, takže se poslední bity můžou lišit
          Assert.Equal(expected.Length, actual.Length);
          for (int i = 0; i < expected.Length; i++)
          {
              Assert.True(Math.Abs(expected[i] - actual[i]) < 1e-12,
                  $"váha {i}: očekáváno {expected[i]}, dostal {actual[i]}");
          }
      }

      [Fact]
      public void Zero_batch_size_is_rejected()
      {
          // dřív se s nulou tiše netrénovalo; nově by "start += 0" byla nekonečná smyčka
          var (X, Y) = Data(20);

          Assert.Throws<ArgumentException>(() => Quiet().Train.TrainLoop(X, Y, 5, 0));
      }

      [Fact]
      public void Epochs_are_reproducible_with_a_seed()
      {
          // míchání pořadí každou epochu čerpá z Context.Random → se seedem musí vyjít stejně
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
  //  Nálezy z procházení příkladů (2026-08-11) — obojí byly starší chyby, ne regrese
  // ==========================================================================================

  public class SmallDatasetTests : GradientCheckTestBase
  {
      // Rozdělení počítalo (int)(n * 0.15), což je pro n < 7 NULA → Slice(x, 0) hodilo
      // "Invalid slice range!". Každý dataset menší než 7 vzorků tedy TrainLoop shodil.
      // Vylezlo to při ověřování ukázky z README, která má 4 vzorky.
      [Theory]
      [InlineData(3)]
      [InlineData(4)]
      [InlineData(5)]
      [InlineData(6)]
      [InlineData(7)]
      [InlineData(20)]
      public void Tiny_dataset_still_splits_into_three_nonempty_sets(int sampleCount)
      {
          var model = new My_DNN.MDNN(new Dense(1, new Linear()), new SGD(0.01), new MSE());

          var xs = new double[sampleCount, 1];
          var ys = new double[sampleCount, 1];
          for (int i = 0; i < sampleCount; i++) { xs[i, 0] = i; ys[i, 0] = i * 2; }

          model.Train.DividingDataIntoDatasets(new Tensor(xs), new Tensor(ys));

          Assert.True(model.Train.TrainDataInputs!.Shape[0] >= 1, "train je prázdný");
          Assert.True(model.Train.ValidDataInputs!.Shape[0] >= 1, "valid je prázdný");
          Assert.True(model.Train.TestDataInputs!.Shape[0] >= 1, "test je prázdný");

          Assert.Equal(sampleCount,
              model.Train.TrainDataInputs.Shape[0]
              + model.Train.ValidDataInputs.Shape[0]
              + model.Train.TestDataInputs.Shape[0]);
      }

      [Fact]
      public void Two_samples_are_rejected_with_a_clear_message()
      {
          var model = new My_DNN.MDNN(new Dense(1, new Linear()), new SGD(0.01), new MSE());
          var xs = new double[2, 1];
          var ys = new double[2, 1];

          var ex = Assert.Throws<ArgumentException>(
              () => model.Train.DividingDataIntoDatasets(new Tensor(xs), new Tensor(ys)));
          Assert.Contains("SetDatasets", ex.Message);
      }

      [Fact]
      public void Readme_quick_start_with_four_samples_trains()
      {
          // přesně ukázka z README (4 vzorky) — dřív spadla na "Invalid slice range!"
          double[][] X = { new double[] { 0 }, new double[] { 1 }, new double[] { 2 }, new double[] { 3 } };
          double[][] Y = { new double[] { 0 }, new double[] { 2 }, new double[] { 4 }, new double[] { 6 } };

          var model = new My_DNN.MDNN(new Dense(1, new Linear()), new SGD(0.01), new MSE());
          model.Layers.Add(new Dense(8, new ReLu()));
          model.Train.ShowLossChartInTrainLoop = false;
          model.Train.ShowModelInfoIntrainLoop = false;
          model.Train.AutoSaveInTrainLoop = false;
          model.Train.TestNeuralNetworkAfterTraining = false;

          model.Train.TrainLoop(X, Y, 5, 1);

          Assert.Equal(5u, model.Train.CurrentEpoch);
      }
  }

  public class TargetSizeMismatchTests : GradientCheckTestBase
  {
      // Když cíl nemá tolik hodnot, kolik má výstupní vrstva neuronů, padal
      // IndexOutOfRangeException z útrob Gradient.GetGradients — z toho se příčina nepozná.
      // Taky nález z ukázek: README má `new Dense(3)` a čtenář snadno dá skalární cíl.
      [Fact]
      public void Mismatched_target_size_gives_a_clear_error()
      {
          var model = new My_DNN.MDNN(new Dense(3, new Linear()), new SGD(0.01), new MSE());
          model.Layers.SetInputSizeForFirstLayer(new int[] { 1 });

          var ex = Assert.Throws<ArgumentException>(() =>
              model.Train.Fit(new Tensor(new double[] { 1 }), new Tensor(new double[] { 0 })));

          Assert.Contains("3", ex.Message);
          Assert.Contains("1", ex.Message);
      }
  }

  public class SequenceScoreTests : GradientCheckTestBase
  {
      // 2c-11: v sekvenční větvi se maxScore zvyšovalo o počet výstupních NEURONŮ, zatímco
      // score o 1 za časový krok → poměr score/maxScore byl nesmysl. Nesekvenční větev
      // to měla správně.
      [Fact]
      public void Sequential_test_counts_timesteps_not_output_neurons()
      {
          var model = new My_DNN.MDNN(new Dense(2, new Linear()), new SGD(0.01), new MSE());
          model.Context.SequenceTrain = true;

          // 2 sekvence × 3 kroky × 2 featury, výstup 2 neurony
          var inputs  = new Tensor(new double[2 * 3 * 2], new int[] { 2, 3, 2 });
          var targets = new Tensor(new double[2 * 3 * 2], new int[] { 2, 3, 2 });

          (int score, int maxScore) = model.Train.TestNeuralNetwork(inputs, targets, showInConsole: false);

          Assert.Equal(6, maxScore);          // 2 sekvence × 3 kroky; dřív 12 (× 2 neurony)
          Assert.InRange(score, 0, maxScore); // skóre nesmí přelézt maximum
      }
  }
