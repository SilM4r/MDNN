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

  public class DataSplitTests : GradientCheckTestBase
  {
      // Dřív tu byl test `DividingDataIntoDatasets_userValidNoTest_carves_test_from_original_valid`,
      // který hlídal větev „uživatel dodal VALID, ale ne TEST → vyřízni test z validu".
      // Ta větev byla ZRUŠENÁ spolu se settery: datasety se teď dodávají jedním voláním
      // SetDatasets(train, valid, test) a použijí se, jak jsou. Regrese, kterou ten test
      // hlídal (test se krájel z už zmenšeného validu → "Invalid slice range!"), tedy
      // nemůže nastat, protože ta cesta v kódu není. Nahrazeno testy nové sémantiky níž.

      [Fact]
      public void Explicit_datasets_are_used_as_given()
      {
          var model = new My_DNN.MDNN(new Dense(1, new Linear()), new SGD(0.01), new MSE());

          var (trainX, trainY) = MakeSamples(30);
          var (validX, validY) = MakeSamples(8);
          var (testX, testY) = MakeSamples(5);

          model.Train.SetDatasets(
              new LabeledData(trainX, trainY),
              new LabeledData(validX, validY),
              new LabeledData(testX, testY));

          model.Train.DividingDataIntoDatasets(trainX, trainY);

          // žádné krájení, žádné poměry — přesně to, co uživatel dal
          Assert.Equal(30, model.Train.TrainDataInputs!.Shape[0]);
          Assert.Equal(8, model.Train.ValidDataInputs!.Shape[0]);
          Assert.Equal(5, model.Train.TestDataInputs!.Shape[0]);
          Assert.True(model.Train.HasExplicitDatasets);
      }

      [Fact]
      public void Missing_test_is_carved_from_valid()
      {
          // Chybějící sada se ukrojí ze SVÉ SOUROZENKYNĚ, ne z trainu. Poměr valid:test
          // je defaultně 0.15:0.15, takže z 8 validních vzorků vyjde 4 valid / 4 test.
          var model = new My_DNN.MDNN(new Dense(1, new Linear()), new SGD(0.01), new MSE());

          var (trainX, trainY) = MakeSamples(30);
          var (validX, validY) = MakeSamples(8);

          model.Train.SetDatasets(new LabeledData(trainX, trainY), new LabeledData(validX, validY));
          model.Train.DividingDataIntoDatasets(trainX, trainY);

          Assert.Equal(30, model.Train.TrainDataInputs!.Shape[0]);   // train NEDOTČENÝ
          Assert.Equal(4, model.Train.ValidDataInputs!.Shape[0]);
          Assert.Equal(4, model.Train.TestDataInputs!.Shape[0]);
      }

      [Fact]
      public void Missing_valid_is_carved_from_train_not_from_test()
      {
          // Typicky MNIST: dodán train a test, validační sada chybí.
          // Testovací sada je finální nezaujatý odhad → z ní se nekrájí NIKDY.
          // Validační se v praxi bere z trainu, takže se ukrojí odtamtud.
          var model = new My_DNN.MDNN(new Dense(1, new Linear()), new SGD(0.01), new MSE());

          var (trainX, trainY) = MakeSamples(30);
          var (testX, testY) = MakeSamples(10);

          model.Train.SetDatasets(new LabeledData(trainX, trainY), valid: null, test: new LabeledData(testX, testY));
          model.Train.DividingDataIntoDatasets(trainX, trainY);

          Assert.Equal(10, model.Train.TestDataInputs!.Shape[0]);    // test NEDOTČENÝ
          Assert.Equal(24, model.Train.TrainDataInputs!.Shape[0]);   // 30 × 0.7/(0.7+0.15)
          Assert.Equal(6, model.Train.ValidDataInputs!.Shape[0]);    // zbytek
      }

      [Fact]
      public void Test_is_not_carved_when_final_testing_is_off()
      {
          // Ukrojit test z validu má smysl jen když se po tréninku opravdu testuje.
          // Jinak by uživatel přišel o kus validační sady kvůli datům, co nikdo nepoužije.
          // Řídí to TestNeuralNetworkAfterTraining — jedno rozhodnutí, ne dvě.
          var model = new My_DNN.MDNN(new Dense(1, new Linear()), new SGD(0.01), new MSE());
          model.Train.TestNeuralNetworkAfterTraining = false;

          var (trainX, trainY) = MakeSamples(30);
          var (validX, validY) = MakeSamples(8);

          model.Train.SetDatasets(new LabeledData(trainX, trainY), new LabeledData(validX, validY));
          model.Train.DividingDataIntoDatasets(trainX, trainY);

          Assert.Equal(30, model.Train.TrainDataInputs!.Shape[0]);
          Assert.Equal(8, model.Train.ValidDataInputs!.Shape[0]);   // celá, neukrojená
          Assert.Null(model.Train.TestDataInputs);
      }

      [Fact]
      public void Explicit_test_survives_final_testing_being_off()
      {
          // Když test dodáš sám, vypnutý přehled ho nesmí zahodit — jen se nevypíše.
          var model = new My_DNN.MDNN(new Dense(1, new Linear()), new SGD(0.01), new MSE());
          model.Train.TestNeuralNetworkAfterTraining = false;

          var (trainX, trainY) = MakeSamples(30);
          var (validX, validY) = MakeSamples(8);
          var (testX, testY) = MakeSamples(5);

          model.Train.SetDatasets(
              new LabeledData(trainX, trainY),
              new LabeledData(validX, validY),
              new LabeledData(testX, testY));
          model.Train.DividingDataIntoDatasets(trainX, trainY);

          Assert.Equal(8, model.Train.ValidDataInputs!.Shape[0]);
          Assert.Equal(5, model.Train.TestDataInputs!.Shape[0]);
      }

      [Fact]
      public void SetDatasets_with_train_only_splits_everything()
      {
          // Když nedodáš ani valid ani test, není z čeho jiného krájet než z trainu —
          // dostaneš klasické rozdělení podle poměrů — dřív na to byla samostatná metoda.
          var model = new My_DNN.MDNN(new Dense(1, new Linear()), new SGD(0.01), new MSE());
          var (x, y) = MakeSamples(100);

          model.Train.SetDatasets(new LabeledData(x, y));
          model.Train.DividingDataIntoDatasets(x, y);

          Assert.Equal(70, model.Train.TrainDataInputs!.Shape[0]);
          Assert.Equal(15, model.Train.ValidDataInputs!.Shape[0]);
          Assert.Equal(15, model.Train.TestDataInputs!.Shape[0]);
      }

      [Fact]
      public void Carving_needs_at_least_two_samples()
      {
          var model = new My_DNN.MDNN(new Dense(1, new Linear()), new SGD(0.01), new MSE());
          var (trainX, trainY) = MakeSamples(30);
          var (validX, validY) = MakeSamples(1);

          model.Train.SetDatasets(new LabeledData(trainX, trainY), new LabeledData(validX, validY));

          var ex = Assert.Throws<ArgumentException>(() => model.Train.DividingDataIntoDatasets(trainX, trainY));
          Assert.Contains("TestNeuralNetworkAfterTraining", ex.Message);
      }

      [Fact]
      public void TrainLoop_with_data_is_a_shortcut_for_split_plus_loop()
      {
          // TrainLoop(X, Y, ...) musí dělat PŘESNĚ totéž co SetDatasets(all) + TrainLoop(...)
          var (X, Y) = MakeArrays(40);
          var probe = new Tensor(new double[] { 0.3 });

          var shortcut = Quiet(seed: 3);
          shortcut.Train.TrainLoop(X, Y, 4, 2);

          var explicitForm = Quiet(seed: 3);
          explicitForm.Train.SetDatasets(new LabeledData(X, Y));
          explicitForm.Train.TrainLoop(4, 2);

          Assert.Equal(shortcut.GetResults(probe).Data[0], explicitForm.GetResults(probe).Data[0]);
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

      private static (double[][] X, double[][] Y) MakeArrays(int n)
      {
          var xs = new double[n][];
          var ys = new double[n][];
          for (int i = 0; i < n; i++) { xs[i] = new double[] { i * 0.03 }; ys[i] = new double[] { i * 0.05 }; }
          return (xs, ys);
      }

      [Fact]
      public void LabeledData_rejects_mismatched_sample_counts()
      {
          // Nekonzistentní dvojice teď nejde ani vyrobit — dřív šlo nastavit vstupy
          // bez cílů a spadlo to až u prvního valid reportu.
          var (x, _) = MakeSamples(10);
          var (_, y) = MakeSamples(7);

          var ex = Assert.Throws<ArgumentException>(() => new LabeledData(x, y));
          Assert.Contains("10", ex.Message);
          Assert.Contains("7", ex.Message);
      }

      [Fact]
      public void TrainLoop_without_data_requires_SetDatasets()
      {
          var model = new My_DNN.MDNN(new Dense(1, new Linear()), new SGD(0.01), new MSE());

          var ex = Assert.Throws<InvalidOperationException>(() => model.Train.TrainLoop(5));
          Assert.Contains("SetDatasets", ex.Message);
      }

      [Fact]
      public void Default_split_ratios_are_70_15_15()
      {
          // beze změny konfigurace = dosavadní default (100 vzorků → 70/15/15)
          var model = new My_DNN.MDNN(new Dense(1, new Linear()), new SGD(0.01), new MSE());
          var (x, y) = MakeSamples(100);

          model.Train.DividingDataIntoDatasets(x, y);

          Assert.Equal(15, model.Train.ValidDataInputs!.Shape[0]);
          Assert.Equal(15, model.Train.TestDataInputs!.Shape[0]);   // train = zbytek = 70
      }

      [Fact]
      public void Custom_split_ratios_respected_and_remainder_auto_computed()
      {
          // Train=0.6, Valid=0.2, Test vynechaný → dopočítá se 0.2 (zbytek).
          var model = new My_DNN.MDNN(new Dense(1, new Linear()), new SGD(0.01), new MSE());
          model.Train.TrainSplitRatio = 0.6;
          model.Train.ValidSplitRatio = 0.2;
          // TestSplitRatio zůstává null

          var (x, y) = MakeSamples(100);
          model.Train.DividingDataIntoDatasets(x, y);

          Assert.Equal(20, model.Train.ValidDataInputs!.Shape[0]);
          Assert.Equal(20, model.Train.TestDataInputs!.Shape[0]);   // train = zbytek = 60
      }

      private static (Tensor x, Tensor y) MakeSamples(int n)
      {
          var xs = new double[n, 1];
          var ys = new double[n, 1];
          for (int i = 0; i < n; i++) { xs[i, 0] = i; ys[i, 0] = i; }
          return (new Tensor(xs), new Tensor(ys));
      }
  }

  public class DatasetSplitLeakTests : GradientCheckTestBase
  {
      // 2c-14: druhý TrainLoop na stejném modelu viděl valid/test z prvního běhu, spadl do
      // větve „uživatel dodal valid i test" a nastavil _trainDataInputs = CELÝ dataset —
      // takže druhý běh trénoval i na testovacích datech. Tiše.
      private static (double[][] X, double[][] Y) Data(int n)
      {
          var X = new double[n][];
          var Y = new double[n][];
          for (int i = 0; i < n; i++)
          {
              X[i] = new double[] { i * 0.05 };
              Y[i] = new double[] { i * 0.10 };
          }
          return (X, Y);
      }

      private static My_DNN.MDNN QuietModel()
      {
          var model = new My_DNN.MDNN(new Dense(1, new Linear()), new SGD(0.01), new MSE());
          model.Train.ShowLossChartInTrainLoop = false;
          model.Train.ShowModelInfoIntrainLoop = false;
          model.Train.AutoSaveInTrainLoop = false;
          model.Train.TestNeuralNetworkAfterTraining = false;
          return model;
      }

      [Fact]
      public void Second_train_loop_splits_dataset_the_same_way()
      {
          var (X, Y) = Data(20);
          var model = QuietModel();

          model.Train.TrainLoop(X, Y, 5);
          int train1 = model.Train.TrainDataInputs!.Shape[0];
          int valid1 = model.Train.ValidDataInputs!.Shape[0];
          int test1  = model.Train.TestDataInputs!.Shape[0];

          model.Train.TrainLoop(X, Y, 5);
          int train2 = model.Train.TrainDataInputs!.Shape[0];
          int valid2 = model.Train.ValidDataInputs!.Shape[0];
          int test2  = model.Train.TestDataInputs!.Shape[0];

          Assert.Equal(14, train1);                       // 0.7 z 20
          Assert.Equal((train1, valid1, test1), (train2, valid2, test2));
          Assert.Equal(20, train2 + valid2 + test2);      // nic se neztratilo ani nezdvojilo
      }

      // POZNÁMKA: tenhle test bug 2c-14 NECHYTÁ (ověřeno — bez fixu procházel). Když uživatel
      // dodá vlastní valid set, vyjdou obě cesty shodně: nefixnutá spadne do větve „mám valid
      // i test" a nechá je být, fixnutá je přeřízne znovu ze stejného originálu. Drží se tu
      // jako popis zamýšlené sémantiky (uživatelův vstup se nesmí s každým během dál
      // ukrajovat), ne jako důkaz opravy — tím je test výš, kde train set ujel 14 → 20.
      [Fact]
      public void User_supplied_datasets_survive_repeated_runs()
      {
          var (X, Y) = Data(20);
          var model = QuietModel();

          // uživatel dodá vlastní datasety → musí zůstat nedotčené i po opakovaných bězích
          var trainData = new LabeledData(
              new Tensor(X.SelectMany(r => r).ToArray(), new int[] { X.Length, 1 }),
              new Tensor(Y.SelectMany(r => r).ToArray(), new int[] { Y.Length, 1 }));
          var validData = new LabeledData(
              new Tensor(new double[] { 1, 2, 3, 4 }, new int[] { 4, 1 }),
              new Tensor(new double[] { 2, 4, 6, 8 }, new int[] { 4, 1 }));

          model.Train.SetDatasets(trainData, validData);

          // ať se test z validu opravdu ukrojí (jinak by nebylo co hlídat na stabilitu)
          model.Train.TestNeuralNetworkAfterTraining = true;

          model.Train.TrainLoop(5);
          int train1 = model.Train.TrainDataInputs!.Shape[0];
          int valid1 = model.Train.ValidDataInputs!.Shape[0];
          int test1 = model.Train.TestDataInputs!.Shape[0];

          model.Train.TrainLoop(5);

          // opakovaný běh musí dát stejné rozdělení — krájí se vždycky z PŮVODNÍ zadané
          // sady, ne z té už zmenšené (tam byl kdysi "Invalid slice range!")
          Assert.Equal(train1, model.Train.TrainDataInputs!.Shape[0]);
          Assert.Equal(valid1, model.Train.ValidDataInputs!.Shape[0]);
          Assert.Equal(test1, model.Train.TestDataInputs!.Shape[0]);
          Assert.Equal(20, train1);                       // celý dodaný train set, nedotčený
          Assert.Equal(4, valid1 + test1);                // jeho 4 valid vzorky, rozdělené 2/2
      }
  }

  public class TensorInputTypeTests : GradientCheckTestBase
  {
      // 2c-5: Tensor.ConvertArrayToTensor porovnává elementType.Name s "Int"/"Float"/"Double".
      // Reálná .NET jména jsou Int32/Single/Double → projde jen double, ostatní vyhodí
      // výjimku, jejíž vlastní text int i float slibuje.
      [Fact]
      public void ConvertArrayToTensor_accepts_double_jagged()
          => Assert.Equal(new double[] { 1, 2 },
                          Tensor.ConvertArrayToTensor(new double[][] { new double[] { 1, 2 } }).Data);

      [Fact]
      public void ConvertArrayToTensor_accepts_int_jagged()
          => Assert.Equal(new double[] { 1, 2 },
                          Tensor.ConvertArrayToTensor(new int[][] { new int[] { 1, 2 } }).Data);

      [Fact]
      public void ConvertArrayToTensor_accepts_float_jagged()
          => Assert.Equal(new double[] { 1, 2 },
                          Tensor.ConvertArrayToTensor(new float[][] { new float[] { 1f, 2f } }).Data);

      // Nález z procházení příkladů: rozbalovala se jen JEDNA úroveň jagged pole, takže
      // double[][] prošlo, ale double[][][] skončilo na "invalid type on input array".
      // A to je přesně tvar sekvenčních dat ([sekvence][krok][rys]).
      [Fact]
      public void ConvertArrayToTensor_accepts_triple_jagged()
      {
          var sequences = new double[][][]
          {
              new double[][] { new double[] { 1, 2 }, new double[] { 3, 4 } },
              new double[][] { new double[] { 5, 6 }, new double[] { 7, 8 } },
          };

          Tensor tensor = Tensor.ConvertArrayToTensor(sequences);

          Assert.Equal(new int[] { 2, 2, 2 }, tensor.Shape);
          Assert.Equal(new double[] { 1, 2, 3, 4, 5, 6, 7, 8 }, tensor.Data);
      }

      [Fact]
      public void ConvertArrayToTensor_accepts_int_multidim()
          => Assert.Equal(new double[] { 1, 2 },
                          Tensor.ConvertArrayToTensor(new int[,] { { 1, 2 } }).Data);

      // Zubaté (nepravidelné) pole nemá tvar, takže z něj tenzor udělat NELZE.
      // Dřív to dopadlo pro každý typ jinak a ani jednou dobře: `double[][]` spadlo
      // na IndexOutOfRange z hloubi kopírování, kdežto `int[][]` a `float[][]` prošly
      // a vyrobily tvar podle DÉLKY PRVNÍHO ŘÁDKU — data se tím tiše zamíchala.
      //
      //   int[][] { {1,2}, {3,4,5}, {6} }  ->  tvar [3,2], Data.Length = 6
      //
      // Šest prvků se do 3x2 vejde, takže ani délka nic neprozradí.
      [Theory]
      [InlineData("double")]
      [InlineData("int")]
      [InlineData("float")]
      public void ConvertArrayToTensor_rejects_ragged_arrays(string type)
      {
          Array ragged = type switch
          {
              "double" => new double[][] { new double[] { 1, 2 }, new double[] { 3, 4, 5 }, new double[] { 6 } },
              "int" => new int[][] { new int[] { 1, 2 }, new int[] { 3, 4, 5 }, new int[] { 6 } },
              _ => new float[][] { new float[] { 1, 2 }, new float[] { 3, 4, 5 }, new float[] { 6 } },
          };

          ArgumentException ex = Assert.Throws<ArgumentException>(
              () => Tensor.ConvertArrayToTensor(ragged));

          Assert.Contains("same length", ex.Message);
      }

      [Fact]
      public void ConvertArrayToTensor_accepts_deep_ragged_free_arrays()
      {
          // Kontrola, že se stráž nechytá na pravidelných vnořených polích.
          double[][][] cube = new double[2][][];
          for (int i = 0; i < 2; i++)
          {
              cube[i] = new double[3][];
              for (int j = 0; j < 3; j++) cube[i][j] = new double[] { i, j, i + j, i * j };
          }

          Tensor tensor = Tensor.ConvertArrayToTensor(cube);

          Assert.Equal(new[] { 2, 3, 4 }, tensor.Shape);
          Assert.Equal(24, tensor.Data.Length);
      }
  }
