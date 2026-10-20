using System;
  using Xunit;
  using MDNN;
  using MDNN.Layers;
  using MDNN.Activations;
  using MDNN.Optimizers;
  using MDNN.Losses;
  using System.Linq;

  namespace MDNN.Tests;

  public class DataSplitTests : GradientCheckTestBase
  {
      // There used to be a test `DividingDataIntoDatasets_userValidNoTest_carves_test_from_original_valid`
      // guarding the branch "user supplied VALID but not TEST → carve test out of valid".
      // That branch was REMOVED together with the setters: datasets are now supplied with a
      // single SetDatasets(train, valid, test) call. The regression it guarded (test carved
      // from an already shrunk valid → "Invalid slice range!") can no longer happen because
      // that code path is gone. Replaced by the tests of the new semantics below.

      [Fact]
      public void Explicit_datasets_are_used_as_given()
      {
          var model = new NeuralNetwork(new Dense(1, new Linear()), new SGD(0.01), new MSE());

          var (trainX, trainY) = MakeSamples(30);
          var (validX, validY) = MakeSamples(8);
          var (testX, testY) = MakeSamples(5);

          model.Train.SetDatasets(
              new LabeledData(trainX, trainY),
              new LabeledData(validX, validY),
              new LabeledData(testX, testY));

          model.Train.SplitDatasets(trainX, trainY);

          // no carving, no ratios — exactly what the user supplied
          Assert.Equal(30, model.Train.TrainInputs!.Shape[0]);
          Assert.Equal(8, model.Train.ValidInputs!.Shape[0]);
          Assert.Equal(5, model.Train.TestInputs!.Shape[0]);
          Assert.True(model.Train.HasExplicitDatasets);
      }

      [Fact]
      public void Missing_test_is_carved_from_valid()
      {
          // A missing set is carved from its SIBLING, not from train. The valid:test ratio
          // defaults to 0.15:0.15, so 8 valid samples give 4 valid / 4 test.
          var model = new NeuralNetwork(new Dense(1, new Linear()), new SGD(0.01), new MSE());

          var (trainX, trainY) = MakeSamples(30);
          var (validX, validY) = MakeSamples(8);

          model.Train.SetDatasets(new LabeledData(trainX, trainY), new LabeledData(validX, validY));
          model.Train.SplitDatasets(trainX, trainY);

          Assert.Equal(30, model.Train.TrainInputs!.Shape[0]);   // train UNTOUCHED
          Assert.Equal(4, model.Train.ValidInputs!.Shape[0]);
          Assert.Equal(4, model.Train.TestInputs!.Shape[0]);
      }

      [Fact]
      public void Missing_valid_is_carved_from_train_not_from_test()
      {
          // Typically MNIST: train and test supplied, validation set missing.
          // The test set is the final unbiased estimate → it is NEVER carved from.
          // In practice validation comes from train, so it is carved from there.
          var model = new NeuralNetwork(new Dense(1, new Linear()), new SGD(0.01), new MSE());

          var (trainX, trainY) = MakeSamples(30);
          var (testX, testY) = MakeSamples(10);

          model.Train.SetDatasets(new LabeledData(trainX, trainY), valid: null, test: new LabeledData(testX, testY));
          model.Train.SplitDatasets(trainX, trainY);

          Assert.Equal(10, model.Train.TestInputs!.Shape[0]);    // test UNTOUCHED
          Assert.Equal(24, model.Train.TrainInputs!.Shape[0]);   // 30 × 0.7/(0.7+0.15)
          Assert.Equal(6, model.Train.ValidInputs!.Shape[0]);    // remainder
      }

      [Fact]
      public void Test_is_not_carved_when_final_testing_is_off()
      {
          // Carving test out of valid only makes sense if testing actually runs after training.
          // Otherwise the user would lose part of the validation set to data nobody uses.
          // TestAfterTraining controls it — one decision, not two.
          var model = new NeuralNetwork(new Dense(1, new Linear()), new SGD(0.01), new MSE());
          model.Train.TestAfterTraining = false;

          var (trainX, trainY) = MakeSamples(30);
          var (validX, validY) = MakeSamples(8);

          model.Train.SetDatasets(new LabeledData(trainX, trainY), new LabeledData(validX, validY));
          model.Train.SplitDatasets(trainX, trainY);

          Assert.Equal(30, model.Train.TrainInputs!.Shape[0]);
          Assert.Equal(8, model.Train.ValidInputs!.Shape[0]);   // whole, not carved
          Assert.Null(model.Train.TestInputs);
      }

      [Fact]
      public void Explicit_test_survives_final_testing_being_off()
      {
          // If you supply test yourself, disabled final testing must not discard it — it just isn't reported.
          var model = new NeuralNetwork(new Dense(1, new Linear()), new SGD(0.01), new MSE());
          model.Train.TestAfterTraining = false;

          var (trainX, trainY) = MakeSamples(30);
          var (validX, validY) = MakeSamples(8);
          var (testX, testY) = MakeSamples(5);

          model.Train.SetDatasets(
              new LabeledData(trainX, trainY),
              new LabeledData(validX, validY),
              new LabeledData(testX, testY));
          model.Train.SplitDatasets(trainX, trainY);

          Assert.Equal(8, model.Train.ValidInputs!.Shape[0]);
          Assert.Equal(5, model.Train.TestInputs!.Shape[0]);
      }

      [Fact]
      public void SetDatasets_with_train_only_splits_everything()
      {
          // With neither valid nor test supplied, there is nothing to carve from but train —
          // you get the classic ratio split (this used to be a separate method).
          var model = new NeuralNetwork(new Dense(1, new Linear()), new SGD(0.01), new MSE());
          var (x, y) = MakeSamples(100);

          model.Train.SetDatasets(new LabeledData(x, y));
          model.Train.SplitDatasets(x, y);

          Assert.Equal(70, model.Train.TrainInputs!.Shape[0]);
          Assert.Equal(15, model.Train.ValidInputs!.Shape[0]);
          Assert.Equal(15, model.Train.TestInputs!.Shape[0]);
      }

      [Fact]
      public void Carving_needs_at_least_two_samples()
      {
          var model = new NeuralNetwork(new Dense(1, new Linear()), new SGD(0.01), new MSE());
          var (trainX, trainY) = MakeSamples(30);
          var (validX, validY) = MakeSamples(1);

          model.Train.SetDatasets(new LabeledData(trainX, trainY), new LabeledData(validX, validY));

          var ex = Assert.Throws<ArgumentException>(() => model.Train.SplitDatasets(trainX, trainY));
          Assert.Contains("TestAfterTraining", ex.Message);
      }

      [Fact]
      public void TrainLoop_with_data_is_a_shortcut_for_split_plus_loop()
      {
          // TrainLoop(X, Y, ...) must do EXACTLY the same as SetDatasets(all) + TrainLoop(...)
          var (X, Y) = MakeArrays(40);
          var probe = new Tensor(new double[] { 0.3 });

          var shortcut = Quiet(seed: 3);
          shortcut.Train.TrainLoop(X, Y, 4, 2);

          var explicitForm = Quiet(seed: 3);
          explicitForm.Train.SetDatasets(new LabeledData(X, Y));
          explicitForm.Train.TrainLoop(4, 2);

          Assert.Equal(shortcut.GetResults(probe).Data[0], explicitForm.GetResults(probe).Data[0]);
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
          // An inconsistent pair cannot even be constructed now — previously you could set
          // inputs without targets and it only failed at the first valid report.
          var (x, _) = MakeSamples(10);
          var (_, y) = MakeSamples(7);

          var ex = Assert.Throws<ArgumentException>(() => new LabeledData(x, y));
          Assert.Contains("10", ex.Message);
          Assert.Contains("7", ex.Message);
      }

      [Fact]
      public void TrainLoop_without_data_requires_SetDatasets()
      {
          var model = new NeuralNetwork(new Dense(1, new Linear()), new SGD(0.01), new MSE());

          var ex = Assert.Throws<InvalidOperationException>(() => model.Train.TrainLoop(5));
          Assert.Contains("SetDatasets", ex.Message);
      }

      [Fact]
      public void Default_split_ratios_are_70_15_15()
      {
          // no configuration change = the default (100 samples → 70/15/15)
          var model = new NeuralNetwork(new Dense(1, new Linear()), new SGD(0.01), new MSE());
          var (x, y) = MakeSamples(100);

          model.Train.SplitDatasets(x, y);

          Assert.Equal(15, model.Train.ValidInputs!.Shape[0]);
          Assert.Equal(15, model.Train.TestInputs!.Shape[0]);   // train = remainder = 70
      }

      [Fact]
      public void Custom_split_ratios_respected_and_remainder_auto_computed()
      {
          // Train=0.6, Valid=0.2, Test omitted → computed as 0.2 (the remainder).
          var model = new NeuralNetwork(new Dense(1, new Linear()), new SGD(0.01), new MSE());
          model.Train.TrainSplitRatio = 0.6;
          model.Train.ValidSplitRatio = 0.2;
          // TestSplitRatio stays null

          var (x, y) = MakeSamples(100);
          model.Train.SplitDatasets(x, y);

          Assert.Equal(20, model.Train.ValidInputs!.Shape[0]);
          Assert.Equal(20, model.Train.TestInputs!.Shape[0]);   // train = remainder = 60
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
      // A second TrainLoop on the same model saw valid/test from the first run, took the
      // "user supplied both valid and test" branch and set _trainDataInputs = the WHOLE dataset —
      // so the second run also trained on the test data. Silently.
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

      private static NeuralNetwork QuietModel()
      {
          var model = new NeuralNetwork(new Dense(1, new Linear()), new SGD(0.01), new MSE());
          model.Train.ShowLossChart = false;
          model.Train.ShowModelInfo = false;
          model.Train.AutoSave = false;
          model.Train.TestAfterTraining = false;
          return model;
      }

      [Fact]
      public void Second_train_loop_splits_dataset_the_same_way()
      {
          var (X, Y) = Data(20);
          var model = QuietModel();

          model.Train.TrainLoop(X, Y, 5);
          int train1 = model.Train.TrainInputs!.Shape[0];
          int valid1 = model.Train.ValidInputs!.Shape[0];
          int test1  = model.Train.TestInputs!.Shape[0];

          model.Train.TrainLoop(X, Y, 5);
          int train2 = model.Train.TrainInputs!.Shape[0];
          int valid2 = model.Train.ValidInputs!.Shape[0];
          int test2  = model.Train.TestInputs!.Shape[0];

          Assert.Equal(14, train1);                       // 0.7 of 20
          Assert.Equal((train1, valid1, test1), (train2, valid2, test2));
          Assert.Equal(20, train2 + valid2 + test2);      // nothing lost or duplicated
      }

      // NOTE: this test does NOT catch the bug above (verified — it passed without the fix).
      // When the user supplies their own valid set, both paths behave the same: the unfixed one
      // takes the "have valid and test" branch and leaves them alone, the fixed one re-carves
      // them from the same original. It is kept as a description of the intended semantics
      // (user input must not keep shrinking with every run), not as proof of the fix — that is
      // the test above, where the train set drifted 14 → 20.
      [Fact]
      public void User_supplied_datasets_survive_repeated_runs()
      {
          var (X, Y) = Data(20);
          var model = QuietModel();

          // the user supplies their own datasets → they must stay untouched across repeated runs
          var trainData = new LabeledData(
              new Tensor(X.SelectMany(r => r).ToArray(), new int[] { X.Length, 1 }),
              new Tensor(Y.SelectMany(r => r).ToArray(), new int[] { Y.Length, 1 }));
          var validData = new LabeledData(
              new Tensor(new double[] { 1, 2, 3, 4 }, new int[] { 4, 1 }),
              new Tensor(new double[] { 2, 4, 6, 8 }, new int[] { 4, 1 }));

          model.Train.SetDatasets(trainData, validData);

          // make sure test really is carved from valid (otherwise there would be no stability to check)
          model.Train.TestAfterTraining = true;

          model.Train.TrainLoop(5);
          int train1 = model.Train.TrainInputs!.Shape[0];
          int valid1 = model.Train.ValidInputs!.Shape[0];
          int test1 = model.Train.TestInputs!.Shape[0];

          model.Train.TrainLoop(5);

          // a repeated run must give the same split — carving always starts from the ORIGINAL
          // supplied set, not the already shrunk one (that once caused "Invalid slice range!")
          Assert.Equal(train1, model.Train.TrainInputs!.Shape[0]);
          Assert.Equal(valid1, model.Train.ValidInputs!.Shape[0]);
          Assert.Equal(test1, model.Train.TestInputs!.Shape[0]);
          Assert.Equal(20, train1);                       // the whole supplied train set, untouched
          Assert.Equal(4, valid1 + test1);                // its 4 valid samples, split 2/2
      }
  }

  public class TensorInputTypeTests : GradientCheckTestBase
  {
      // Tensor.ConvertArrayToTensor used to compare elementType.Name with "Int"/"Float"/"Double".
      // The real .NET names are Int32/Single/Double → only double got through; the others threw
      // an exception whose own message promised int and float support.
      [Fact]
      public void ConvertArrayToTensor_accepts_double_jagged()
          => Assert.Equal(new double[] { 1, 2 },
                          new Tensor(new double[][] { new double[] { 1, 2 } }).Data);

      [Fact]
      public void ConvertArrayToTensor_accepts_int_jagged()
          => Assert.Equal(new double[] { 1, 2 },
                          new Tensor(new int[][] { new int[] { 1, 2 } }).Data);

      [Fact]
      public void ConvertArrayToTensor_accepts_float_jagged()
          => Assert.Equal(new double[] { 1, 2 },
                          new Tensor(new float[][] { new float[] { 1f, 2f } }).Data);

      // Found while going through the examples: only ONE level of a jagged array was unpacked,
      // so double[][] worked but double[][][] failed with "invalid type on input array".
      // And that is exactly the shape of sequence data ([sequence][step][feature]).
      [Fact]
      public void ConvertArrayToTensor_accepts_triple_jagged()
      {
          var sequences = new double[][][]
          {
              new double[][] { new double[] { 1, 2 }, new double[] { 3, 4 } },
              new double[][] { new double[] { 5, 6 }, new double[] { 7, 8 } },
          };

          Tensor tensor = new Tensor(sequences);

          Assert.Equal(new int[] { 2, 2, 2 }, tensor.Shape);
          Assert.Equal(new double[] { 1, 2, 3, 4, 5, 6, 7, 8 }, tensor.Data);
      }

      [Fact]
      public void ConvertArrayToTensor_accepts_int_multidim()
          => Assert.Equal(new double[] { 1, 2 },
                          new Tensor(new int[,] { { 1, 2 } }).Data);

      // A ragged (irregular) array has no shape, so it CANNOT be turned into a tensor.
      // Previously each type failed differently and none correctly: `double[][]` threw
      // IndexOutOfRange from deep inside the copy, while `int[][]` and `float[][]` passed
      // and produced a shape based on the FIRST ROW'S LENGTH — silently scrambling the data.
      //
      //   int[][] { {1,2}, {3,4,5}, {6} }  ->  shape [3,2], Data.Length = 6
      //
      // Six elements fit into 3x2, so not even the length gives it away.
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
              () => new Tensor(ragged));

          Assert.Contains("same length", ex.Message);
      }

      [Fact]
      public void ConvertArrayToTensor_accepts_deep_ragged_free_arrays()
      {
          // Checks that the guard does not trip on regular nested arrays.
          double[][][] cube = new double[2][][];
          for (int i = 0; i < 2; i++)
          {
              cube[i] = new double[3][];
              for (int j = 0; j < 3; j++) cube[i][j] = new double[] { i, j, i + j, i * j };
          }

          Tensor tensor = new Tensor(cube);

          Assert.Equal(new[] { 2, 3, 4 }, tensor.Shape);
          Assert.Equal(24, tensor.Data.Length);
      }
  }
