using System;
  using Xunit;
  using MDNN;
  using MDNN.Layers;
  using MDNN.Activations;
  using MDNN.Optimizers;
  using MDNN.Losses;
  using System.Linq;

  namespace MDNN.Tests;

  public class SerializationTests : GradientCheckTestBase
  {
      [Fact]
      public void LoadWeightsFromString_restores_weights_in_place()
      {
          // In-memory snapshot: serialize to a string + restore back INTO THE SAME instance.
          // Checks that the snapshot captures the weights faithfully, restoring brings them back and
          // keeps the identity of the object and its Train (basis for "always keep the best model" without disk).
          var model = new NeuralNetwork(new Dense(2, new Linear()), new SGD(0.1), new MSE());
          model.Layers.Add(new Dense(3, new Tanh()));

          var x = new Tensor(new double[] { 0.5, -0.3, 0.8, 0.1 });   // 4 inputs
          double[] outBefore = model.GetResults(x).Data;              // triggers wiring + remembers the output

          string snapshot = model.SaveAsJsonString();                // snapshot of the best state
          var trainBefore = model.Train;                             // Train identity (holds the datasets)

          // change the weights by training → the output must move
          for (int i = 0; i < 20; i++)
          {
              model.Train.Fit(x, new Tensor(new double[] { 1.0, -1.0 }));
              model.Train.UpdateParams();
          }
          double[] outMutated = model.GetResults(x).Data;
          Assert.NotEqual(outBefore, outMutated);                   // training really changed the weights

          // restore from the snapshot → in place
          model.LoadWeightsFromString(snapshot);

          double[] outRestored = model.GetResults(x).Data;
          Assert.Equal(outBefore, outRestored);                     // weights faithfully restored (bit-identical forward)
          Assert.Same(trainBefore, model.Train);                    // identity kept (in place, not a new object)
      }
  }

  public class SaveLoadRoundTripTests : GradientCheckTestBase
  {
      // Save/load had no round-trip test, so nobody noticed ExportCNNLayer using
      // Activation.ToString() instead of .Name → after loading, ReLU silently became Linear.
      private static string TempModelBase()
          => System.IO.Path.Combine(System.IO.Path.GetTempPath(), "mdnn_rt_" + Guid.NewGuid().ToString("N"));

      private static void Cleanup(string basePath)
      {
          try { System.IO.File.Delete(basePath + ".json"); } catch { /* cleanup must not fail the test */ }
      }

      [Fact]
      public void Dense_roundtrip_preserves_activation_and_forward()
      {
          string path = TempModelBase();
          try
          {
              var model = new NeuralNetwork(new Dense(2, new Linear()), new SGD(0.01), new MSE());
              model.Layers.Add(new Dense(4, new ReLU()));
              var x = new Tensor(new double[] { 0.5, -0.3, 0.8 });
              double[] before = model.GetResults(x).Data;

              model.SaveAsJson(path);
              var loaded = NeuralNetwork.LoadModel(path + ".json");

              Assert.Equal("ReLU", loaded.Layers.Layers[0].Activation.Name);
              Assert.Equal(before, loaded.GetResults(x).Data);
          }
          finally { Cleanup(path); }
      }

      [Fact]
      public void Conv_roundtrip_preserves_activation_and_forward()
      {
          string path = TempModelBase();
          try
          {
              var conv = new Conv(2, 2, new ReLU(), "valid");
              var model = new NeuralNetwork(conv, new SGD(0.01), new MSE());
              model.Layers.SetInputShape(new int[] { 4, 4, 1 });

              var input = new double[4, 4, 1];
              for (int i = 0; i < 4; i++)
              for (int j = 0; j < 4; j++)
                  input[i, j, 0] = ((i * 4 + j) - 7) * 0.2;      // negative and positive → ReLU ≠ Linear
              var x = new Tensor(input);
              double[] before = model.GetResults(x).Data;

              model.SaveAsJson(path);
              var loaded = NeuralNetwork.LoadModel(path + ".json");

              Assert.Equal("ReLU", loaded.Layers.Layers[0].Activation.Name);
              Assert.Equal(before, loaded.GetResults(x).Data);
          }
          finally { Cleanup(path); }
      }

      [Fact]
      public void RNN_roundtrip_preserves_activation_and_forward()
      {
          string path = TempModelBase();
          try
          {
              var model = new NeuralNetwork(new RNN(3, new Tanh()), new SGD(0.01), new MSE());
              model.Layers.SetInputShape(new int[] { 2 });

              var x = new Tensor(new double[] { 0.5, -0.3 });
              model.ResetSequence();
              double[] before = model.GetResults(x).Data;

              model.SaveAsJson(path);
              var loaded = NeuralNetwork.LoadModel(path + ".json");

              Assert.Equal("Tanh", loaded.Layers.Layers[0].Activation.Name);
              loaded.ResetSequence();                            // same initial hidden state
              model.ResetSequence();
              Assert.Equal(model.GetResults(x).Data, loaded.GetResults(x).Data);
          }
          finally { Cleanup(path); }
      }
  }

  public class SaveLoadPathTests : GradientCheckTestBase
  {
      // Previously Save always appended ".json" and Load did not → SaveAsJson("model") wrote
      // "model.json", but LoadModel("model") failed. And SaveAsJson("model.json")
      // produced "model.json.json".
      private static string TempBase()
          => System.IO.Path.Combine(System.IO.Path.GetTempPath(), "mdnn_path_" + Guid.NewGuid().ToString("N"));

      private static NeuralNetwork SmallModel()
      {
          var model = new NeuralNetwork(new Dense(2, new Linear()), new SGD(0.01), new MSE());
          model.GetResults(new Tensor(new double[] { 0.5, -0.3 }));
          return model;
      }

      [Fact]
      public void LoadModel_finds_file_saved_without_extension()
      {
          string basePath = TempBase();
          try
          {
              SmallModel().SaveAsJson(basePath);                       // writes basePath.json
              Assert.True(System.IO.File.Exists(basePath + ".json"));

              var loaded = NeuralNetwork.LoadModel(basePath);            // WITHOUT the extension
              Assert.NotNull(loaded);
          }
          finally { try { System.IO.File.Delete(basePath + ".json"); } catch { } }
      }

      [Fact]
      public void SaveAsJson_does_not_double_the_extension()
      {
          string path = TempBase() + ".json";
          try
          {
              SmallModel().SaveAsJson(path);

              Assert.True(System.IO.File.Exists(path), "expected a file with a single extension");
              Assert.False(System.IO.File.Exists(path + ".json"), "a double .json.json extension was created");

              Assert.NotNull(NeuralNetwork.LoadModel(path));
          }
          finally { try { System.IO.File.Delete(path); System.IO.File.Delete(path + ".json"); } catch { } }
      }

      [Fact]
      public void LoadModel_missing_file_throws_clear_error()
      {
          string missing = TempBase();
          var ex = Record.Exception(() => NeuralNetwork.LoadModel(missing));

          Assert.IsType<System.IO.FileNotFoundException>(ex);
          Assert.Contains(".json", ex.Message);      // the message shows both paths that were tried
      }
  }

  // ==========================================================================================
  //  Saved model format v1 (FormatVersion, seed, timestamp, checksum)
  // ==========================================================================================

  public class SaveFormatTests : GradientCheckTestBase
  {
      private static string TempBase()
          => System.IO.Path.Combine(System.IO.Path.GetTempPath(), "mdnn_fmt_" + Guid.NewGuid().ToString("N"));

      private static NeuralNetwork Model(int? seed = null)
      {
          var m = new NeuralNetwork(new Dense(2, new Linear()), new SGD(0.01), new MSE(), seed);
          m.Layers.Add(new Dense(3, new ReLU()));
          m.GetResults(new Tensor(new double[] { 0.5, -0.3 }));
          return m;
      }

      [Fact]
      public void Saved_file_carries_version_checksum_seed_and_timestamp()
      {
          string path = TempBase();
          try
          {
              DateTime before = DateTime.UtcNow.AddSeconds(-1);
              Model(seed: 4242).SaveAsJson(path);

              using var doc = System.Text.Json.JsonDocument.Parse(System.IO.File.ReadAllText(path + ".json"));
              var root = doc.RootElement;

              Assert.Equal(1, root.GetProperty("FormatVersion").GetInt32());
              Assert.StartsWith("sha256:", root.GetProperty("Checksum").GetString());

              var model = root.GetProperty("Model");
              Assert.Equal(4242, model.GetProperty("Seed").GetInt32());

              DateTime savedAt = model.GetProperty("SavedAtUtc").GetDateTime();
              Assert.InRange(savedAt, before, DateTime.UtcNow.AddSeconds(1));
          }
          finally { try { System.IO.File.Delete(path + ".json"); } catch { } }
      }

      [Fact]
      public void Seed_is_restored_from_file_when_caller_gives_none()
      {
          string path = TempBase();
          try
          {
              Model(seed: 777).SaveAsJson(path);

              var loaded = NeuralNetwork.LoadModel(path);
              Assert.Equal(777, loaded.Context.Seed);          // provenance survives saving
          }
          finally { try { System.IO.File.Delete(path + ".json"); } catch { } }
      }

      [Fact]
      public void Explicit_seed_wins_over_the_one_in_file()
      {
          string path = TempBase();
          try
          {
              Model(seed: 777).SaveAsJson(path);

              var loaded = NeuralNetwork.LoadModel(path, seed: 111);
              Assert.Equal(111, loaded.Context.Seed);
          }
          finally { try { System.IO.File.Delete(path + ".json"); } catch { } }
      }

      [Fact]
      public void Round_trip_still_reproduces_the_forward_pass()
      {
          // the envelope must not break the essential part: the loaded model computes the same
          string path = TempBase();
          try
          {
              var src = Model(seed: 5);
              var x = new Tensor(new double[] { 0.5, -0.3 });
              double[] before = src.GetResults(x).Data;

              src.SaveAsJson(path);
              Assert.Equal(before, NeuralNetwork.LoadModel(path).GetResults(x).Data);
          }
          finally { try { System.IO.File.Delete(path + ".json"); } catch { } }
      }

      // Deliberately NOT "Tampered": the checksum does not catch deliberate tampering (whoever
      // edits the file recomputes the hash too) and must not pretend to. It catches UNINTENDED
      // changes: a typo during manual editing, an accidentally overwritten value, a corrupted transfer.
      [Fact]
      public void Accidentally_modified_file_is_rejected()
      {
          string path = TempBase();
          try
          {
              Model(seed: 1).SaveAsJson(path);

              // change one weight in the data, leave the checksum → must not pass
              string json = System.IO.File.ReadAllText(path + ".json");
              string broken = json.Replace("\"Bias\": 0", "\"Bias\": 0.123456");
              Assert.NotEqual(json, broken);                   // sanity check that the replacement worked
              System.IO.File.WriteAllText(path + ".json", broken);

              var ex = Assert.Throws<ModelFileCorruptedException>(() => NeuralNetwork.LoadModel(path));
              Assert.StartsWith("sha256:", ex.ExpectedChecksum);
              Assert.NotEqual(ex.ExpectedChecksum, ex.ActualChecksum);
          }
          finally { try { System.IO.File.Delete(path + ".json"); } catch { } }
      }

      [Fact]
      public void Truncated_file_is_rejected()
      {
          string path = TempBase();
          try
          {
              Model().SaveAsJson(path);
              string json = System.IO.File.ReadAllText(path + ".json");
              System.IO.File.WriteAllText(path + ".json", json.Substring(0, json.Length / 2));

              Assert.Throws<ModelFileCorruptedException>(() => NeuralNetwork.LoadModel(path));
          }
          finally { try { System.IO.File.Delete(path + ".json"); } catch { } }
      }

      [Fact]
      public void Future_format_version_is_rejected_with_clear_message()
      {
          string path = TempBase();
          try
          {
              Model().SaveAsJson(path);
              string json = System.IO.File.ReadAllText(path + ".json");
              System.IO.File.WriteAllText(path + ".json", json.Replace("\"FormatVersion\": 1", "\"FormatVersion\": 99"));

              var ex = Assert.Throws<ModelFileCorruptedException>(() => NeuralNetwork.LoadModel(path));
              Assert.Contains("Update the library", ex.Message);
          }
          finally { try { System.IO.File.Delete(path + ".json"); } catch { } }
      }

      [Fact]
      public void Legacy_file_without_envelope_still_loads()
      {
          // backward compatibility: models saved by an old library version (flat JSON,
          // no FormatVersion or checksum) must still load
          string path = TempBase();
          try
          {
              Model(seed: 3).SaveAsJson(path);

              // unwrap the envelope → exactly the old format
              string json = System.IO.File.ReadAllText(path + ".json");
              using var doc = System.Text.Json.JsonDocument.Parse(json);
              System.IO.File.WriteAllText(path + ".json", doc.RootElement.GetProperty("Model").GetRawText());

              var loaded = NeuralNetwork.LoadModel(path);
              Assert.NotNull(loaded);
              Assert.Equal(2, loaded.GetResults(new Tensor(new double[] { 0.5, -0.3 })).Data.Length);
          }
          finally { try { System.IO.File.Delete(path + ".json"); } catch { } }
      }

      [Fact]
      public void In_memory_snapshot_round_trips_through_the_same_path()
      {
          // SaveAsJsonString/LoadWeightsFromString use the same envelope as the disk path
          var model = Model(seed: 9);
          var x = new Tensor(new double[] { 0.5, -0.3 });
          double[] before = model.GetResults(x).Data;

          string snapshot = model.SaveAsJsonString();
          Assert.Contains("FormatVersion", snapshot);

          model.LoadWeightsFromString(snapshot);
          Assert.Equal(before, model.GetResults(x).Data);
      }
  }

  public class ExtendingLoadedModelTests : GradientCheckTestBase
  {
      // Crash found 2026-08-11: LoadModel → Add → GetResults ended in IndexOutOfRange.
      // LayerManager.Add() rebuilt only the LAST layer; the inserted one kept placeholder
      // neurons (0 weights) and relied on SetInputShape, which only ran under the
      // condition "Layers[0].InputShape[0] <= 0". In a loaded model the first layer has
      // its input shape from the file, so the condition was false and the new layer stayed empty.
      //
      // Loading a checkpoint and extending the architecture is a basic operation (e.g. for an AutoML runner).
      private static string TempBase()
          => System.IO.Path.Combine(System.IO.Path.GetTempPath(), "mdnn_ext_" + Guid.NewGuid().ToString("N"));

      private static void Save(string path, int? seed = null)
      {
          var m = new NeuralNetwork(new Dense(2, new Linear()), new SGD(0.01), new MSE(), seed);
          m.Layers.Add(new Dense(3, new ReLU()));
          m.GetResults(new Tensor(new double[] { 1, 1 }));
          m.SaveAsJson(path);
      }

      [Fact]
      public void Adding_layer_to_loaded_model_does_not_crash()
      {
          string path = TempBase();
          try
          {
              Save(path);

              var loaded = NeuralNetwork.LoadModel(path);
              loaded.Layers.Add(new Dense(4, new ReLU()));

              double[] output = loaded.GetResults(new Tensor(new double[] { 1, 1 })).Data;
              Assert.Equal(2, output.Length);
              Assert.All(output, v => Assert.True(double.IsFinite(v)));
          }
          finally { try { System.IO.File.Delete(path + ".json"); } catch { } }
      }

      [Fact]
      public void Loaded_model_can_be_trained_after_extension()
      {
          // not only must it not crash, the extended model must also keep learning
          string path = TempBase();
          try
          {
              Save(path);

              var loaded = NeuralNetwork.LoadModel(path);
              loaded.Layers.Add(new Dense(4, new ReLU()));

              var x = new Tensor(new double[] { 1, 1 });
              var t = new Tensor(new double[] { 0.5, -0.5 });

              double lossBefore = loaded.Loss.CalculateAndGetLoss(loaded.GetResults(x).Data, t.Data);
              for (int i = 0; i < 200; i++)
              {
                  loaded.Train.Fit(x, t);
                  loaded.Train.UpdateParams();
              }
              double lossAfter = loaded.Loss.CalculateAndGetLoss(loaded.GetResults(x).Data, t.Data);

              Assert.True(lossAfter < lossBefore, $"loss {lossBefore} → {lossAfter}, the model is not learning");
          }
          finally { try { System.IO.File.Delete(path + ".json"); } catch { } }
      }

      [Fact]
      public void Extending_loaded_model_keeps_weights_of_untouched_layers()
      {
          // layers whose shape does not change must keep their trained weights
          string path = TempBase();
          try
          {
              Save(path);

              var loaded = NeuralNetwork.LoadModel(path);
              double[] hiddenBefore = (double[])((Dense)loaded.Layers.Layers[0]).Neurons[0].Weights.Clone();

              loaded.Layers.Add(new Dense(4, new ReLU()));
              loaded.GetResults(new Tensor(new double[] { 1, 1 }));

              // the first hidden layer still receives the model input → unchanged
              Assert.Equal(hiddenBefore, ((Dense)loaded.Layers.Layers[0]).Neurons[0].Weights);
          }
          finally { try { System.IO.File.Delete(path + ".json"); } catch { } }
      }

      [Fact]
      public void Removing_layer_from_loaded_model_rewires_the_rest()
      {
          string path = TempBase();
          try
          {
              Save(path);

              var loaded = NeuralNetwork.LoadModel(path);
              Assert.Equal(2, loaded.Layers.Layers.Count);

              loaded.Layers.RemoveAt(0);                       // remove the hidden layer

              double[] output = loaded.GetResults(new Tensor(new double[] { 1, 1 })).Data;
              Assert.Equal(2, output.Length);
              Assert.All(output, v => Assert.True(double.IsFinite(v)));
          }
          finally { try { System.IO.File.Delete(path + ".json"); } catch { } }
      }
  }

  public class ModelInfoTests : GradientCheckTestBase
  {
      [Fact]
      public void CountTrainableParams_includes_all_conv_filters()
      {
          // Same CNN as in mdnn_test: Conv16 → MaxPool → Conv32 → MaxPool → Dense64 → Dense10.
          // Regression: conv used to be undercounted (missing × filter count) → 52115 instead of 56714.
          var model = new NeuralNetwork(new Dense(10, new Softmax()), new Adam(0.001), new CrossEntropy());
          model.Layers.Add(new Conv(16, 3, new ReLU(), "valid"));   // 28×28×1 → 26×26×16
          model.Layers.Add(new MaxPool(2));                         // → 13×13×16
          model.Layers.Add(new Conv(32, 3, new ReLU(), "valid"));   // → 11×11×32
          model.Layers.Add(new MaxPool(2));                         // → 5×5×32
          model.Layers.Add(new Dense(64, new ReLU()));             // flatten 800 → 64

          model.GetResults(new Tensor(new double[784]));           // triggers wiring (sets shapes/channels)

          // Conv1:  16·(3·3·1)+16  = 160
          // Conv2:  32·(3·3·16)+32 = 4640
          // Dense64: 64·800+64     = 51264
          // Dense10: 10·64+10      = 650
          Assert.Equal(160 + 4640 + 51264 + 650, ConsoleController.CountTrainableParams(model));  // 56714
      }
  }

  public class SchemaTests : GradientCheckTestBase
  {
      // The schema used to be cached on the first Info()/save, and Add/Insert/RemoveAt did not
      // invalidate the cache, so the file then described an architecture the model no longer had.
      [Fact]
      public void Schema_follows_layer_changes_after_info()
      {
          var model = new NeuralNetwork(new Dense(1, new Linear()), new SGD(0.01), new MSE());
          var so = Console.Out;
          Console.SetOut(System.IO.TextWriter.Null);
          try { model.Info(); } finally { Console.SetOut(so); }

          model.Layers.Add(new Dense(4, new ReLU()));

          Assert.Equal("[Dense,Dense]", model.Schema);
          Assert.Contains("[Dense,Dense]", model.SaveAsJsonString());
      }
  }
