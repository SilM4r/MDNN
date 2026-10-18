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

  public class SerializationTests : GradientCheckTestBase
  {
      [Fact]
      public void LoadWeightsFromString_restores_weights_in_place()
      {
          // Krok 1 in-memory snapshotu: serializace do stringu + obnova zpět DO STEJNÉ instance.
          // Ověřuje, že snapshot věrně zachytí váhy, obnova je vrátí a zachová identitu objektu
          // i jeho Train (základ pro „vždy si nech nejlepší model" bez nutného disku).
          var model = new My_DNN.MDNN(new Dense(2, new Linear()), new SGD(0.1), new MSE());
          model.Layers.Add(new Dense(3, new Tanh()));

          var x = new Tensor(new double[] { 0.5, -0.3, 0.8, 0.1 });   // 4 vstupy
          double[] outBefore = model.GetResults(x).Data;              // spustí wiring + zapamatuje výstup

          string snapshot = model.SaveAsJsonString();                // snapshot nejlepšího stavu
          var trainBefore = model.Train;                             // identita Train (drží datasety)

          // změň váhy tréninkem → výstup se musí pohnout
          for (int i = 0; i < 20; i++)
          {
              model.Train.Fit(x, new Tensor(new double[] { 1.0, -1.0 }));
              model.Train.UpdateParams();
          }
          double[] outMutated = model.GetResults(x).Data;
          Assert.NotEqual(outBefore, outMutated);                   // trénink váhy opravdu změnil

          // obnova ze snapshotu → in-place
          model.LoadWeightsFromString(snapshot);

          double[] outRestored = model.GetResults(x).Data;
          Assert.Equal(outBefore, outRestored);                     // váhy věrně obnoveny (bit-identický forward)
          Assert.Same(trainBefore, model.Train);                    // identita zachována (in-place, ne nový objekt)
      }
  }

  public class SaveLoadRoundTripTests : GradientCheckTestBase
  {
      // 2c-2: save/load nemá round-trip test, proto neodhalen ExportCNNLayer používající
      // Activation_Func.ToString() místo .Name → po načtení z ReLu tiše Linear.
      private static string TempModelBase()
          => System.IO.Path.Combine(System.IO.Path.GetTempPath(), "mdnn_rt_" + Guid.NewGuid().ToString("N"));

      private static void Cleanup(string basePath)
      {
          try { System.IO.File.Delete(basePath + ".json"); } catch { /* úklid nesmí shodit test */ }
      }

      [Fact]
      public void Dense_roundtrip_preserves_activation_and_forward()
      {
          string path = TempModelBase();
          try
          {
              var model = new My_DNN.MDNN(new Dense(2, new Linear()), new SGD(0.01), new MSE());
              model.Layers.Add(new Dense(4, new ReLu()));
              var x = new Tensor(new double[] { 0.5, -0.3, 0.8 });
              double[] before = model.GetResults(x).Data;

              model.SaveAsJson(path);
              var loaded = My_DNN.MDNN.LoadModel(path + ".json");

              Assert.Equal("ReLu", loaded.Layers.Layers[0].Activation_Func.Name);
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
              var conv = new Conv(2, 2, new ReLu(), "valid");
              var model = new My_DNN.MDNN(conv, new SGD(0.01), new MSE());
              model.Layers.SetInputSizeForFirstLayer(new int[] { 4, 4, 1 });

              var input = new double[4, 4, 1];
              for (int i = 0; i < 4; i++)
              for (int j = 0; j < 4; j++)
                  input[i, j, 0] = ((i * 4 + j) - 7) * 0.2;      // záporné i kladné → ReLu ≠ Linear
              var x = new Tensor(input);
              double[] before = model.GetResults(x).Data;

              model.SaveAsJson(path);
              var loaded = My_DNN.MDNN.LoadModel(path + ".json");

              Assert.Equal("ReLu", loaded.Layers.Layers[0].Activation_Func.Name);
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
              var model = new My_DNN.MDNN(new RNN(3, new Tanh()), new SGD(0.01), new MSE());
              model.Layers.SetInputSizeForFirstLayer(new int[] { 2 });

              var x = new Tensor(new double[] { 0.5, -0.3 });
              model.ResetSequence();
              double[] before = model.GetResults(x).Data;

              model.SaveAsJson(path);
              var loaded = My_DNN.MDNN.LoadModel(path + ".json");

              Assert.Equal("Tanh", loaded.Layers.Layers[0].Activation_Func.Name);
              loaded.ResetSequence();                            // stejný počáteční skrytý stav
              model.ResetSequence();
              Assert.Equal(model.GetResults(x).Data, loaded.GetResults(x).Data);
          }
          finally { Cleanup(path); }
      }
  }

  public class SaveLoadPathTests : GradientCheckTestBase
  {
      // 2c-13: Save vždycky přilepil ".json", Load ne → SaveAsJson("model") zapsalo
      // "model.json", ale LoadModel("model") selhalo. A SaveAsJson("model.json")
      // vyrobilo "model.json.json".
      private static string TempBase()
          => System.IO.Path.Combine(System.IO.Path.GetTempPath(), "mdnn_path_" + Guid.NewGuid().ToString("N"));

      private static My_DNN.MDNN SmallModel()
      {
          var model = new My_DNN.MDNN(new Dense(2, new Linear()), new SGD(0.01), new MSE());
          model.GetResults(new Tensor(new double[] { 0.5, -0.3 }));
          return model;
      }

      [Fact]
      public void LoadModel_finds_file_saved_without_extension()
      {
          string basePath = TempBase();
          try
          {
              SmallModel().SaveAsJson(basePath);                       // zapíše basePath.json
              Assert.True(System.IO.File.Exists(basePath + ".json"));

              var loaded = My_DNN.MDNN.LoadModel(basePath);            // BEZ přípony
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

              Assert.True(System.IO.File.Exists(path), "očekáván soubor s jednou příponou");
              Assert.False(System.IO.File.Exists(path + ".json"), "vznikla dvojitá přípona .json.json");

              Assert.NotNull(My_DNN.MDNN.LoadModel(path));
          }
          finally { try { System.IO.File.Delete(path); System.IO.File.Delete(path + ".json"); } catch { } }
      }

      [Fact]
      public void LoadModel_missing_file_throws_clear_error()
      {
          string missing = TempBase();
          var ex = Record.Exception(() => My_DNN.MDNN.LoadModel(missing));

          Assert.IsType<System.IO.FileNotFoundException>(ex);
          Assert.Contains(".json", ex.Message);      // hláška ukáže obě zkoušené cesty
      }
  }

  // ==========================================================================================
  //  Fáze 4 — formát uloženého modelu v1 (FormatVersion, seed, datum, checksum)
  // ==========================================================================================

  public class SaveFormatTests : GradientCheckTestBase
  {
      private static string TempBase()
          => System.IO.Path.Combine(System.IO.Path.GetTempPath(), "mdnn_fmt_" + Guid.NewGuid().ToString("N"));

      private static My_DNN.MDNN Model(int? seed = null)
      {
          var m = new My_DNN.MDNN(new Dense(2, new Linear()), new SGD(0.01), new MSE(), seed);
          m.Layers.Add(new Dense(3, new ReLu()));
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

              var loaded = My_DNN.MDNN.LoadModel(path);
              Assert.Equal(777, loaded.Context.Seed);          // provenience přežije uložení
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

              var loaded = My_DNN.MDNN.LoadModel(path, seed: 111);
              Assert.Equal(111, loaded.Context.Seed);
          }
          finally { try { System.IO.File.Delete(path + ".json"); } catch { } }
      }

      [Fact]
      public void Round_trip_still_reproduces_the_forward_pass()
      {
          // obálka nesmí rozbít to podstatné — že načtený model počítá stejně
          string path = TempBase();
          try
          {
              var src = Model(seed: 5);
              var x = new Tensor(new double[] { 0.5, -0.3 });
              double[] before = src.GetResults(x).Data;

              src.SaveAsJson(path);
              Assert.Equal(before, My_DNN.MDNN.LoadModel(path).GetResults(x).Data);
          }
          finally { try { System.IO.File.Delete(path + ".json"); } catch { } }
      }

      // Záměrně NE "Tampered" — checksum cílenou manipulaci nechytá (kdo soubor upraví,
      // přepočítá si i hash) a nemá se to tak tvářit. Chytá NEZAMÝŠLENOU změnu: překlep
      // při ruční editaci, omylem přepsanou hodnotu, poškozený přenos.
      [Fact]
      public void Accidentally_modified_file_is_rejected()
      {
          string path = TempBase();
          try
          {
              Model(seed: 1).SaveAsJson(path);

              // změň jednu váhu v datech, checksum nech být → nesmí projít
              string json = System.IO.File.ReadAllText(path + ".json");
              string broken = json.Replace("\"Bias\": 0", "\"Bias\": 0.123456");
              Assert.NotEqual(json, broken);                   // pojistka, že se náhrada povedla
              System.IO.File.WriteAllText(path + ".json", broken);

              var ex = Assert.Throws<ModelFileCorruptedException>(() => My_DNN.MDNN.LoadModel(path));
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

              Assert.Throws<ModelFileCorruptedException>(() => My_DNN.MDNN.LoadModel(path));
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

              var ex = Assert.Throws<ModelFileCorruptedException>(() => My_DNN.MDNN.LoadModel(path));
              Assert.Contains("Update the library", ex.Message);
          }
          finally { try { System.IO.File.Delete(path + ".json"); } catch { } }
      }

      [Fact]
      public void Legacy_file_without_envelope_still_loads()
      {
          // zpětná kompatibilita: modely uložené starou verzí knihovny (plochý JSON,
          // žádná FormatVersion ani checksum) musí jít načíst dál
          string path = TempBase();
          try
          {
              Model(seed: 3).SaveAsJson(path);

              // rozbal obálku → vznikne přesně starý formát
              string json = System.IO.File.ReadAllText(path + ".json");
              using var doc = System.Text.Json.JsonDocument.Parse(json);
              System.IO.File.WriteAllText(path + ".json", doc.RootElement.GetProperty("Model").GetRawText());

              var loaded = My_DNN.MDNN.LoadModel(path);
              Assert.NotNull(loaded);
              Assert.Equal(2, loaded.GetResults(new Tensor(new double[] { 0.5, -0.3 })).Data.Length);
          }
          finally { try { System.IO.File.Delete(path + ".json"); } catch { } }
      }

      [Fact]
      public void In_memory_snapshot_round_trips_through_the_same_path()
      {
          // SaveAsJsonString/LoadWeightsFromString jede stejnou obálkou jako disk
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
      // Pád nalezený 2026-08-11: LoadModel → Add → GetResults skončilo IndexOutOfRange.
      // LayerManager.Add() přestavěl jen POSLEDNÍ vrstvu, vložená zůstala s placeholder
      // neurony (0 vah) a spoléhala na SetInputSizeForFirstLayer — jenže to se pouštělo jen
      // pod podmínkou "Layers[0].Input_size_and_shape[0] <= 0". U načteného modelu má první
      // vrstva vstup ze souboru, podmínka neplatila a nová vrstva zůstala prázdná.
      //
      // Pro AutoML blokující: načíst checkpoint a rozšířit architekturu je základní operace.
      private static string TempBase()
          => System.IO.Path.Combine(System.IO.Path.GetTempPath(), "mdnn_ext_" + Guid.NewGuid().ToString("N"));

      private static void Save(string path, int? seed = null)
      {
          var m = new My_DNN.MDNN(new Dense(2, new Linear()), new SGD(0.01), new MSE(), seed);
          m.Layers.Add(new Dense(3, new ReLu()));
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

              var loaded = My_DNN.MDNN.LoadModel(path);
              loaded.Layers.Add(new Dense(4, new ReLu()));

              double[] output = loaded.GetResults(new Tensor(new double[] { 1, 1 })).Data;
              Assert.Equal(2, output.Length);
              Assert.All(output, v => Assert.True(double.IsFinite(v)));
          }
          finally { try { System.IO.File.Delete(path + ".json"); } catch { } }
      }

      [Fact]
      public void Loaded_model_can_be_trained_after_extension()
      {
          // nejen že to nespadne — rozšířený model se musí i doučit
          string path = TempBase();
          try
          {
              Save(path);

              var loaded = My_DNN.MDNN.LoadModel(path);
              loaded.Layers.Add(new Dense(4, new ReLu()));

              var x = new Tensor(new double[] { 1, 1 });
              var t = new Tensor(new double[] { 0.5, -0.5 });

              double lossBefore = loaded.Loss.CalculateAndGetLoss(loaded.GetResults(x).Data, t.Data);
              for (int i = 0; i < 200; i++)
              {
                  loaded.Train.Fit(x, t);
                  loaded.Train.UpdateParams();
              }
              double lossAfter = loaded.Loss.CalculateAndGetLoss(loaded.GetResults(x).Data, t.Data);

              Assert.True(lossAfter < lossBefore, $"loss {lossBefore} → {lossAfter}, model se neučí");
          }
          finally { try { System.IO.File.Delete(path + ".json"); } catch { } }
      }

      [Fact]
      public void Extending_loaded_model_keeps_weights_of_untouched_layers()
      {
          // vrstvy, jejichž tvar se nemění, si musí natrénované váhy podržet (2c-10)
          string path = TempBase();
          try
          {
              Save(path);

              var loaded = My_DNN.MDNN.LoadModel(path);
              double[] hiddenBefore = (double[])((Dense)loaded.Layers.Layers[0]).Neurons[0].Weights.Clone();

              loaded.Layers.Add(new Dense(4, new ReLu()));
              loaded.GetResults(new Tensor(new double[] { 1, 1 }));

              // první skrytá vrstva dostává pořád vstup modelu → beze změny
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

              var loaded = My_DNN.MDNN.LoadModel(path);
              Assert.Equal(2, loaded.Layers.Layers.Count);

              loaded.Layers.RemoveAt(0);                       // pryč se skrytou vrstvou

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
          // Stejný CNN jako v mdnn_test: Conv16 → MaxPool → Conv32 → MaxPool → Dense64 → Dense10.
          // Regrese: conv se dřív podpočítával (chybělo × počet filtrů) → 52115 místo 56714.
          var model = new My_DNN.MDNN(new Dense(10, new Softmax()), new Adam(0.001), new CrossEntropy());
          model.Layers.Add(new Conv(16, 3, new ReLu(), "valid"));   // 28×28×1 → 26×26×16
          model.Layers.Add(new MaxPool(2));                         // → 13×13×16
          model.Layers.Add(new Conv(32, 3, new ReLu(), "valid"));   // → 11×11×32
          model.Layers.Add(new MaxPool(2));                         // → 5×5×32
          model.Layers.Add(new Dense(64, new ReLu()));             // flatten 800 → 64

          model.GetResults(new Tensor(new double[784]));           // spustí wiring (nastaví tvary/kanály)

          // Conv1:  16·(3·3·1)+16  = 160
          // Conv2:  32·(3·3·16)+32 = 4640
          // Dense64: 64·800+64     = 51264
          // Dense10: 10·64+10      = 650
          Assert.Equal(160 + 4640 + 51264 + 650, ConsoleControler.CountTrainableParams(model));  // 56714
      }
  }
