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

  public class LayerInsertTests : GradientCheckTestBase
  {
      // 2c-7: Insert na pozici 0 předá Input_size_and_shape následující vrstvy = placeholder {0}
      // → Conv.ConvertTo2D udělá rows = ceil(sqrt(0)) = 0 → DivideByZeroException.
      [Fact]
      public void Insert_conv_before_dense_does_not_crash()
      {
          var model = new My_DNN.MDNN(new Dense(2, new Linear()), new SGD(0.01), new MSE());

          var ex = Record.Exception(() => model.Layers.Insert(0, new Conv(2, 2, new ReLu(), "valid")));

          // Buď to projde, nebo to musí být SROZUMITELNÁ výjimka — ne DivideByZeroException
          // z hloubi ConvertTo2D.
          Assert.False(ex is DivideByZeroException,
              $"Insert spadl na {ex?.GetType().Name} místo srozumitelné hlášky (2c-7)");
      }
  }

  // ==========================================================================================
  //  Fáze 2c — dávka 1 (2c-11, 2c-12, 2c-13)
  // ==========================================================================================

  public class ConvSamePaddingTests : GradientCheckTestBase
  {
      // 2c-12: `same` se SUDÝM kernelem paddoval symetricky k/2 na každou stranu, takže
      // padded rozměr byl in+k a konvoluce měla dát in+1 řádků — smyčka ale jela jen do
      // outputShape[0] = in, takže poslední řádek a sloupec tiše zmizely.
      //
      // POZOR: gradient check tohle NECHYTÍ. Forward i backward používaly stejný (špatný)
      // padding, takže spolu konzistentně souhlasily. Musí se testovat SÉMANTIKA forwardu
      // proti ručně spočítané referenci.
      [Fact]
      public void Same_padding_with_even_kernel_matches_hand_computed_reference()
      {
          // vstup 2×2×1 = [[1,2],[3,4]], kernel 2×2 samé jedničky, bias 0, Linear
          var conv = new Conv(1, 2, new Linear(), "same");
          conv.LayerAdjustment(null, new int[] { 2, 2, 1 });

          for (int i = 0; i < 2; i++)
          for (int j = 0; j < 2; j++)
              conv.Kernel[0][i][j][0] = 1.0;
          conv.Biases[0] = 0.0;

          var input = new double[2, 2, 1];
          input[0, 0, 0] = 1; input[0, 1, 0] = 2;
          input[1, 0, 0] = 3; input[1, 1, 0] = 4;

          double[] output = conv.FeedForward(new Tensor(input)).Data;

          // Keras konvence pro k=2: pad celkem k-1 = 1 → 0 před, 1 za. Padded 3×3:
          //   1 2 0
          //   3 4 0
          //   0 0 0
          // okna 2×2 se součtem jedniček:
          //   [0,0] = 1+2+3+4 = 10     [0,1] = 2+0+4+0 = 6
          //   [1,0] = 3+4+0+0 = 7      [1,1] = 4+0+0+0 = 4
          Assert.Equal(new double[] { 10, 6, 7, 4 }, output);

          // (Se starým symetrickým paddingem vycházelo [1, 3, 4, 10] — posunuté okno.)
      }

      // POZNÁMKA: tenhle test bug 2c-12 NECHYTÁ (ověřeno — na starém kódu procházel).
      // Tvar byl „správný" i předtím, jen obsah posunutý o pixel. Drží se tu jako guard
      // proti budoucí regresi v geometrii, ne jako důkaz opravy — tím je test výš.
      [Fact]
      public void Same_padding_preserves_spatial_shape_for_even_and_odd_kernels()
      {
          foreach (int k in new[] { 2, 3, 4, 5 })
          {
              var conv = new Conv(2, k, new Linear(), "same");
              conv.LayerAdjustment(null, new int[] { 6, 6, 1 });

              Assert.Equal(new int[] { 6, 6, 2 }, conv.Output_size_and_shape);

              // a hlavně: forward ten tvar opravdu vyplní (6·6·2 hodnot)
              double[] output = conv.FeedForward(new Tensor(new double[6, 6, 1])).Data;
              Assert.Equal(6 * 6 * 2, output.Length);
          }
      }
  }

  // ==========================================================================================
  //  Fáze 2c — dávka 3 (2c-10)
  // ==========================================================================================

  public class LayerAdjustmentIdempotenceTests : GradientCheckTestBase
  {
      // 2c-10: LayerAdjustment vždycky zahodilo neurony a postavilo je znovu s náhodnou
      // inicializací — i když se tvar vůbec neměnil. Natrénované váhy tak mizely při
      // volání, které nemá co změnit.
      //
      // POZOR na správné vymezení: když se dimenze MĚNÍ (jiný počet neuronů předchozí
      // vrstvy → jiná délka vektoru vah), je přestavba nevyhnutelná a je to korektní
      // chování, ne chyba. Testujeme tedy oba směry.
      private static My_DNN.MDNN TrainedModel(out Dense outputLayer, int hiddenNeurons)
      {
          outputLayer = new Dense(2, new Linear());
          var model = new My_DNN.MDNN(outputLayer, new SGD(0.3), new MSE());
          model.Layers.Add(new Dense(hiddenNeurons, new ReLu()));
          model.Layers.SetInputSizeForFirstLayer(new int[] { 2 });

          for (int i = 0; i < 30; i++)
              model.Train.Fit(new Tensor(new double[] { 1, 0.5 }), new Tensor(new double[] { 0.5, -0.5 }));
          model.Train.UpdateParams();

          return model;
      }

      [Fact]
      public void Repeated_SetInputSizeForFirstLayer_keeps_trained_weights()
      {
          var model = TrainedModel(out Dense outputLayer, 3);
          double[] before = (double[])outputLayer.Neurons[0].Weights.Clone();

          model.Layers.SetInputSizeForFirstLayer(new int[] { 2 });   // stejný tvar → nic měnit netřeba

          Assert.Equal(before, outputLayer.Neurons[0].Weights);
      }

      [Fact]
      public void Adding_layer_with_same_width_keeps_neighbour_weights()
      {
          var model = TrainedModel(out Dense outputLayer, 3);
          double[] before = (double[])outputLayer.Neurons[0].Weights.Clone();

          model.Layers.Add(new Dense(3, new ReLu()));   // 3 == předchozí šířka → vstup výstupní vrstvy se nemění

          Assert.Equal(before, outputLayer.Neurons[0].Weights);
      }

      [Fact]
      public void Adding_layer_with_different_width_must_rebuild()
      {
          // opačný směr: tady se přestavět MUSÍ, jinak by váhy nesouhlasily s novým vstupem
          var model = TrainedModel(out Dense outputLayer, 3);
          Assert.Equal(3, outputLayer.Neurons[0].Weights.Length);

          model.Layers.Add(new Dense(5, new ReLu()));
          // Add() nově jen ZAPOJÍ tvary; parametry se materializují až před prvním forwardem
          model.GetResults(new Tensor(new double[] { 1, 0.5 }));

          Assert.Equal(5, outputLayer.Neurons[0].Weights.Length);
      }

      [Fact]
      public void Loaded_model_survives_layer_adjustment()
      {
          // nejtišší varianta téhož: načtu natrénovaný model a cokoli, co spustí
          // LayerAdjustment se stejným tvarem, ho dřív přepsalo náhodnými vahami
          var model = TrainedModel(out Dense _, 3);
          string path = System.IO.Path.Combine(System.IO.Path.GetTempPath(), "mdnn_adj_" + Guid.NewGuid().ToString("N"));
          try
          {
              model.SaveAsJson(path);
              var loaded = My_DNN.MDNN.LoadModel(path);
              var loadedOutput = (Dense)loaded.Layers.Layers[loaded.Layers.Layers.Count - 1];
              double[] before = (double[])loadedOutput.Neurons[0].Weights.Clone();

              loaded.Layers.SetInputSizeForFirstLayer(new int[] { 2 });

              Assert.Equal(before, loadedOutput.Neurons[0].Weights);
          }
          finally { try { System.IO.File.Delete(path + ".json"); } catch { } }
      }

      [Fact]
      public void Conv_kernels_survive_adjustment_with_same_shape()
      {
          var conv = new Conv(2, 3, new ReLu(), "valid");
          conv.LayerAdjustment(null, new int[] { 8, 8, 1 });

          conv.Kernel[0][0][0][0] = 42.0;                       // rozpoznatelná hodnota
          double[][][][] before = conv.Kernel;
          double firstBefore = before[0][0][0][0];

          conv.LayerAdjustment(null, new int[] { 8, 8, 1 });     // stejný tvar

          Assert.Equal(firstBefore, conv.Kernel[0][0][0][0]);
      }

      [Fact]
      public void Conv_kernels_rebuild_when_channel_count_changes()
      {
          var conv = new Conv(2, 3, new ReLu(), "valid");
          conv.LayerAdjustment(null, new int[] { 8, 8, 1 });
          Assert.Single(conv.Kernel[0][0][0]);                   // zatím 1 vstupní kanál

          conv.LayerAdjustment(null, new int[] { 8, 8, 3 });     // 1 → 3 kanály

          Assert.Equal(3, conv.Kernel[0][0][0].Length);
      }
  }
