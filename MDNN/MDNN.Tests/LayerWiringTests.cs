using System;
  using Xunit;
  using MDNN;
  using MDNN.Layers;
  using MDNN.Activations;
  using MDNN.Optimizers;
  using MDNN.Losses;
  using System.Linq;

  namespace MDNN.Tests;

  public class LayerInsertTests : GradientCheckTestBase
  {
      // Previously: Insert at position 0 passed the next layer's InputShape = placeholder {0}
      // → Conv.ConvertTo2D computed rows = ceil(sqrt(0)) = 0 → DivideByZeroException.
      [Fact]
      public void Insert_conv_before_dense_does_not_crash()
      {
          var model = new NeuralNetwork(new Dense(2, new Linear()), new SGD(0.01), new MSE());

          var ex = Record.Exception(() => model.Layers.Insert(0, new Conv(2, 2, new ReLU(), "valid")));

          // Either it succeeds, or it must be a CLEAR exception, not a DivideByZeroException
          // from deep inside ConvertTo2D.
          Assert.False(ex is DivideByZeroException,
              $"Insert failed with {ex?.GetType().Name} instead of a clear message");
      }
  }

  // ==========================================================================================
  //  Conv "same" padding
  // ==========================================================================================

  public class ConvSamePaddingTests : GradientCheckTestBase
  {
      // Previously `same` with an EVEN kernel padded symmetrically by k/2 on each side, so the
      // padded size was in+k and the convolution should have produced in+1 rows; the loop only
      // ran up to outputShape[0] = in, so the last row and column silently disappeared.
      //
      // NOTE: a gradient check does NOT catch this. Forward and backward used the same (wrong)
      // padding, so they agreed with each other. The forward SEMANTICS must be tested against
      // a hand-computed reference.
      [Fact]
      public void Same_padding_with_even_kernel_matches_hand_computed_reference()
      {
          // input 2×2×1 = [[1,2],[3,4]], 2×2 kernel of all ones, bias 0, Linear
          var conv = new Conv(1, 2, new Linear(), "same");
          conv.Build(null, new int[] { 2, 2, 1 });

          for (int i = 0; i < 2; i++)
          for (int j = 0; j < 2; j++)
              conv.Kernels[0][i][j][0] = 1.0;
          conv.Biases[0] = 0.0;

          var input = new double[2, 2, 1];
          input[0, 0, 0] = 1; input[0, 1, 0] = 2;
          input[1, 0, 0] = 3; input[1, 1, 0] = 4;

          double[] output = conv.FeedForward(new Tensor(input)).Data;

          // Keras convention for k=2: total pad k-1 = 1 → 0 before, 1 after. Padded 3×3:
          //   1 2 0
          //   3 4 0
          //   0 0 0
          // 2×2 windows summed with the all-ones kernel:
          //   [0,0] = 1+2+3+4 = 10     [0,1] = 2+0+4+0 = 6
          //   [1,0] = 3+4+0+0 = 7      [1,1] = 4+0+0+0 = 4
          Assert.Equal(new double[] { 10, 6, 7, 4 }, output);

          // (The old symmetric padding gave [1, 3, 4, 10]: a shifted window.)
      }

      // NOTE: this test does NOT catch the even-kernel padding bug (verified: it passed on the
      // old code). The shape was "right" before too, only the content was shifted by a pixel. It
      // stays as a guard against future geometry regressions, not as proof of the fix; that is the test above.
      [Fact]
      public void Same_padding_preserves_spatial_shape_for_even_and_odd_kernels()
      {
          foreach (int k in new[] { 2, 3, 4, 5 })
          {
              var conv = new Conv(2, k, new Linear(), "same");
              conv.Build(null, new int[] { 6, 6, 1 });

              Assert.Equal(new int[] { 6, 6, 2 }, conv.OutputShape);

              // and above all: forward actually fills that shape (6·6·2 values)
              double[] output = conv.FeedForward(new Tensor(new double[6, 6, 1])).Data;
              Assert.Equal(6 * 6 * 2, output.Length);
          }
      }
  }

  // ==========================================================================================
  //  Build must be idempotent for an unchanged shape
  // ==========================================================================================

  public class LayerAdjustmentIdempotenceTests : GradientCheckTestBase
  {
      // Previously Build always discarded the neurons and rebuilt them with random
      // initialization, even when the shape did not change at all. Trained weights were lost
      // on a call that had nothing to change.
      //
      // Mind the exact boundary: when the dimensions DO change (different neuron count in the
      // previous layer → different weight vector length), rebuilding is unavoidable and is
      // correct behaviour, not a bug. So both directions are tested.
      private static NeuralNetwork TrainedModel(out Dense outputLayer, int hiddenNeurons)
      {
          outputLayer = new Dense(2, new Linear());
          var model = new NeuralNetwork(outputLayer, new SGD(0.3), new MSE());
          model.Layers.Add(new Dense(hiddenNeurons, new ReLU()));
          model.Layers.SetInputShape(new int[] { 2 });

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

          model.Layers.SetInputShape(new int[] { 2 });   // same shape → nothing to change

          Assert.Equal(before, outputLayer.Neurons[0].Weights);
      }

      [Fact]
      public void Adding_layer_with_same_width_keeps_neighbour_weights()
      {
          var model = TrainedModel(out Dense outputLayer, 3);
          double[] before = (double[])outputLayer.Neurons[0].Weights.Clone();

          model.Layers.Add(new Dense(3, new ReLU()));   // 3 == previous width → the output layer's input does not change

          Assert.Equal(before, outputLayer.Neurons[0].Weights);
      }

      [Fact]
      public void Adding_layer_with_different_width_must_rebuild()
      {
          // the opposite direction: here a rebuild is REQUIRED, otherwise the weights would not match the new input
          var model = TrainedModel(out Dense outputLayer, 3);
          Assert.Equal(3, outputLayer.Neurons[0].Weights.Length);

          model.Layers.Add(new Dense(5, new ReLU()));
          // Add() only WIRES the shapes; parameters are materialized just before the first forward pass
          model.GetResults(new Tensor(new double[] { 1, 0.5 }));

          Assert.Equal(5, outputLayer.Neurons[0].Weights.Length);
      }

      [Fact]
      public void Loaded_model_survives_layer_adjustment()
      {
          // the quietest variant of the same bug: load a trained model; anything that triggered
          // Build with the same shape used to overwrite it with random weights
          var model = TrainedModel(out Dense _, 3);
          string path = System.IO.Path.Combine(System.IO.Path.GetTempPath(), "mdnn_adj_" + Guid.NewGuid().ToString("N"));
          try
          {
              model.SaveAsJson(path);
              var loaded = NeuralNetwork.LoadModel(path);
              var loadedOutput = (Dense)loaded.Layers.Layers[loaded.Layers.Layers.Count - 1];
              double[] before = (double[])loadedOutput.Neurons[0].Weights.Clone();

              loaded.Layers.SetInputShape(new int[] { 2 });

              Assert.Equal(before, loadedOutput.Neurons[0].Weights);
          }
          finally { try { System.IO.File.Delete(path + ".json"); } catch { } }
      }

      [Fact]
      public void Conv_kernels_survive_adjustment_with_same_shape()
      {
          var conv = new Conv(2, 3, new ReLU(), "valid");
          conv.Build(null, new int[] { 8, 8, 1 });

          conv.Kernels[0][0][0][0] = 42.0;                       // recognizable value
          double[][][][] before = conv.Kernels;
          double firstBefore = before[0][0][0][0];

          conv.Build(null, new int[] { 8, 8, 1 });     // same shape

          Assert.Equal(firstBefore, conv.Kernels[0][0][0][0]);
      }

      [Fact]
      public void Conv_kernels_rebuild_when_channel_count_changes()
      {
          var conv = new Conv(2, 3, new ReLU(), "valid");
          conv.Build(null, new int[] { 8, 8, 1 });
          Assert.Single(conv.Kernels[0][0][0]);                   // 1 input channel so far

          conv.Build(null, new int[] { 8, 8, 3 });     // 1 → 3 channels

          Assert.Equal(3, conv.Kernels[0][0][0].Length);
      }
  }
