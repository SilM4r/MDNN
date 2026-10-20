using System;
  using Xunit;
  using MDNN;
  using MDNN.Layers;
  using MDNN.Activations;
  using MDNN.Optimizers;
  using MDNN.Losses;
  using System.Linq;

  namespace MDNN.Tests;

  // ---------- activation derivatives ----------

  public class ActivationDerivativeTests : GradientCheckTestBase
  {
      [Theory]
      [InlineData(-2.0)]
      [InlineData(-0.5)]
      [InlineData(0.3)]
      [InlineData(1.7)]
      public void Tanh_derivative_matches_numeric(double x)
      {
          ActivationFunction act = new Tanh();

          double analytic = act.Derivative(x);
          double numeric  = Numeric.CentralDiff(act.Apply, x);

          Assert.True(Numeric.RelErr(analytic, numeric) < 1e-6,
              $"x={x}: analytic={analytic}, numeric={numeric}");
      }
      [Theory]
      [InlineData(-2.0)]
      [InlineData(-0.5)]
      [InlineData(0.3)]
      [InlineData(1.7)]
      public void Sigmoid_derivative_matches_numeric(double x)
      {
          ActivationFunction act = new Sigmoid();

          double analytic = act.Derivative(x);
          double numeric  = Numeric.CentralDiff(act.Apply, x);

          Assert.True(Numeric.RelErr(analytic, numeric) < 1e-6,
              $"x={x}: analytic={analytic}, numeric={numeric}");
      }
      
      [Theory]
      [InlineData(-2.0)]
      [InlineData(-0.5)]
      [InlineData(0.3)]
      [InlineData(1.7)]
      public void Linear_derivative_matches_numeric(double x)
      {
          ActivationFunction act = new Linear();

          double analytic = act.Derivative(x);
          double numeric  = Numeric.CentralDiff(act.Apply, x);

          Assert.True(Numeric.RelErr(analytic, numeric) < 1e-6,
              $"x={x}: analytic={analytic}, numeric={numeric}");
      }
      
      [Theory]
      // NOTE: don't test x=0 — ReLU has a kink at zero, numeric and analytic results diverge
      [InlineData(-2.0)]
      [InlineData(-0.5)]
      [InlineData(0.3)]
      [InlineData(1.7)]
      public void ReLu_derivative_matches_numeric(double x)
      {
          ActivationFunction act = new ReLU();

          double analytic = act.Derivative(x);
          double numeric  = Numeric.CentralDiff(act.Apply, x);

          Assert.True(Numeric.RelErr(analytic, numeric) < 1e-6,
              $"x={x}: analytic={analytic}, numeric={numeric}");
      }
      
      [Theory]
      // NOTE: don't test x=0 — ReLU has a kink at zero, numeric and analytic results diverge
      [InlineData(-2.0)]
      [InlineData(-0.5)]
      [InlineData(0.3)]
      [InlineData(1.7)]
      public void Leak_ReLu_derivative_matches_numeric(double x)
      {
          ActivationFunction act = new LeakyReLU();

          double analytic = act.Derivative(x);
          double numeric  = Numeric.CentralDiff(act.Apply, x);

          Assert.True(Numeric.RelErr(analytic, numeric) < 1e-6,
              $"x={x}: analytic={analytic}, numeric={numeric}");
      }
      
      [Theory]
      [InlineData(new[] {1.0, 2.0, 3.0})]
      [InlineData(new[] {1000.0, 1001.0})]
      public void Softmax_ApplyToLayer_isValidDistribution(double[] x)
      {
          LayerActivationFunction act = new Softmax();
          
          var s = act.ApplyToLayer(x);
          Assert.Equal(1.0, s.Sum(), 10);
          Assert.All(s, v => Assert.InRange(v, 0, 1));
      }
      
      [Theory]
      [InlineData(new[] {1.0, 2.0, 3.0})]
      [InlineData(new[] {1000.0, 1001.0})]
      public void Softmax_Jacobian_formula_ApplyToLayer_matches_numeric(double[] x)
      {
          LayerActivationFunction act = new Softmax();
          
          var s = act.ApplyToLayer(x);
          for (int i = 0; i < x.Length; i++)          // which output
          for (int k = 0; k < x.Length; k++)      // which input is perturbed
          {
              double analytic = s[i] * ((i == k ? 1.0 : 0.0) - s[k]);  // formula
              double numeric  = Numeric.NumJac((Softmax)act, x, i, k);                    // measured
              Assert.True(Numeric.RelErr(analytic, numeric) < 1e-5,
                  $"J[{i},{k}]: analytic={analytic}, numeric={numeric}");
          }
      }
      
      [Fact]
      public void Softmax_crossentropy_weight_grads_match_numeric()
      {
          var x      = new Tensor(new double[] { 0.5, -0.3, 0.8 });   // 3 inputs
          var target = new Tensor(new double[] { 0.0, 1.0, 0.0 });    // one-hot: class 1

          // 3 neurons → 3 classes, softmax output, CrossEntropy loss
          var model = new NeuralNetwork(new Dense(3, new Softmax()), new SGD(0.01), new CrossEntropy());

          // ---- ANALYTIC: what the library thinks ----
          model.Train.Fit(x, target);                 // forward + backprop, do NOT call UpdateParams
          var neurons = ((Dense)model.Layers.Layers[0]).Neurons;
          var analytic = neurons.Select(n => (double[])n.gradientsW.Clone()).ToList();

          // ---- NUMERIC: what actually happens to the loss ----
          Func<double> lossAt = () =>
              model.Loss.CalculateAndGetLoss(
                  model.GetResults(x).Data, target.Data);

          double maxRel = 0; int worstI = -1, worstJ = -1;
          for (int i = 0; i < neurons.Count; i++)
          for (int j = 0; j < neurons[i].Weights.Length; j++)
          {
              double num = Numeric.ParamGrad(lossAt, ref neurons[i].Weights[j]);
              double rel = Numeric.RelErr(analytic[i][j], num);
              if (rel > maxRel) { maxRel = rel; worstI = i; worstJ = j; }
          }

          Assert.True(maxRel < 1e-5,
              $"maxRel={maxRel} at neuron {worstI} weight {worstJ}:" +
              $"analytic={analytic[worstI][worstJ]}");
      }
      
  }

  public class LossFunctionTests: GradientCheckTestBase
  {
      [Fact]
      public void MSE_Loss_matches_numeric()
      {
          Assert.Equal(0.09, new MSE().LossFunction(0.5, 0.2), 10);
      }
      
      [Theory]
      [InlineData(0.5, 0.2)]
      [InlineData(2.0, -1.0)]
      [InlineData(-0.3, 0.4)]
      public void MSE_derivative_matches_numeric(double value, double target)
      {
          var mse = new MSE();
          
          double analytic = mse.DerivativeOfLossFunction(value, target);
          double numeric  = Numeric.CentralDiff(v => mse.LossFunction(v, target), value);
          
          Assert.True(Numeric.RelErr(analytic, numeric) < 1e-6,
              $"v={value}, t={target}: analytic={analytic}, numeric={numeric}");
      }
      
      [Fact]
      public void CrossEntropy_LossFunction_matches_numeric()
      {
          var ce = new CrossEntropy();
          
          Assert.Equal(-Math.Log(0.7), ce.LossFunction(0.7, 1.0), 10);   
          Assert.Equal(0.0,            ce.LossFunction(0.7, 0.0), 10);    
      }
  }

  public class LayerTests: GradientCheckTestBase
  {
      [Fact]
      public void Dense_single_layer_test()
      {
          var x      = new Tensor(new double[] { 0.5, -0.3, 0.8 });   // 3 inputs
          var target = new Tensor(new double[] { 0.0, 1.0, 0.0 });    // one-hot: class 1

          var model = new NeuralNetwork(new Dense(3, new Tanh()), new SGD(0.01), new MSE());

          // ---- ANALYTIC: what the library thinks ----
          model.Train.Fit(x, target);                 // forward + backprop, do NOT call UpdateParams
          var neurons = ((Dense)model.Layers.Layers[0]).Neurons;
          var analytic = neurons.Select(n => (double[])n.gradientsW.Clone()).ToList();

          // ---- NUMERIC: what actually happens to the loss ----
          Func<double> lossAt = () =>
              model.Loss.CalculateAndGetLoss(
                  model.GetResults(x).Data, target.Data);

          double maxRel = 0; int worstI = -1, worstJ = -1;
          for (int i = 0; i < neurons.Count; i++)
          for (int j = 0; j < neurons[i].Weights.Length; j++)
          {
              double num = Numeric.ParamGrad(lossAt, ref neurons[i].Weights[j]);
              double rel = Numeric.RelErr(analytic[i][j], num);
              if (rel > maxRel) { maxRel = rel; worstI = i; worstJ = j; }
          }

          Assert.True(maxRel < 1e-5,
              $"maxRel={maxRel} at neuron {worstI} weight {worstJ}:" +
              $"analytic={analytic[worstI][worstJ]}");
      }
      
      [Fact]
      public void Dense_multi_layer_test()
      {
          var x      = new Tensor(new double[] { 0.5, -0.3, 0.8, 0.1});   // 4 inputs
          var target = new Tensor(new double[] { 0.0, 1.0});

          var model = new NeuralNetwork(new Dense(2, new Tanh()), new SGD(0.01), new MSE());
          model.Layers.Add(new Dense(4, new Tanh()));

          // ---- ANALYTIC: what the library thinks ----
          model.Train.Fit(x, target);                 // forward + backprop, do NOT call UpdateParams
          var neurons = new List<Neuron>();
          neurons.AddRange(((Dense)model.Layers.Layers[0]).Neurons);
          neurons.AddRange(((Dense)model.Layers.Layers[1]).Neurons);
          
          var analytic = neurons.Select(n => (double[])n.gradientsW.Clone()).ToList();

          // ---- NUMERIC: what actually happens to the loss ----
          Func<double> lossAt = () =>
              model.Loss.CalculateAndGetLoss(
                  model.GetResults(x).Data, target.Data);

          double maxRel = 0; int worstI = -1, worstJ = -1;
          for (int i = 0; i < neurons.Count; i++)
          for (int j = 0; j < neurons[i].Weights.Length; j++)
          {
              double num = Numeric.ParamGrad(lossAt, ref neurons[i].Weights[j]);
              double rel = Numeric.RelErr(analytic[i][j], num);
              if (rel > maxRel) { maxRel = rel; worstI = i; worstJ = j; }
          }

          Assert.True(maxRel < 1e-5,
              $"maxRel={maxRel} at neuron {worstI} weight {worstJ}:" +
              $"analytic={analytic[worstI][worstJ]}");
      }

      [Fact]
      public void Dense_two_hidden_layers_build_and_forward()
      {
          // Regression: TWO hidden Dense layers in a row + output = [Dense, Dense, Dense].
          // Before the SetInputShape fix the middle one (neuron-to-neuron) was NOT rebuilt
          // and kept placeholder neurons (0 weights) → IndexOutOfRange during forward.
          // (Dense_multi_layer_test only has [hidden, output] → it did not catch the bug.)
          var model = new NeuralNetwork(new Dense(2, new Linear()), new SGD(0.01), new MSE());
          model.Layers.Add(new Dense(3, new ReLU()));
          model.Layers.Add(new Dense(3, new ReLU()));

          var x = new Tensor(new double[] { 0.5, -0.3, 0.8, 0.1 });   // 4 inputs

          var output = model.GetResults(x);   // triggers SetInputShape; must not throw

          // each layer's input weights = previous layer's output (not placeholder 0)
          Assert.Equal(4, ((Dense)model.Layers.Layers[0]).Neurons[0].Weights.Length);  // 1st hidden: 4 inputs
          Assert.Equal(3, ((Dense)model.Layers.Layers[1]).Neurons[0].Weights.Length);  // 2nd hidden: 3 (output of 1st)
          Assert.Equal(3, ((Dense)model.Layers.Layers[2]).Neurons[0].Weights.Length);  // output: 3
          Assert.Equal(2, output.Data.Length);                                         // 2 outputs
      }

      [Fact]
      public void Conv_maxpool_dense_dense_pipeline_wires_correctly()
      {
          // Wiring between DIFFERENT layer types: Conv → MaxPool → Dense → Dense → Dense(out).
          // Verifies wiring across type changes (3D→3D→flatten→neuron) AND neuron-to-neuron
          // in the MIDDLE of the pipeline (Dense→Dense) — the old logic failed there (placeholder 0 weights).
          var model = new NeuralNetwork(new Dense(2, new Linear()), new SGD(0.01), new MSE());
          model.Layers.Add(new Conv(2, 3, new ReLU(), "valid"));   // Layers[0]
          model.Layers.Add(new MaxPool(2));                        // Layers[1]
          model.Layers.Add(new Dense(8, new ReLU()));              // Layers[2]  (after MaxPool)
          model.Layers.Add(new Dense(4, new ReLU()));              // Layers[3]  (neuron-to-neuron!)
          // Layers[4] = output Dense(2)

          var input = new double[8, 8, 1];                          // input 8×8×1
          for (int i = 0; i < 8; i++)
          for (int j = 0; j < 8; j++)
              input[i, j, 0] = (i * 8 + j) * 0.01;

          var output = model.GetResults(new Tensor(input));   // triggers wiring; must not throw

          // Conv valid: 8-3+1=6 → [6,6,2];  MaxPool/2: [3,3,2];  Dense flatten = 3*3*2 = 18
          Assert.Equal(new int[] { 6, 6, 2 }, model.Layers.Layers[0].OutputShape);
          Assert.Equal(new int[] { 3, 3, 2 }, model.Layers.Layers[1].OutputShape);
          Assert.Equal(18, ((Dense)model.Layers.Layers[2]).Neurons[0].Weights.Length);  // MaxPool → Dense (flatten 3*3*2)
          Assert.Equal(8,  ((Dense)model.Layers.Layers[3]).Neurons[0].Weights.Length);  // Dense → Dense (middle)
          Assert.Equal(4,  ((Dense)model.Layers.Layers[4]).Neurons[0].Weights.Length);  // Dense → Dense (output)
          Assert.Equal(2, output.Data.Length);
      }

      [Theory]
      [InlineData(4, 4, 1, 2)]
      [InlineData(6, 6, 1, 3)]
      [InlineData(4, 4, 2, 2)]
      [InlineData(10, 10, 3, 5)]
      public void MaxPool_input_gradient_matches_numeric(int x, int y, int z, int poolsize)
      {
          // distinct values across the WHOLE tensor = flat index (guaranteed unique → no ties)
          var input = new double[x, y, z];
          for (int i = 0; i < x; i++)
          for (int j = 0; j < y; j++)
          for (int c = 0; c < z; c++)
              input[i, j, c] = i * (y * z) + j * z + c;

          var pool = new MaxPool(poolsize);
          pool.Build(null, new int[] { x, y, z });
          pool.FeedForward(new Tensor(input));

          // g has the shape of the OUTPUT, not a hard-coded 2×2×1
          int outH = (x - poolsize) / poolsize + 1;
          int outW = (y - poolsize) / poolsize + 1;
          var g = new double[outH, outW, z];
          int gc = 1;
          for (int i = 0; i < outH; i++)
          for (int j = 0; j < outW; j++)
          for (int c = 0; c < z; c++)
              g[i, j, c] = gc++;
          double[] gFlat = new Tensor(g).Data;

          double[] dInAnalytic = pool.Backward(new Tensor(g)).Data;

          Func<double> lossAt = () =>
          {
              double[] o = pool.FeedForward(new Tensor(input)).Data;
              double s = 0;
              for (int k = 0; k < o.Length; k++) s += gFlat[k] * o[k];
              return s;
          };

          double maxRel = 0;
          for (int i = 0; i < x; i++)
          for (int j = 0; j < y; j++)
          for (int c = 0; c < z; c++)
          {
              int flat = i * (y * z) + j * z + c;   // same stride as dInAnalytic
              double num = Numeric.ParamGrad(lossAt, ref input[i, j, c]);
              maxRel = Math.Max(maxRel, Numeric.RelErr(dInAnalytic[flat], num));
          }

          Assert.True(maxRel < 1e-5, $"maxRel={maxRel}");
      }
      
      [Fact] 
      public void MaxPool_forward_takes_window_max()
      {
          var input = new double[4,4,1];
          
          for (int i=0;i<4;i++) for (int j=0;j<4;j++) input[i,j,0]=i*4+j;
          
          var pool = new MaxPool(2); 
          pool.Build(null, new[]{4,4,1});
          var o = (double[,,])pool.FeedForward(new Tensor(input)).ToArray();
          Assert.Equal(5.0, o[0,0,0]);   // max of window {0,1,4,5}
          Assert.Equal(15.0, o[1,1,0]);  // max of window {10,11,14,15}
      }
      
      [Fact]
      public void RNN_weight_grads_match_numeric_sequence()
      {
          // 2-step sequence, input dim 2
          var xs = new[] {
              new Tensor(new double[]{ 0.5, -0.3 }),
              new Tensor(new double[]{ 0.2,  0.4 }),
          };
          var ts = new[] {
              new Tensor(new double[]{ 0.1, -0.2 }),
              new Tensor(new double[]{ 0.3,  0.0 }),
          };

          var model = new NeuralNetwork(new Dense(2, new Tanh()), new SGD(0.01), new MSE());
          model.Layers.Add(new RNN(2, new Tanh()));      // Layers[0]=RNN, Layers[1]=Dense
          var loss = model.Loss;

          // ANALYTIC: run the sequence, accumulate grads (ResetSequence, per-step Fit, NO UpdateParams)
          model.ResetSequence();
          for (int t = 0; t < xs.Length; t++) model.Train.Fit(xs[t], ts[t]);
          var rnn = ((RNN)model.Layers.Layers[0]).Neurons;
          var analytic = rnn.Select(n => (double[])n.gradientsW.Clone()).ToList();

          // NUMERIC: loss of the WHOLE sequence as a function of the weights (each call resets the state!)
          Func<double> seqLoss = () =>
          {
              model.ResetSequence();
              double L = 0;
              for (int t = 0; t < xs.Length; t++)
                  L += loss.CalculateAndGetLoss(model.GetResults(xs[t]).Data, ts[t].Data);
              return L;
          };

          double maxRel = 0; int wi=-1, wj=-1;
          for (int i = 0; i < rnn.Count; i++)
          for (int j = 0; j < rnn[i].Weights.Length; j++)   // last weight = recurrent
          {
              double num = Numeric.ParamGrad(seqLoss, ref rnn[i].Weights[j]);
              double rel = Numeric.RelErr(analytic[i][j], num);
              if (rel > maxRel) { maxRel = rel; wi=i; wj=j; }
          }

          Assert.True(maxRel < 1e-5, $"maxRel={maxRel} at neuron {wi} weight {wj}");
      }

      [Fact]
      public void RNN_weight_grads_match_numeric_singlestep()
      {
          // sequence of length 1 → recurrence has no effect (h(0)=0); verifies input weights + bias.
          // The recurrent weight trivially gets 0 (its input h(0)=0) → 0 == 0.
          var xs = new[] { new Tensor(new double[]{ 0.5, -0.3 }) };
          var ts = new[] { new Tensor(new double[]{ 0.1, -0.2 }) };

          var model = new NeuralNetwork(new Dense(2, new Tanh()), new SGD(0.01), new MSE());
          model.Layers.Add(new RNN(2, new Tanh()));
          var loss = model.Loss;

          model.ResetSequence();
          for (int t = 0; t < xs.Length; t++) model.Train.Fit(xs[t], ts[t]);
          var rnn = ((RNN)model.Layers.Layers[0]).Neurons;
          var analytic = rnn.Select(n => (double[])n.gradientsW.Clone()).ToList();

          Func<double> seqLoss = () =>
          {
              model.ResetSequence();
              double L = 0;
              for (int t = 0; t < xs.Length; t++)
                  L += loss.CalculateAndGetLoss(model.GetResults(xs[t]).Data, ts[t].Data);
              return L;
          };

          double maxRel = 0; int wi=-1, wj=-1;
          for (int i = 0; i < rnn.Count; i++)
          for (int j = 0; j < rnn[i].Weights.Length; j++)
          {
              double num = Numeric.ParamGrad(seqLoss, ref rnn[i].Weights[j]);
              double rel = Numeric.RelErr(analytic[i][j], num);
              if (rel > maxRel) { maxRel = rel; wi=i; wj=j; }
          }

          Assert.True(maxRel < 1e-5, $"maxRel={maxRel} at neuron {wi} weight {wj}");
      }

      // ---------- Conv gradient checks (3×3×1, kernel 2×2, 1 filtr, Linear, valid → 2×2×1) ----------

      // shared setup: input, conv, g

      private static Func<double> ConvLoss(Conv conv, double[,,] input, double[] gFlat) => () =>
      {
          double[] o = conv.FeedForward(new Tensor(input)).Data;
          double s = 0;
          for (int k = 0; k < o.Length; k++) s += gFlat[k] * o[k];
          return s;                                          // L = Σ g·output
      };
      
      private static ActivationFunction Act(string n) => n == "tanh" ? new Tanh() : new Linear();

      private static (double[,,] input, Conv conv, double[] gFlat, Tensor gTensor) ConvSetup(
          int h, int w, int c, int kernel, int filters, string pad, string actName)
      {
          // small distinct values → Tanh does not saturate (otherwise act'≈0 and the numeric result gets noisy)
          var input = new double[h, w, c];
          for (int i = 0; i < h; i++)
          for (int j = 0; j < w; j++)
          for (int cc = 0; cc < c; cc++)
              input[i, j, cc] = ((i * w + j) * c + cc) * 0.01 + 0.02;

          var conv = new Conv(filters, kernel, Act(actName), pad);
          conv.Build(null, new int[] { h, w, c });
          conv.FeedForward(new Tensor(input));

          // output shape: valid = h-k+1, same = h
          int outH = pad == "same" ? h : h - kernel + 1;
          int outW = pad == "same" ? w : w - kernel + 1;

          var g = new double[outH, outW, filters];
          int gc = 1;
          for (int i = 0; i < outH; i++)
          for (int j = 0; j < outW; j++)
          for (int f = 0; f < filters; f++)
              g[i, j, f] = gc++ * 0.1;

          return (input, conv, new Tensor(g).Data, new Tensor(g));
      }

      [Theory]
      [InlineData(3, 3, 1, 2, 1, "valid", "linear")]   // baseline
      [InlineData(4, 4, 2, 2, 1, "valid", "tanh")]      // multi-channel + tanh
      [InlineData(4, 4, 1, 2, 3, "valid", "tanh")]      // multi-filter
      [InlineData(5, 5, 2, 3, 2, "valid", "tanh")]      // channel+filter+kernel3
      [InlineData(5, 5, 1, 3, 2, "same",  "tanh")]      // same padding (odd kernel!)
      [InlineData(5, 5, 1, 2, 2, "same",  "tanh")]      // same padding EVEN kernel (asymmetric pad)
      public void Conv_input_gradient_matches_numeric(
          int h, int w, int c, int k, int f, string pad, string actName)
      {
          var (input, conv, gFlat, gTensor) = ConvSetup(h, w, c,k, f, pad, actName);

          // ANALYTIC dL/dinput (3D branch)
          double[] dInAnalytic = conv.Backward(gTensor).Data;

          var lossAt = ConvLoss(conv, input, gFlat);

          double maxRel = 0;
          for (int i = 0; i < h; i++)
          for (int j = 0; j < w; j++)
          for (int cc = 0; cc < c; cc++)
          {
              int flat = i * (w * c) + j * c + cc;
              double num = Numeric.ParamGrad(lossAt, ref input[i, j, cc]);
              maxRel = Math.Max(maxRel, Numeric.RelErr(dInAnalytic[flat], num));
          }
          Assert.True(maxRel < 1e-5, $"maxRel={maxRel}");
      }

      [Theory]
      [InlineData(3, 3, 1, 2, 1, "valid", "linear")]   // baseline
      [InlineData(4, 4, 2, 2, 1, "valid", "tanh")]      // multi-channel + tanh
      [InlineData(4, 4, 1, 2, 3, "valid", "tanh")]      // multi-filter
      [InlineData(5, 5, 2, 3, 2, "valid", "tanh")]      // channel+filter+kernel3
      [InlineData(5, 5, 1, 3, 2, "same",  "tanh")]      // same padding (odd kernel!)
      [InlineData(5, 5, 1, 2, 2, "same",  "tanh")]      // same padding EVEN kernel (asymmetric pad)
      public void Conv_kernel_gradient_matches_numeric(int h, int w, int c, int k, int f, string pad, string actName)
      {
          var (input, conv, gFlat, gTensor) = ConvSetup(h, w, c,k, f, pad, actName);

          // ANALYTIC: Backward takes ∂L/∂output, accumulates dKernels/dBiases and returns ∂L/∂input
          conv.Backward(gTensor);
          var dK = conv.dKernels;
          var kern = conv.Kernels;

          var lossAt = ConvLoss(conv, input, gFlat);

          double maxRel = 0;
          for (int fi = 0; fi < kern.Length; fi++)                 // over filters
          for (int ki = 0; ki < kern[0].Length; ki++)
          for (int kj = 0; kj < kern[0][0].Length; kj++)
          for (int cj = 0; cj < kern[0][0][0].Length; cj++)
          {
              double analytic = dK[fi][ki][kj][cj];
              double num = Numeric.ParamGrad(lossAt, ref conv.Kernels[fi][ki][kj][cj]);
              maxRel = Math.Max(maxRel, Numeric.RelErr(analytic, num));
          }
          Assert.True(maxRel < 1e-5, $"maxRel={maxRel}");
      }

      [Theory]
      [InlineData(3, 3, 1, 2, 1, "valid", "linear")]   // baseline
      [InlineData(4, 4, 2, 2, 1, "valid", "tanh")]      // multi-channel + tanh
      [InlineData(4, 4, 1, 2, 3, "valid", "tanh")]      // multi-filter
      [InlineData(5, 5, 2, 3, 2, "valid", "tanh")]      // channel+filter+kernel3
      [InlineData(5, 5, 1, 3, 2, "same",  "tanh")]      // same padding (odd kernel!)
      [InlineData(5, 5, 1, 2, 2, "same",  "tanh")]      // same padding EVEN kernel (asymmetric pad)
      public void Conv_bias_gradient_matches_numeric(int h, int w, int c, int k, int f, string pad, string actName)
      {
          var (input, conv, gFlat, gTensor) = ConvSetup(h, w, c,k, f, pad, actName);

          conv.Backward(gTensor);
          double[] analytic = (double[])conv.dBiases.Clone();

          var lossAt = ConvLoss(conv, input, gFlat);

          double maxRel = 0;
          for (int g = 0; g < conv.Biases.Length; g++)
          {
              double num = Numeric.ParamGrad(lossAt, ref conv.Biases[g]);
              maxRel = Math.Max(maxRel, Numeric.RelErr(analytic[g], num));
          }
          Assert.True(maxRel < 1e-5, $"maxRel={maxRel}");
      }
      [Fact]
      public void Conv_accepts_nonsquare_2D_input()
      {
          var input = new double[3, 4];                 // 2D, H≠W → triggers the reshape branch
          for (int i = 0; i < 3; i++)
          for (int j = 0; j < 4; j++)
              input[i, j] = i * 4 + j + 0.1;

          var conv = new Conv(1, 2, new Linear(), "valid");
          conv.Build(null, new int[] { 3, 4, 1 });

          var outp = conv.FeedForward(new Tensor(input));   // before the fix: ArgumentException (9 ≠ 12 elements)

          // valid, kernel 2 → output (3-2+1)×(4-2+1) = 2×3
          Assert.Equal(2, outp.Shape[0]);
          Assert.Equal(3, outp.Shape[1]);
      }

      [Fact]
      public void Full_CNN_stack_end_to_end_gradients_match_numeric()
      {
          // End-to-end gradient check of the WHOLE CNN pipeline (a mini version of the Program.cs model):
          // Conv → MaxPool → Conv → MaxPool → Dense → Dense(out).
          // The individual layers are gradient-checked SEPARATELY (Conv_*_gradient, MaxPool_*),
          // BUT this verifies the backprop WIRING through the WHOLE chain: the gradient flowing
          // from MaxPool back into Conv and through TWO conv layers in a row. Exactly what cannot
          // be verified by reading the code and what would otherwise break a long MNIST run hours in.
          //
          // Tanh (not ReLU) in hidden layers = clean numerics without the kink at zero (the ReLU
          // derivative is tested separately); small input → tanh does not saturate. Tolerance 1e-4
          // (looser than the isolated 1e-5) because of the chain depth + tanh-of-tanh.
          var input = new double[8, 8, 1];
          for (int i = 0; i < 8; i++)
          for (int j = 0; j < 8; j++)
              input[i, j, 0] = ((i * 8 + j) * 0.005) + 0.01;
          var x = new Tensor(input);
          var target = new Tensor(new double[] { 0.3, -0.2 });

          var model = new NeuralNetwork(new Dense(2, new Linear()), new SGD(0.01), new MSE());
          model.Layers.Add(new Conv(2, 3, new Tanh(), "valid"));   // [0]  8×8×1 → 6×6×2
          model.Layers.Add(new MaxPool(2));                        // [1]  → 3×3×2
          model.Layers.Add(new Conv(3, 2, new Tanh(), "valid"));   // [2]  → 2×2×3
          model.Layers.Add(new MaxPool(2));                        // [3]  → 1×1×3
          model.Layers.Add(new Dense(4, new Tanh()));              // [4]  flatten 3 → 4
          // [5] = output Dense(2, Linear)

          // ---- ANALYTIC: one Fit (forward + backprop through the WHOLE chain), NO UpdateParams ----
          model.Train.Fit(x, target);

          var conv1 = (Conv)model.Layers.Layers[0];
          var conv2 = (Conv)model.Layers.Layers[2];
          var denseH = ((Dense)model.Layers.Layers[4]).Neurons;
          var denseO = ((Dense)model.Layers.Layers[5]).Neurons;

          // sanity: the chain really got wired (not placeholder 0 weights)
          Assert.Equal(new int[] { 6, 6, 2 }, conv1.OutputShape);
          Assert.Equal(new int[] { 2, 2, 3 }, conv2.OutputShape);
          Assert.Equal(3, denseH[0].Weights.Length);   // flatten 1*1*3

          // ---- NUMERIC: what the loss of the WHOLE model actually does ----
          // (forward never overwrites the gradient buffers dKernels/dBiases/gradientsW,
          //  so they can be read inline even while parameters are perturbed)
          Func<double> lossAt = () =>
              model.Loss.CalculateAndGetLoss(model.GetResults(x).Data, target.Data);

          double maxRel = 0; string worst = "";
          void Check(double analytic, ref double param, string label)
          {
              double num = Numeric.ParamGrad(lossAt, ref param);
              double rel = Numeric.RelErr(analytic, num);
              if (rel > maxRel) { maxRel = rel; worst = $"{label}: analytic={analytic}, numeric={num}"; }
          }

          // conv1 + conv2: kernels and biases (the gradient came through the rest of the chain)
          foreach (var conv in new[] { conv1, conv2 })
          {
              for (int f = 0; f < conv.Kernels.Length; f++)
              {
                  for (int ki = 0; ki < conv.Kernels[0].Length; ki++)
                  for (int kj = 0; kj < conv.Kernels[0][0].Length; kj++)
                  for (int c = 0; c < conv.Kernels[0][0][0].Length; c++)
                      Check(conv.dKernels[f][ki][kj][c], ref conv.Kernels[f][ki][kj][c], $"convK[{f}][{ki}][{kj}][{c}]");
                  Check(conv.dBiases[f], ref conv.Biases[f], $"convB[{f}]");
              }
          }

          // dense hidden + output: weights
          foreach (var (layer, name) in new[] { (denseH, "H"), (denseO, "O") })
              for (int i = 0; i < layer.Count; i++)
              for (int j = 0; j < layer[i].Weights.Length; j++)
                  Check(layer[i].gradientsW[j], ref layer[i].Weights[j], $"dense{name}[{i}]W[{j}]");

          Assert.True(maxRel < 1e-4, $"maxRel={maxRel} @ {worst}");
      }
  }

  // ==========================================================================================
  //  Coverage gaps (audit 2026-08-03)
  // ------------------------------------------------------------------------------------------
  //  The earlier suite stayed green despite real bugs, because it systematically never
  //  exercised these paths:
  //    1) the last layer was Dense in ALL tests (Conv/RNN tests drove the layer backward
  //       BY HAND, simulating a middle layer),
  //    2) softmax was only tested fused with CrossEntropy,
  //    3) save/load had no round-trip,
  //    4) nobody ran two TrainLoops in a row in one process.
  // ==========================================================================================

  public class OutputLayerGradientTests : GradientCheckTestBase
  {
      // Previously the backward pass computed the per-layer gradient step only for layers
      // count-2..0, so the output layer never filled the buffers (Conv.dOutput / RNN.gImm) its
      // weight accumulation read from → an output Conv/RNN silently did NOT learn, with no
      // exception and the loss still being reported. Now every layer runs the same Backward.

      [Fact]
      public void Conv_as_output_layer_kernel_grads_match_numeric()
      {
          var input = new double[3, 3, 1];
          for (int i = 0; i < 3; i++)
          for (int j = 0; j < 3; j++)
              input[i, j, 0] = (i * 3 + j) * 0.1 + 0.05;

          var conv = new Conv(1, 2, new Linear(), "valid");                 // 3×3×1 → 2×2×1
          var model = new NeuralNetwork(conv, new SGD(0.0), new MSE());
          model.Layers.SetInputShape(new int[] { 3, 3, 1 });

          var x = new Tensor(input);
          var target = new Tensor(new double[] { 0.2, -0.1, 0.4, 0.3 });

          model.Train.Fit(x, target);                                       // ONLY through the public API
          var analytic = conv.dKernels[0].Select(r => r.Select(c => (double[])c.Clone()).ToArray()).ToArray();

          var loss = model.Loss;
          Func<double> lossAt = () => loss.CalculateAndGetLoss(model.GetResults(x).Data, target.Data);

          double maxRel = 0;
          for (int ki = 0; ki < 2; ki++)
          for (int kj = 0; kj < 2; kj++)
          {
              double num = Numeric.ParamGrad(lossAt, ref conv.Kernels[0][ki][kj][0]);
              maxRel = Math.Max(maxRel, Numeric.RelErr(analytic[ki][kj][0], num));
          }

          Assert.True(maxRel < 1e-5, $"maxRel={maxRel} — the output Conv did not receive a gradient");
      }

      [Fact]
      public void RNN_as_output_layer_weight_grads_match_numeric()
      {
          // sequence of length 1 → h(0)=0, the recurrent weight trivially has gradient 0 (0 == 0);
          // we test the input weights + bias, which previously were never filled at all.
          var model = new NeuralNetwork(new RNN(2, new Tanh()), new SGD(0.0), new MSE());
          model.Layers.SetInputShape(new int[] { 2 });

          var x = new Tensor(new double[] { 0.5, -0.3 });
          var t = new Tensor(new double[] { 0.1, -0.2 });

          model.ResetSequence();
          model.Train.Fit(x, t);

          var neurons = ((RNN)model.Layers.Layers[0]).Neurons;
          var analytic = neurons.Select(n => (double[])n.gradientsW.Clone()).ToList();

          var loss = model.Loss;
          Func<double> lossAt = () =>
          {
              model.ResetSequence();
              return loss.CalculateAndGetLoss(model.GetResults(x).Data, t.Data);
          };

          double maxRel = 0; int wi = -1, wj = -1;
          for (int i = 0; i < neurons.Count; i++)
          for (int j = 0; j < neurons[i].Weights.Length; j++)
          {
              double num = Numeric.ParamGrad(lossAt, ref neurons[i].Weights[j]);
              double rel = Numeric.RelErr(analytic[i][j], num);
              if (rel > maxRel) { maxRel = rel; wi = i; wj = j; }
          }

          Assert.True(maxRel < 1e-5,
              $"maxRel={maxRel} at neuron {wi} weight {wj} — the output RNN did not receive a gradient");
      }

      [Fact]
      public void RNN_as_output_layer_actually_moves_weights()
      {
          // Crude but unambiguous proof of the same bug: after 50 steps and UpdateParams
          // the weights must be DIFFERENT. With the bug they stayed bit-identical.
          var model = new NeuralNetwork(new RNN(2, new Tanh()), new SGD(0.5), new MSE());
          model.Layers.SetInputShape(new int[] { 2 });

          var neurons = ((RNN)model.Layers.Layers[0]).Neurons;
          double[] before = (double[])neurons[0].Weights.Clone();

          for (int i = 0; i < 50; i++)
              model.Train.Fit(new Tensor(new double[] { 1, 1 }), new Tensor(new double[] { 0.5, -0.5 }));
          model.Train.UpdateParams();

          Assert.NotEqual(before, neurons[0].Weights);
      }
  }

  public class SoftmaxWithNonCrossEntropyTests : GradientCheckTestBase
  {
      // Softmax.Derivative used to return only the DIAGONAL of the Jacobian (via a stateful
      // counter `a`). The fused softmax+CE path does not use it, but for any other loss the
      // gradient was wrong (measured analytic / numeric ≈ 0.539). Now softmax goes through
      // BackwardForLayer with the full Jacobian-vector product.
      [Fact]
      public void Softmax_with_MSE_weight_grads_match_numeric()
      {
          var model = new NeuralNetwork(new Dense(3, new Softmax()), new SGD(0.0), new MSE());
          model.Layers.SetInputShape(new int[] { 3 });

          var x = new Tensor(new double[] { 0.5, -0.3, 0.8 });
          var t = new Tensor(new double[] { 0.0, 1.0, 0.0 });

          model.Train.Fit(x, t);

          var neurons = ((Dense)model.Layers.Layers[0]).Neurons;
          var analytic = neurons.Select(n => (double[])n.gradientsW.Clone()).ToList();

          var loss = model.Loss;
          Func<double> lossAt = () => loss.CalculateAndGetLoss(model.GetResults(x).Data, t.Data);

          double maxRel = 0; int wi = -1, wj = -1;
          for (int i = 0; i < neurons.Count; i++)
          for (int j = 0; j < neurons[i].Weights.Length; j++)
          {
              double num = Numeric.ParamGrad(lossAt, ref neurons[i].Weights[j]);
              double rel = Numeric.RelErr(analytic[i][j], num);
              if (rel > maxRel) { maxRel = rel; wi = i; wj = j; }
          }

          Assert.True(maxRel < 1e-5,
              $"maxRel={maxRel} at neuron {wi} weight {wj} — softmax outside the fusion with CE");
      }
  }
