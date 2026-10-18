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

  // ---------- ukázkový test: derivace aktivace (krok 3a) — běží HNED ----------

  public class ActivationDerivativeTests : GradientCheckTestBase
  {
      [Theory]
      [InlineData(-2.0)]
      [InlineData(-0.5)]
      [InlineData(0.3)]
      [InlineData(1.7)]
      public void Tanh_derivative_matches_numeric(double x)
      {
          Activation_func act = new Tanh();

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
          Activation_func act = new Sigmoid();

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
          Activation_func act = new Linear();

          double analytic = act.Derivative(x);
          double numeric  = Numeric.CentralDiff(act.Apply, x);

          Assert.True(Numeric.RelErr(analytic, numeric) < 1e-6,
              $"x={x}: analytic={analytic}, numeric={numeric}");
      }
      
      [Theory]
      // POZOR: netestovat x=0 — ReLU má v nule kink, numerika a analytika se rozejdou
      [InlineData(-2.0)]
      [InlineData(-0.5)]
      [InlineData(0.3)]
      [InlineData(1.7)]
      public void ReLu_derivative_matches_numeric(double x)
      {
          Activation_func act = new ReLu();

          double analytic = act.Derivative(x);
          double numeric  = Numeric.CentralDiff(act.Apply, x);

          Assert.True(Numeric.RelErr(analytic, numeric) < 1e-6,
              $"x={x}: analytic={analytic}, numeric={numeric}");
      }
      
      [Theory]
      // POZOR: netestovat x=0 — ReLU má v nule kink, numerika a analytika se rozejdou
      [InlineData(-2.0)]
      [InlineData(-0.5)]
      [InlineData(0.3)]
      [InlineData(1.7)]
      public void Leak_ReLu_derivative_matches_numeric(double x)
      {
          Activation_func act = new Leak_ReLu();

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
          LayerActivationFunc act = new Softmax();
          
          var s = act.ApplyToLayer(x);
          Assert.Equal(1.0, s.Sum(), 10);
          Assert.All(s, v => Assert.InRange(v, 0, 1));
      }
      
      [Theory]
      [InlineData(new[] {1.0, 2.0, 3.0})]
      [InlineData(new[] {1000.0, 1001.0})]
      public void Softmax_Jacobian_formula_ApplyToLayer_matches_numeric(double[] x)
      {
          LayerActivationFunc act = new Softmax();
          
          var s = act.ApplyToLayer(x);
          for (int i = 0; i < x.Length; i++)          // který výstup
          for (int k = 0; k < x.Length; k++)      // který vstup šťouchám
          {
              double analytic = s[i] * ((i == k ? 1.0 : 0.0) - s[k]);  // vzorec
              double numeric  = Numeric.NumJac((Softmax)act, x, i, k);                    // změřeno
              Assert.True(Numeric.RelErr(analytic, numeric) < 1e-5,
                  $"J[{i},{k}]: analytic={analytic}, numeric={numeric}");
          }
      }
      
      [Fact]
      public void Softmax_crossentropy_weight_grads_match_numeric()
      {
          var x      = new Tensor(new double[] { 0.5, -0.3, 0.8 });   // 3 vstupy
          var target = new Tensor(new double[] { 0.0, 1.0, 0.0 });    // one-hot: třída 1

          // 3 neurony → 3 třídy, softmax výstup, CrossEntropy loss
          var model = new My_DNN.MDNN(new Dense(3, new Softmax()), new SGD(0.01), new CrossEntropy());

          // ---- ANALYTIC: co si myslí knihovna ----
          model.Train.Fit(x, target);                 // forward + backprop, NEvolat UpdateParams
          var neurons = ((Dense)model.Layers.Layers[0]).Neurons;
          var analytic = neurons.Select(n => (double[])n.gradientsW.Clone()).ToList();

          // ---- NUMERIC: co se doopravdy děje s loss ----
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
              $"maxRel={maxRel} u neuronu {worstI} váhy {worstJ}: " +
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
          var x      = new Tensor(new double[] { 0.5, -0.3, 0.8 });   // 3 vstupy
          var target = new Tensor(new double[] { 0.0, 1.0, 0.0 });    // one-hot: třída 1
          
          var model = new My_DNN.MDNN(new Dense(3, new Tanh()), new SGD(0.01), new MSE());

          // ---- ANALYTIC: co si myslí knihovna ----
          model.Train.Fit(x, target);                 // forward + backprop, NEvolat UpdateParams
          var neurons = ((Dense)model.Layers.Layers[0]).Neurons;
          var analytic = neurons.Select(n => (double[])n.gradientsW.Clone()).ToList();

          // ---- NUMERIC: co se doopravdy děje s loss ----
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
              $"maxRel={maxRel} u neuronu {worstI} váhy {worstJ}: " +
              $"analytic={analytic[worstI][worstJ]}");
      }
      
      [Fact]
      public void Dense_multi_layer_test()
      {
          var x      = new Tensor(new double[] { 0.5, -0.3, 0.8, 0.1});   // 4 vstupy
          var target = new Tensor(new double[] { 0.0, 1.0});    
          
          var model = new My_DNN.MDNN(new Dense(2, new Tanh()), new SGD(0.01), new MSE());
          model.Layers.Add(new Dense(4, new Tanh()));

          // ---- ANALYTIC: co si myslí knihovna ----
          model.Train.Fit(x, target);                 // forward + backprop, NEvolat UpdateParams
          var neurons = new List<Neuron>();
          neurons.AddRange(((Dense)model.Layers.Layers[0]).Neurons);
          neurons.AddRange(((Dense)model.Layers.Layers[1]).Neurons);
          
          var analytic = neurons.Select(n => (double[])n.gradientsW.Clone()).ToList();

          // ---- NUMERIC: co se doopravdy děje s loss ----
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
              $"maxRel={maxRel} u neuronu {worstI} váhy {worstJ}: " +
              $"analytic={analytic[worstI][worstJ]}");
      }

      [Fact]
      public void Dense_two_hidden_layers_build_and_forward()
      {
          // Regrese: DVĚ skryté Dense za sebou + výstupní = [Dense, Dense, Dense].
          // Před opravou SetInputSizeForFirstLayer se prostřední (neuron-po-neuronu)
          // NEpřestavěla a zůstala s placeholder neurony (0 vah) → IndexOutOfRange
          // při forwardu. (Dense_multi_layer_test má jen [skrytá, výstupní] → bug nechytil.)
          var model = new My_DNN.MDNN(new Dense(2, new Linear()), new SGD(0.01), new MSE());
          model.Layers.Add(new Dense(3, new ReLu()));
          model.Layers.Add(new Dense(3, new ReLu()));

          var x = new Tensor(new double[] { 0.5, -0.3, 0.8, 0.1 });   // 4 vstupy

          var output = model.GetResults(x);   // spustí SetInputSizeForFirstLayer; nesmí spadnout

          // každá vrstva má vstupní váhy = výstup předchozí (ne placeholder 0)
          Assert.Equal(4, ((Dense)model.Layers.Layers[0]).Neurons[0].Weights.Length);  // 1. skrytá: 4 vstupy
          Assert.Equal(3, ((Dense)model.Layers.Layers[1]).Neurons[0].Weights.Length);  // 2. skrytá: 3 (výstup 1.)
          Assert.Equal(3, ((Dense)model.Layers.Layers[2]).Neurons[0].Weights.Length);  // výstupní: 3
          Assert.Equal(2, output.Data.Length);                                         // 2 výstupy
      }

      [Fact]
      public void Conv_maxpool_dense_dense_pipeline_wires_correctly()
      {
          // Napojení mezi RŮZNÝMI typy vrstev: Conv → MaxPool → Dense → Dense → Dense(out).
          // Ověřuje wiring napříč změnou typu (3D→3D→flatten→neuron) I neuron-po-neuronu
          // UPROSTŘED pipeline (Dense→Dense) — ten na staré logice padal (placeholder 0 vah).
          var model = new My_DNN.MDNN(new Dense(2, new Linear()), new SGD(0.01), new MSE());
          model.Layers.Add(new Conv(2, 3, new ReLu(), "valid"));   // Layers[0]
          model.Layers.Add(new MaxPool(2));                        // Layers[1]
          model.Layers.Add(new Dense(8, new ReLu()));              // Layers[2]  (po MaxPoolu)
          model.Layers.Add(new Dense(4, new ReLu()));              // Layers[3]  (neuron-po-neuronu!)
          // Layers[4] = výstupní Dense(2)

          var input = new double[8, 8, 1];                          // vstup 8×8×1
          for (int i = 0; i < 8; i++)
          for (int j = 0; j < 8; j++)
              input[i, j, 0] = (i * 8 + j) * 0.01;

          var output = model.GetResults(new Tensor(input));   // spustí wiring; nesmí spadnout

          // Conv valid: 8-3+1=6 → [6,6,2];  MaxPool/2: [3,3,2];  Dense flatten = 3*3*2 = 18
          Assert.Equal(new int[] { 6, 6, 2 }, model.Layers.Layers[0].Output_size_and_shape);
          Assert.Equal(new int[] { 3, 3, 2 }, model.Layers.Layers[1].Output_size_and_shape);
          Assert.Equal(18, ((Dense)model.Layers.Layers[2]).Neurons[0].Weights.Length);  // MaxPool → Dense (flatten 3*3*2)
          Assert.Equal(8,  ((Dense)model.Layers.Layers[3]).Neurons[0].Weights.Length);  // Dense → Dense (uprostřed)
          Assert.Equal(4,  ((Dense)model.Layers.Layers[4]).Neurons[0].Weights.Length);  // Dense → Dense (výstup)
          Assert.Equal(2, output.Data.Length);
      }

      [Theory]
      [InlineData(4, 4, 1, 2)]
      [InlineData(6, 6, 1, 3)]
      [InlineData(4, 4, 2, 2)]
      [InlineData(10, 10, 3, 5)]
      public void MaxPool_input_gradient_matches_numeric(int x, int y, int z, int poolsize)
      {
          // distinktní hodnoty přes CELÝ tenzor = flat index (zaručeně unikátní → žádné remízy)
          var input = new double[x, y, z];
          for (int i = 0; i < x; i++)
          for (int j = 0; j < y; j++)
          for (int c = 0; c < z; c++)
              input[i, j, c] = i * (y * z) + j * z + c;

          var pool = new MaxPool(poolsize);
          pool.LayerAdjustment(null, new int[] { x, y, z });
          pool.FeedForward(new Tensor(input));

          // g má tvar VÝSTUPU, ne natvrdo 2×2×1
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
              int flat = i * (y * z) + j * z + c;   // stejný stride jako dInAnalytic
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
          pool.LayerAdjustment(null, new[]{4,4,1});
          var o = (double[,,])pool.FeedForward(new Tensor(input)).GetOriginalData();
          Assert.Equal(5.0, o[0,0,0]);   // max okna {0,1,4,5}
          Assert.Equal(15.0, o[1,1,0]);  // max okna {10,11,14,15}
      }
      
      [Fact]
      public void RNN_weight_grads_match_numeric_sequence()
      {
          // sekvence 2 kroků, vstup dim 2
          var xs = new[] {
              new Tensor(new double[]{ 0.5, -0.3 }),
              new Tensor(new double[]{ 0.2,  0.4 }),
          };
          var ts = new[] {
              new Tensor(new double[]{ 0.1, -0.2 }),
              new Tensor(new double[]{ 0.3,  0.0 }),
          };

          var model = new My_DNN.MDNN(new Dense(2, new Tanh()), new SGD(0.01), new MSE());
          model.Layers.Add(new RNN(2, new Tanh()));      // Layers[0]=RNN, Layers[1]=Dense
          var loss = model.Loss;

          // ANALYTIC: projeď sekvenci, akumuluj grady (ResetSequence, per-krok Fit, BEZ UpdateParams)
          model.ResetSequence();
          for (int t = 0; t < xs.Length; t++) model.Train.Fit(xs[t], ts[t]);
          var rnn = ((RNN)model.Layers.Layers[0]).Neurons;
          var analytic = rnn.Select(n => (double[])n.gradientsW.Clone()).ToList();

          // NUMERIC: loss CELÉ sekvence jako funkce vah (každé volání resetuje stav!)
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
          for (int j = 0; j < rnn[i].Weights.Length; j++)   // poslední váha = rekurentní
          {
              double num = Numeric.ParamGrad(seqLoss, ref rnn[i].Weights[j]);
              double rel = Numeric.RelErr(analytic[i][j], num);
              if (rel > maxRel) { maxRel = rel; wi=i; wj=j; }
          }

          Assert.True(maxRel < 1e-5, $"maxRel={maxRel} u neuronu {wi} váhy {wj}");
      }

      [Fact]
      public void RNN_weight_grads_match_numeric_singlestep()
      {
          // sekvence délky 1 → recurrence se neprojeví (h(0)=0), ověří vstupní váhy + bias.
          // Rekurentní váha dostane triviálně 0 (její vstup h(0)=0) → 0 == 0.
          var xs = new[] { new Tensor(new double[]{ 0.5, -0.3 }) };
          var ts = new[] { new Tensor(new double[]{ 0.1, -0.2 }) };

          var model = new My_DNN.MDNN(new Dense(2, new Tanh()), new SGD(0.01), new MSE());
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

          Assert.True(maxRel < 1e-5, $"maxRel={maxRel} u neuronu {wi} váhy {wj}");
      }

      // ---------- Conv gradient checks (3×3×1, kernel 2×2, 1 filtr, Linear, valid → 2×2×1) ----------

      // sdílený setup: vstup, conv, g

      private static Func<double> ConvLoss(Conv conv, double[,,] input, double[] gFlat) => () =>
      {
          double[] o = conv.FeedForward(new Tensor(input)).Data;
          double s = 0;
          for (int k = 0; k < o.Length; k++) s += gFlat[k] * o[k];
          return s;                                          // L = Σ g·output
      };
      
      private static Activation_func Act(string n) => n == "tanh" ? new Tanh() : new Linear();

      private static (double[,,] input, Conv conv, double[] gFlat, Tensor gTensor) ConvSetup(
          int h, int w, int c, int kernel, int filters, string pad, string actName)
      {
          // malé distinktní hodnoty → Tanh nesaturuje (jinak act'≈0 a numerika zašumí)
          var input = new double[h, w, c];
          for (int i = 0; i < h; i++)
          for (int j = 0; j < w; j++)
          for (int cc = 0; cc < c; cc++)
              input[i, j, cc] = ((i * w + j) * c + cc) * 0.01 + 0.02;

          var conv = new Conv(filters, kernel, Act(actName), pad);
          conv.LayerAdjustment(null, new int[] { h, w, c });
          conv.FeedForward(new Tensor(input));

          // tvar výstupu: valid = h-k+1, same = h
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
      [InlineData(3, 3, 1, 2, 1, "valid", "linear")]   // baseline (jako dosud)
      [InlineData(4, 4, 2, 2, 1, "valid", "tanh")]      // multi-channel + tanh
      [InlineData(4, 4, 1, 2, 3, "valid", "tanh")]      // multi-filter
      [InlineData(5, 5, 2, 3, 2, "valid", "tanh")]      // channel+filter+kernel3
      [InlineData(5, 5, 1, 3, 2, "same",  "tanh")]      // same padding (lichý kernel!)
      [InlineData(5, 5, 1, 2, 2, "same",  "tanh")]      // same padding SUDÝ kernel (asymetrický pad)
      public void Conv_input_gradient_matches_numeric(
          int h, int w, int c, int k, int f, string pad, string actName)
      {
          var (input, conv, gFlat, gTensor) = ConvSetup(h, w, c,k, f, pad, actName);

          // ANALYTIC dL/dinput (3D větev; next_layer se nepoužije)
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
      [InlineData(3, 3, 1, 2, 1, "valid", "linear")]   // baseline (jako dosud)
      [InlineData(4, 4, 2, 2, 1, "valid", "tanh")]      // multi-channel + tanh
      [InlineData(4, 4, 1, 2, 3, "valid", "tanh")]      // multi-filter
      [InlineData(5, 5, 2, 3, 2, "valid", "tanh")]      // channel+filter+kernel3
      [InlineData(5, 5, 1, 3, 2, "same",  "tanh")]      // same padding (lichý kernel!)
      [InlineData(5, 5, 1, 2, 2, "same",  "tanh")]      // same padding SUDÝ kernel (asymetrický pad)
      public void Conv_kernel_gradient_matches_numeric(int h, int w, int c, int k, int f, string pad, string actName)
      {
          var (input, conv, gFlat, gTensor) = ConvSetup(h, w, c,k, f, pad, actName);

          // ANALYTIC: Backward vezme ∂L/∂output, naakumuluje dKernels/dBiases a vrátí ∂L/∂input
          conv.Backward(gTensor);
          var dK = conv.dKernels;
          var kern = conv.Kernel;

          var lossAt = ConvLoss(conv, input, gFlat);

          double maxRel = 0;
          for (int fi = 0; fi < kern.Length; fi++)                 // přes filtry
          for (int ki = 0; ki < kern[0].Length; ki++)
          for (int kj = 0; kj < kern[0][0].Length; kj++)
          for (int cj = 0; cj < kern[0][0][0].Length; cj++)
          {
              double analytic = dK[fi][ki][kj][cj];
              double num = Numeric.ParamGrad(lossAt, ref conv.Kernel[fi][ki][kj][cj]);
              maxRel = Math.Max(maxRel, Numeric.RelErr(analytic, num));
          }
          Assert.True(maxRel < 1e-5, $"maxRel={maxRel}");
      }

      [Theory]
      [InlineData(3, 3, 1, 2, 1, "valid", "linear")]   // baseline (jako dosud)
      [InlineData(4, 4, 2, 2, 1, "valid", "tanh")]      // multi-channel + tanh
      [InlineData(4, 4, 1, 2, 3, "valid", "tanh")]      // multi-filter
      [InlineData(5, 5, 2, 3, 2, "valid", "tanh")]      // channel+filter+kernel3
      [InlineData(5, 5, 1, 3, 2, "same",  "tanh")]      // same padding (lichý kernel!)
      [InlineData(5, 5, 1, 2, 2, "same",  "tanh")]      // same padding SUDÝ kernel (asymetrický pad)
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
          var input = new double[3, 4];                 // 2D, H≠W → spustí reshape větev
          for (int i = 0; i < 3; i++)
          for (int j = 0; j < 4; j++)
              input[i, j] = i * 4 + j + 0.1;

          var conv = new Conv(1, 2, new Linear(), "valid");
          conv.LayerAdjustment(null, new int[] { 3, 4, 1 });

          var outp = conv.FeedForward(new Tensor(input));   // před opravou: ArgumentException (9 ≠ 12 prvků)

          // valid, kernel 2 → výstup (3-2+1)×(4-2+1) = 2×3
          Assert.Equal(2, outp.Shape[0]);
          Assert.Equal(3, outp.Shape[1]);
      }

      [Fact]
      public void Full_CNN_stack_end_to_end_gradients_match_numeric()
      {
          // End-to-end gradient check CELÉ CNN pipeline (mini verze modelu z Program.cs):
          // Conv → MaxPool → Conv → MaxPool → Dense → Dense(out).
          // Jednotlivé vrstvy jsou gradient-checknuté ZVLÁŠŤ (Conv_*_gradient, MaxPool_*),
          // ALE tohle ověřuje WIRING backpropu skrz CELÝ řetězec: gradient tekoucí
          // z MaxPoolu zpět do Convu a přes DVĚ conv vrstvy za sebou. Přesně to, co nejde
          // ověřit čtením kódu a co by jinak shodilo dlouhý MNIST běh až po hodinách.
          //
          // Tanh (ne ReLu) ve skrytých vrstvách = čistá numerika bez kinku v nule (ReLu
          // derivace je testovaná zvlášť); malý vstup → tanh nesaturuje. Tolerance 1e-4
          // (volnější než izolovaných 1e-5) kvůli hloubce řetězce + tanh-of-tanh.
          var input = new double[8, 8, 1];
          for (int i = 0; i < 8; i++)
          for (int j = 0; j < 8; j++)
              input[i, j, 0] = ((i * 8 + j) * 0.005) + 0.01;
          var x = new Tensor(input);
          var target = new Tensor(new double[] { 0.3, -0.2 });

          var model = new My_DNN.MDNN(new Dense(2, new Linear()), new SGD(0.01), new MSE());
          model.Layers.Add(new Conv(2, 3, new Tanh(), "valid"));   // [0]  8×8×1 → 6×6×2
          model.Layers.Add(new MaxPool(2));                        // [1]  → 3×3×2
          model.Layers.Add(new Conv(3, 2, new Tanh(), "valid"));   // [2]  → 2×2×3
          model.Layers.Add(new MaxPool(2));                        // [3]  → 1×1×3
          model.Layers.Add(new Dense(4, new Tanh()));              // [4]  flatten 3 → 4
          // [5] = výstupní Dense(2, Linear)

          // ---- ANALYTIC: jeden Fit (forward + backprop CELÝM řetězcem), BEZ UpdateParams ----
          model.Train.Fit(x, target);

          var conv1 = (Conv)model.Layers.Layers[0];
          var conv2 = (Conv)model.Layers.Layers[2];
          var denseH = ((Dense)model.Layers.Layers[4]).Neurons;
          var denseO = ((Dense)model.Layers.Layers[5]).Neurons;

          // sanity: řetězec se opravdu propojil (ne placeholder 0 vah)
          Assert.Equal(new int[] { 6, 6, 2 }, conv1.Output_size_and_shape);
          Assert.Equal(new int[] { 2, 2, 3 }, conv2.Output_size_and_shape);
          Assert.Equal(3, denseH[0].Weights.Length);   // flatten 1*1*3

          // ---- NUMERIC: co doopravdy dělá loss CELÉHO modelu ----
          // (forward nikdy nepřepisuje gradientní buffery dKernels/dBiases/gradientsW,
          //  takže je čteme inline i během perturbace parametrů)
          Func<double> lossAt = () =>
              model.Loss.CalculateAndGetLoss(model.GetResults(x).Data, target.Data);

          double maxRel = 0; string worst = "";
          void Check(double analytic, ref double param, string label)
          {
              double num = Numeric.ParamGrad(lossAt, ref param);
              double rel = Numeric.RelErr(analytic, num);
              if (rel > maxRel) { maxRel = rel; worst = $"{label}: analytic={analytic}, numeric={num}"; }
          }

          // conv1 + conv2: kernely i biasy (gradient přišel skrz zbytek řetězce)
          foreach (var conv in new[] { conv1, conv2 })
          {
              for (int f = 0; f < conv.Kernel.Length; f++)
              {
                  for (int ki = 0; ki < conv.Kernel[0].Length; ki++)
                  for (int kj = 0; kj < conv.Kernel[0][0].Length; kj++)
                  for (int c = 0; c < conv.Kernel[0][0][0].Length; c++)
                      Check(conv.dKernels[f][ki][kj][c], ref conv.Kernel[f][ki][kj][c], $"convK[{f}][{ki}][{kj}][{c}]");
                  Check(conv.dBiases[f], ref conv.Biases[f], $"convB[{f}]");
              }
          }

          // dense skrytá + výstupní: váhy
          foreach (var (layer, name) in new[] { (denseH, "H"), (denseO, "O") })
              for (int i = 0; i < layer.Count; i++)
              for (int j = 0; j < layer[i].Weights.Length; j++)
                  Check(layer[i].gradientsW[j], ref layer[i].Weights[j], $"dense{name}[{i}]W[{j}]");

          Assert.True(maxRel < 1e-4, $"maxRel={maxRel} @ {worst}");
      }
  }

  // ==========================================================================================
  //  Fáze 1b — díry v coveragi (audit 2026-08-03)
  // ------------------------------------------------------------------------------------------
  //  Dosavadní sada je zelená i s chybami z Fáze 2c, protože systematicky nikdy neprojde
  //  tyhle cesty:
  //    1) poslední vrstva je ve VŠECH testech Dense (Conv/RNN testy volají
  //       CalculateLayerGradients RUČNĚ, čímž simulují prostřední vrstvu),
  //    2) softmax se testuje jen fúzovaný s CrossEntropy,
  //    3) save/load nemá round-trip,
  //    4) nikdo nepustil dva TrainLoopy za sebou v jednom procesu.
  // ==========================================================================================

  public class OutputLayerGradientTests : GradientCheckTestBase
  {
      // 2c-1: Gradient.GetGradients volá CalculateLayerGradients jen pro vrstvy count-2..0.
      // Poslední vrstva ji nedostane — jenže právě ta plní Conv.dOutput / RNN.gImm, ze kterých
      // Conv.BackPropagation / RNN.BackPropagation počítají (svůj argument ignorují).
      // → Výstupní Conv/RNN se tiše NEUČÍ. Žádná výjimka, loss se počítá dál.

      [Fact]
      public void Conv_as_output_layer_kernel_grads_match_numeric()
      {
          var input = new double[3, 3, 1];
          for (int i = 0; i < 3; i++)
          for (int j = 0; j < 3; j++)
              input[i, j, 0] = (i * 3 + j) * 0.1 + 0.05;

          var conv = new Conv(1, 2, new Linear(), "valid");                 // 3×3×1 → 2×2×1
          var model = new My_DNN.MDNN(conv, new SGD(0.0), new MSE());
          model.Layers.SetInputSizeForFirstLayer(new int[] { 3, 3, 1 });

          var x = new Tensor(input);
          var target = new Tensor(new double[] { 0.2, -0.1, 0.4, 0.3 });

          model.Train.Fit(x, target);                                       // POUZE přes veřejné API
          var analytic = conv.dKernels[0].Select(r => r.Select(c => (double[])c.Clone()).ToArray()).ToArray();

          var loss = model.Loss;
          Func<double> lossAt = () => loss.CalculateAndGetLoss(model.GetResults(x).Data, target.Data);

          double maxRel = 0;
          for (int ki = 0; ki < 2; ki++)
          for (int kj = 0; kj < 2; kj++)
          {
              double num = Numeric.ParamGrad(lossAt, ref conv.Kernel[0][ki][kj][0]);
              maxRel = Math.Max(maxRel, Numeric.RelErr(analytic[ki][kj][0], num));
          }

          Assert.True(maxRel < 1e-5, $"maxRel={maxRel} — výstupní Conv nedostal gradient (2c-1)");
      }

      [Fact]
      public void RNN_as_output_layer_weight_grads_match_numeric()
      {
          // sekvence délky 1 → h(0)=0, rekurentní váha má gradient triviálně 0 (0 == 0);
          // testujeme vstupní váhy + bias, které se dnes vůbec nenaplní.
          var model = new My_DNN.MDNN(new RNN(2, new Tanh()), new SGD(0.0), new MSE());
          model.Layers.SetInputSizeForFirstLayer(new int[] { 2 });

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
              $"maxRel={maxRel} u neuronu {wi} váhy {wj} — výstupní RNN nedostal gradient (2c-1)");
      }

      [Fact]
      public void RNN_as_output_layer_actually_moves_weights()
      {
          // Hrubý, ale nezpochybnitelný důkaz stejné chyby: po 50 krocích a UpdateParams
          // musí být váhy JINÉ. Dnes jsou bit-identické.
          var model = new My_DNN.MDNN(new RNN(2, new Tanh()), new SGD(0.5), new MSE());
          model.Layers.SetInputSizeForFirstLayer(new int[] { 2 });

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
      // 2c-4: Softmax.Derivative vrací jen DIAGONÁLU Jacobiánu (přes stavový čítač `a`).
      // Pro fúzovanou cestu softmax+CE se neuplatní, pro jinou loss je gradient špatně.
      // Naměřeno: analytický / numerický ≈ 0,539.
      [Fact]
      public void Softmax_with_MSE_weight_grads_match_numeric()
      {
          var model = new My_DNN.MDNN(new Dense(3, new Softmax()), new SGD(0.0), new MSE());
          model.Layers.SetInputSizeForFirstLayer(new int[] { 3 });

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
              $"maxRel={maxRel} u neuronu {wi} váhy {wj} — softmax mimo fúzi s CE (2c-4)");
      }
  }
