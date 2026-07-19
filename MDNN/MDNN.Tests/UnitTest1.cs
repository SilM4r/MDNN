using System;
  using Xunit;
  using My_DNN;
  using My_DNN.Layers;
  using My_DNN.Activation_functions;
  using My_DNN.Layers.classes;
  using My_DNN.Optimizers;
  using My_DNN.Loss_functions;
  using System.Linq;

  // statický stav v knihovně → testy nesmí běžet paralelně
  [assembly: CollectionBehavior(DisableTestParallelization = true)]

  namespace MDNN.Tests;

  // ---------- infrastruktura ----------

  public static class TestGlobals
  {
      public static void Reset()
      {
          GeneralNeuralNetworkSettings.rnd = new Random(42);   // determinismus (init vah)
          GeneralNeuralNetworkSettings.optimizer = new SGD(0.0001);
          GeneralNeuralNetworkSettings.default_output_activation_func = new Linear();
          GeneralNeuralNetworkSettings.default_hidden_layers_activation_func = new ReLu();
      }
  }

  public static class Numeric
  {
      // derivace funkce podle jejího argumentu (pro aktivace/loss)
      public static double CentralDiff(Func<double, double> f, double x, double eps = 1e-5)
          => (f(x + eps) - f(x - eps)) / (2 * eps);

      // numerický gradient parametru: perturbuj slot, spusť skalární loss, vrať zpět
      public static double ParamGrad(Func<double> lossAt, ref double param, double eps = 1e-5)
      {
          double orig = param;
          param = orig + eps; double lp = lossAt();
          param = orig - eps; double lm = lossAt();
          param = orig;                                  // RESTORE
          return (lp - lm) / (2 * eps);
      }

      public static double RelErr(double a, double b)
          => Math.Abs(a - b) / (Math.Abs(a) + Math.Abs(b) + 1e-12);
      
      public static double NumJac(Softmax sm, double[] z, int i, int k, double eps = 1e-5)
      {
          var zp = (double[])z.Clone(); zp[k] += eps;
          var zm = (double[])z.Clone(); zm[k] -= eps;
          return (sm.ApplyToLayer(zp)[i] - sm.ApplyToLayer(zm)[i]) / (2 * eps);
      }
  }

  public abstract class GradientCheckTestBase
  {
      protected GradientCheckTestBase() => TestGlobals.Reset();   // xUnit: běží před každým testem
  }

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

          double[] dInAnalytic = pool.CalculateLayerGradients(new Tensor(g), null!).Data;

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
      public void Conv_input_gradient_matches_numeric(
          int h, int w, int c, int k, int f, string pad, string actName)
      {
          var (input, conv, gFlat, gTensor) = ConvSetup(h, w, c,k, f, pad, actName);

          // ANALYTIC dL/dinput (3D větev; next_layer se nepoužije)
          double[] dInAnalytic = conv.CalculateLayerGradients(gTensor, null!).Data;

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
      public void Conv_kernel_gradient_matches_numeric(int h, int w, int c, int k, int f, string pad, string actName)
      {
          var (input, conv, gFlat, gTensor) = ConvSetup(h, w, c,k, f, pad, actName);

          // ANALYTIC: CalculateLayerGradients nastaví dOutput, BackPropagation z něj napočítá dKernels
          conv.CalculateLayerGradients(gTensor, null!);
          conv.BackPropagation(gTensor);
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
      public void Conv_bias_gradient_matches_numeric(int h, int w, int c, int k, int f, string pad, string actName)
      {
          var (input, conv, gFlat, gTensor) = ConvSetup(h, w, c,k, f, pad, actName);

          conv.CalculateLayerGradients(gTensor, null!);
          conv.BackPropagation(gTensor);
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
  }
  
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
          Assert.IsType<SGD>(((Dense)a.Layers.Layers[0]).Neurons[0].optimizer);
          Assert.IsType<Adam>(((Dense)b.Layers.Layers[0]).Neurons[0].optimizer);

          // po trénování obou: každý si drží svou architekturu i loss (nešlapou si po sobě)
          Assert.Equal(1, a.GetResults(new Tensor(new double[] { 0.5 })).Data.Length);
          Assert.Equal(3, b.GetResults(new Tensor(new double[] { 0.5, 0.3, 0.2 })).Data.Length);
          Assert.IsType<MSE>(a.Loss);
          Assert.IsType<CrossEntropy>(b.Loss);
      }
  }