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
      [InlineData(5, 5, 1, 2, 2, "same",  "tanh")]      // same padding SUDÝ kernel (asymetrický pad)
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
      [InlineData(5, 5, 1, 2, 2, "same",  "tanh")]      // same padding SUDÝ kernel (asymetrický pad)
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
      [InlineData(5, 5, 1, 2, 2, "same",  "tanh")]      // same padding SUDÝ kernel (asymetrický pad)
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

      [Fact]
      public void ConvertArrayToTensor_accepts_int_multidim()
          => Assert.Equal(new double[] { 1, 2 },
                          Tensor.ConvertArrayToTensor(new int[,] { { 1, 2 } }).Data);
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

  // ==========================================================================================
  //  Fáze 4 — seed / reprodukovatelnost (odemyká díru 1b-5)
  // ==========================================================================================

  public class SeedReproducibilityTests : GradientCheckTestBase
  {
      // Dřív existovaly TŘI nezávislé zdroje náhody a žádný nešel nastavit:
      // GeneralNeuralNetworkSettings.rnd (váhy), Train._rnd (výběr vzorků),
      // Random.Shared (ShuffleTensor). Stejný experiment tedy nešlo spustit dvakrát —
      // což je předpoklad pro férové porovnání dvou kandidátů v AutoML.
      private static (double[][] X, double[][] Y) Data(int n)
      {
          var X = new double[n][];
          var Y = new double[n][];
          for (int i = 0; i < n; i++)
          {
              X[i] = new double[] { i * 0.05, 1 - i * 0.03 };
              Y[i] = new double[] { i * 0.10 };
          }
          return (X, Y);
      }

      private static My_DNN.MDNN Build(int? seed)
      {
          var model = new My_DNN.MDNN(new Dense(1, new Linear()), new SGD(0.01), new MSE(), seed);
          model.Layers.Add(new Dense(4, new ReLu()));
          model.Train.ShowLossChartInTrainLoop = false;
          model.Train.ShowModelInfoIntrainLoop = false;
          model.Train.AutoSaveInTrainLoop = false;
          model.Train.TestNeuralNetworkAfterTraining = false;
          return model;
      }

      [Fact]
      public void Same_seed_gives_identical_initial_weights()
      {
          var a = Build(42);
          var b = Build(42);

          a.GetResults(new Tensor(new double[] { 1, 1 }));   // spustí wiring = inicializaci vah
          b.GetResults(new Tensor(new double[] { 1, 1 }));

          var na = ((Dense)a.Layers.Layers[0]).Neurons;
          var nb = ((Dense)b.Layers.Layers[0]).Neurons;

          for (int i = 0; i < na.Count; i++)
              Assert.Equal(na[i].Weights, nb[i].Weights);
      }

      [Fact]
      public void Different_seed_gives_different_initial_weights()
      {
          var a = Build(1);
          var b = Build(2);

          a.GetResults(new Tensor(new double[] { 1, 1 }));
          b.GetResults(new Tensor(new double[] { 1, 1 }));

          Assert.NotEqual(
              ((Dense)a.Layers.Layers[0]).Neurons[0].Weights,
              ((Dense)b.Layers.Layers[0]).Neurons[0].Weights);
      }

      [Fact]
      public void Same_seed_gives_identical_result_after_full_training()
      {
          // nejtvrdší varianta: seed musí pokrýt i míchání datasetu a výběr vzorků,
          // ne jen inicializaci vah
          var (X, Y) = Data(20);
          var probe = new Tensor(new double[] { 0.5, 0.5 });

          var a = Build(7);
          a.Train.TrainLoop(X, Y, 30, 4);
          double outA = a.GetResults(probe).Data[0];

          var b = Build(7);
          b.Train.TrainLoop(X, Y, 30, 4);
          double outB = b.GetResults(probe).Data[0];

          Assert.Equal(outA, outB);
      }

      [Fact]
      public void Same_seed_splits_dataset_identically()
      {
          var (X, Y) = Data(20);

          var a = Build(7);
          a.Train.TrainLoop(X, Y, 5);
          double[] validA = a.Train.ValidDataInputs!.Data;

          var b = Build(7);
          b.Train.TrainLoop(X, Y, 5);

          Assert.Equal(validA, b.Train.ValidDataInputs!.Data);   // shodné míchání → shodný split
      }

      [Fact]
      public void Without_seed_behaviour_is_unchanged()
      {
          // zpětná kompatibilita: bez seedu se pořád tahá ze sdíleného globálního
          // generátoru, takže dva modely NEmají stejné váhy
          var a = Build(null);
          var b = Build(null);

          a.GetResults(new Tensor(new double[] { 1, 1 }));
          b.GetResults(new Tensor(new double[] { 1, 1 }));

          Assert.NotEqual(
              ((Dense)a.Layers.Layers[0]).Neurons[0].Weights,
              ((Dense)b.Layers.Layers[0]).Neurons[0].Weights);
      }

      [Fact]
      public void Seeded_conv_kernels_are_reproducible()
      {
          My_DNN.MDNN WithConv(int seed)
          {
              var m = new My_DNN.MDNN(new Dense(2, new Linear()), new SGD(0.01), new MSE(), seed);
              m.Layers.Add(new Conv(2, 3, new ReLu(), "valid"));
              m.Layers.SetInputSizeForFirstLayer(new int[] { 6, 6, 1 });
              return m;
          }

          var a = (Conv)WithConv(99).Layers.Layers[0];
          var b = (Conv)WithConv(99).Layers.Layers[0];

          Assert.Equal(a.Kernel[0][0][0], b.Kernel[0][0][0]);
          Assert.Equal(a.Biases, b.Biases);
      }
  }

  public class SeedErgonomicsTests : GradientCheckTestBase
  {
      [Fact]
      public void Seed_can_be_read_back()
      {
          // AutoML si u kandidáta potřebuje poznamenat, čím se běh dá zopakovat
          var model = new My_DNN.MDNN(new Dense(2, new Linear()), new SGD(0.01), new MSE(), seed: 1234);
          Assert.Equal(1234, model.Context.Seed);
      }

      [Fact]
      public void Seed_is_null_when_not_given()
      {
          var model = new My_DNN.MDNN(new Dense(2, new Linear()), new SGD(0.01), new MSE());
          Assert.Null(model.Context.Seed);
      }

      [Fact]
      public void Seed_can_be_set_via_context_before_layers_are_built()
      {
          My_DNN.MDNN Build()
          {
              var m = new My_DNN.MDNN(new Dense(2, new Linear()), new SGD(0.01), new MSE());
              m.Context.Seed = 42;                       // ještě se nic nelosovalo → OK
              m.Layers.Add(new Dense(3, new ReLu()));
              m.GetResults(new Tensor(new double[] { 1, 1 }));
              return m;
          }

          Assert.Equal(
              ((Dense)Build().Layers.Layers[0]).Neurons[0].Weights,
              ((Dense)Build().Layers.Layers[0]).Neurons[0].Weights);
      }

      [Fact]
      public void Seed_can_still_be_set_after_adding_layers()
      {
          // Od sjednocení materializace do jednoho průchodu Add() NELOSUJE — jen zapojí tvary.
          // Okno pro nastavení seedu je proto mnohem širší než dřív: stačí to stihnout před
          // prvním forwardem. (Dřív Add() vylosoval váhy výstupní vrstvy a seed nastavený
          // potom už na ni nedosáhl → model vyšel jen ČÁSTEČNĚ reprodukovatelný.)
          My_DNN.MDNN Build()
          {
              var m = new My_DNN.MDNN(new Dense(2, new Linear()), new SGD(0.01), new MSE());
              m.Layers.Add(new Dense(3, new ReLu()));
              m.Context.Seed = 42;                        // po Add(), ale před forwardem → OK
              m.GetResults(new Tensor(new double[] { 1, 1 }));
              return m;
          }

          Assert.Equal(
              ((Dense)Build().Layers.Layers[0]).Neurons[0].Weights,
              ((Dense)Build().Layers.Layers[0]).Neurons[0].Weights);
      }

      [Fact]
      public void Setting_seed_after_weights_were_drawn_throws()
      {
          // Guard pořád platí — jen se posunul tam, kde se opravdu losuje: za první forward.
          var model = new My_DNN.MDNN(new Dense(2, new Linear()), new SGD(0.01), new MSE());
          model.Layers.Add(new Dense(3, new ReLu()));
          model.GetResults(new Tensor(new double[] { 1, 1 }));   // tady se losuje

          var ex = Assert.Throws<InvalidOperationException>(() => model.Context.Seed = 42);

          Assert.Contains("konstruktoru", ex.Message);   // hláška má říct, kudy z toho ven
      }

      [Fact]
      public void Setting_random_after_use_throws_too()
      {
          var model = new My_DNN.MDNN(new Dense(2, new Linear()), new SGD(0.01), new MSE());
          model.Layers.Add(new Dense(3, new ReLu()));
          model.GetResults(new Tensor(new double[] { 1, 1 }));

          Assert.Throws<InvalidOperationException>(() => model.Context.Random = new Random(42));
      }

      [Fact]
      public void LoadModel_accepts_seed_for_further_training()
      {
          // váhy se berou ze souboru, ale míchání dat a výběr vzorků při dotrénování
          // musí jít zafixovat taky
          string path = System.IO.Path.Combine(System.IO.Path.GetTempPath(), "mdnn_seed_" + Guid.NewGuid().ToString("N"));
          try
          {
              var src = new My_DNN.MDNN(new Dense(1, new Linear()), new SGD(0.01), new MSE(), seed: 1);
              src.GetResults(new Tensor(new double[] { 1, 1 }));
              src.SaveAsJson(path);

              var loaded = My_DNN.MDNN.LoadModel(path, seed: 99);

              Assert.Equal(99, loaded.Context.Seed);
          }
          finally { try { System.IO.File.Delete(path + ".json"); } catch { } }
      }

      // ------------------------------------------------------------------------------
      // Dřív tu bylo ZNÁMÉ OMEZENÍ s Assert.NotEqual: seed platil jen pro stejně poskládaný
      // model, protože LayerAdjustment materializovalo parametry už při skládání a pořadí
      // losování bylo funkcí historie volání (Add/Insert), ne výsledné architektury.
      //
      // Po sjednocení materializace do jediného průchodu (seed krok 3) to platit přestalo,
      // takže test je OTOČENÝ na Assert.Equal — přesně jak si ten komentář žádal.
      // ------------------------------------------------------------------------------
      [Fact]
      public void Seed_survives_different_assembly_order()
      {
          My_DNN.MDNN ViaAddAdd()
          {
              var m = new My_DNN.MDNN(new Dense(2, new Linear()), new SGD(0.01), new MSE(), seed: 42);
              m.Layers.Add(new Dense(3, new ReLu()));
              m.Layers.Add(new Dense(4, new ReLu()));
              m.GetResults(new Tensor(new double[] { 1, 1 }));
              return m;
          }

          My_DNN.MDNN ViaAddInsert()
          {
              var m = new My_DNN.MDNN(new Dense(2, new Linear()), new SGD(0.01), new MSE(), seed: 42);
              m.Layers.Add(new Dense(4, new ReLu()));
              m.Layers.Insert(0, new Dense(3, new ReLu()));
              m.GetResults(new Tensor(new double[] { 1, 1 }));
              return m;
          }

          var a = ViaAddAdd();
          var b = ViaAddInsert();

          // stejná výsledná topologie…
          Assert.Equal(a.Layers.Layers.Count, b.Layers.Layers.Count);
          for (int i = 0; i < a.Layers.Layers.Count; i++)
              Assert.Equal(
                  ((Dense)a.Layers.Layers[i]).Neurons.Count,
                  ((Dense)b.Layers.Layers[i]).Neurons.Count);

          // …a nově i stejné váhy, ve VŠECH vrstvách
          for (int i = 0; i < a.Layers.Layers.Count; i++)
              Assert.Equal(
                  ((Dense)a.Layers.Layers[i]).Neurons[0].Weights,
                  ((Dense)b.Layers.Layers[i]).Neurons[0].Weights);
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
              Assert.Contains("Aktualizuj knihovnu", ex.Message);
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
