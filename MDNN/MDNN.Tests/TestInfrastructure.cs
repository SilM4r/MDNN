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
