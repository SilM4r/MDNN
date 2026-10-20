using System;
  using Xunit;
  using MDNN;
  using MDNN.Layers;
  using MDNN.Activations;
  using MDNN.Optimizers;
  using MDNN.Losses;
  using System.Linq;

  // static state in the library → tests must not run in parallel
  [assembly: CollectionBehavior(DisableTestParallelization = true)]

  namespace MDNN.Tests;

  // ---------- infrastructure ----------

  public static class TestGlobals
  {
      public static void Reset()
      {
          GeneralNeuralNetworkSettings.Random = new Random(42);   // determinism (weight init)
          GeneralNeuralNetworkSettings.DefaultOptimizer = new SGD(0.0001);
          GeneralNeuralNetworkSettings.DefaultHiddenActivation = new ReLU();
      }
  }

  public static class Numeric
  {
      // derivative of a function with respect to its argument (for activations/loss)
      public static double CentralDiff(Func<double, double> f, double x, double eps = 1e-5)
          => (f(x + eps) - f(x - eps)) / (2 * eps);

      // numerical gradient of a parameter: perturb the slot, run the scalar loss, restore it
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
      protected GradientCheckTestBase() => TestGlobals.Reset();   // xUnit: runs before every test
  }
