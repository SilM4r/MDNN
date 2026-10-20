using MDNN.Activations;
using MDNN.Optimizers;

namespace MDNN
{
    /// <summary>
    /// Process-wide defaults. A model never reads these when it was given its own value:
    /// the optimizer and the random generator come from <see cref="NetworkContext"/>,
    /// these are only the fallback for layers and neurons used outside a model.
    /// </summary>
    public static class GeneralNeuralNetworkSettings
    {
        /// <summary>Activation used when a layer is constructed without one.</summary>
        public static ActivationFunction DefaultHiddenActivation { get; set; } = new ReLU();

        /// <summary>Optimizer cloned into neurons and layers created outside a model.</summary>
        public static Optimizer DefaultOptimizer { get; set; } = new SGD(0.0001);

        /// <summary>Shared generator for models created without a seed.</summary>
        public static Random Random { get; set; } = new Random();
    }
}
