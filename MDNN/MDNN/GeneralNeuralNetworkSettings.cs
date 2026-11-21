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
        /// <remarks>
        /// All models without a seed draw from it, so it must be thread-safe when such models are used
        /// from several threads (e.g. an AutoML runner training candidates in parallel). The default,
        /// <see cref="System.Random.Shared"/>, is. An instance created with <c>new Random(…)</c> is NOT:
        /// measured with 8 threads, <c>new Random()</c> handed the same numbers to several threads
        /// (27 % of the draws were duplicates) and <c>new Random(42)</c> got stuck returning 0.
        /// Give each model its own seed instead when you need reproducibility.
        /// </remarks>
        public static Random Random { get; set; } = Random.Shared;
    }
}
