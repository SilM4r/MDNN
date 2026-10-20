using MDNN.Activations;

namespace MDNN.Activations
{
    /// <summary>
    /// Base class for activation functions.
    /// </summary>
    public abstract class ActivationFunction
    {
        /// <summary>
        /// Name used in the model file; built-in activations are reconstructed from it on load.
        /// </summary>
        public abstract string Name { get; }
        /// <summary>
        /// Applies the activation to a single value.
        /// </summary>
        public abstract double Apply(double value);
        /// <summary>
        /// Derivative of the activation at <paramref name="value"/> (the pre-activation input).
        /// </summary>
        public abstract double Derivative(double value);
        /// <summary>
        /// True if the activation acts on the whole layer at once (e.g. softmax) rather than element-wise.
        /// </summary>
        public abstract bool AppliesToLayer { get; }

        // Activation by the name stored in the model file.
        //
        // Previously anything unknown silently fell back to `default: return new Linear()`. A custom
        // activation thus became linear after loading, and so did old files with a Conv layer, which
        // saved the name via ToString(), i.e. the full type name "MDNN.Activations.ReLU". The model
        // loaded without error and predicted with a different function than it was trained with.
        //
        // Now the full type name is shortened to the class name (so old Conv files load CORRECTLY,
        // not as linear) and an unknown name is an error, not Linear.
        internal static ActivationFunction FromName(string? savedName)
        {
            if (string.IsNullOrEmpty(savedName))
            {
                throw new ArgumentException("The model file is missing the name of a layer's activation function.");
            }

            string name = savedName;
            int lastDot = name.LastIndexOf('.');
            if (lastDot >= 0)
            {
                name = name.Substring(lastDot + 1);
            }

            switch (name.ToLowerInvariant())
            {
                case "linear":
                    return new Linear();
                case "sigmoid":
                    return new Sigmoid();
                case "tanh":
                    return new Tanh();
                case "relu":
                    return new ReLU();
                case "leakyrelu":
                case "leak_relu":   // name used by files saved before the rename to LeakyReLU
                    return new LeakyReLU();
                case "softmax":
                    return new Softmax();
                default:
                    throw new ArgumentException(
                        $"Activation function '{savedName}' is not built in. A model " +
                        "with a custom activation cannot be loaded from a file yet — loading looks it up " +
                        "by name and only knows Linear, Sigmoid, Tanh, ReLU, LeakyReLU and Softmax.");
            }
        }

    }
}
