
namespace MDNN.Activations
{
    /// <summary>
    /// Hyperbolic tangent activation.
    /// </summary>
    public class Tanh : ElementwiseActivationFunction
    {
        /// <inheritdoc/>
        public override string Name
        {
            get { return "Tanh"; }
        }

        /// <inheritdoc/>
        public override double Apply(double value)
        {
            return Math.Tanh(value);
        }

        /// <inheritdoc/>
        public override double Derivative(double value)
        {
            // Math.Pow is a general power function (branching + internal log/exp); squaring needs
            // just one multiplication. This runs for every neuron in every backward pass.
            double s = Apply(value);
            return 1 - s * s;
        }
    }
}
