

namespace MDNN.Activations
{
    /// <summary>
    /// Identity activation: <c>f(x) = x</c>.
    /// </summary>
    public class Linear : ElementwiseActivationFunction
    {
        // public override string Name => "Linear";
        /// <inheritdoc/>
        public override string Name
        {
            get { return "Linear"; }
        }

        /// <inheritdoc/>
        public override double Apply(double value)
        {
            return value;
        }

        /// <inheritdoc/>
        public override double Derivative(double value)
        {
            return 1;
        }
    }
}
