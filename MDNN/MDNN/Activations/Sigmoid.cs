
namespace MDNN.Activations
{
    /// <summary>
    /// Logistic sigmoid: <c>1 / (1 + e^−x)</c>.
    /// </summary>
    public class Sigmoid : ElementwiseActivationFunction
    {

        /// <inheritdoc/>
        public override string Name
        {
            get { return "Sigmoid"; }
        }
        /// <inheritdoc/>
        public override double Apply(double value)
        {
            return 1.0 / (1.0 + Math.Exp(-value));
        }

        /// <inheritdoc/>
        public override double Derivative(double value)
        {
            // s(1-s) needs the sigmoid once, not twice — previously Math.Exp was computed 2×.
            // The result is bit-identical, just half the cost.
            double s = Apply(value);
            return s * (1 - s);
        }
    }
}
