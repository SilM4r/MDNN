
namespace MDNN.Activations
{
    /// <summary>
    /// Leaky ReLU: <c>x</c> for <c>x ≥ 0</c>, <c>0.01·x</c> otherwise.
    /// </summary>
    public class LeakyReLU : ElementwiseActivationFunction
    {

        /// <inheritdoc/>
        public override string Name
        {
            get { return "LeakyReLU"; }
        }

        /// <inheritdoc/>
        public override double Apply(double value)
        {
            if (value < 0)
                return value * 0.01;
            else
                return value;
        }

        /// <inheritdoc/>
        public override double Derivative(double value)
        {
            if (value < 0)
                return 0.01;
            else return 1;
        }
    }
}
