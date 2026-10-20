

namespace MDNN.Activations
{
    /// <summary>
    /// Rectified linear unit: <c>max(0, x)</c>.
    /// </summary>
    public class ReLU : ElementwiseActivationFunction
    {
        /// <inheritdoc/>
        public override string Name
        {
            get { return "ReLU"; }
        }

        /// <inheritdoc/>
        public override double Apply(double value)
        {
            return Math.Max(0, value);
        }

        /// <inheritdoc/>
        public override double Derivative(double value)
        {
            if (value < 0)
                return 0;

            else return 1; 
        }
    }
}
