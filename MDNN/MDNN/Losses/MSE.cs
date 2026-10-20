using MDNN.Losses;

namespace MDNN.Losses
{
    /// <summary>
    /// Squared-error loss. Despite the name it is the sum of squared errors Σ(v−t)², not a mean,
    /// matching PyTorch <c>reduction='sum'</c>.
    /// </summary>
    public class MSE: Loss
    {
        /// <inheritdoc/>
        public override string Name => "MSE";
        /// <inheritdoc/>
        public override double LossFunction(double output, double target)
        {
            // multiplication instead of Math.Pow — see Tanh.Derivative; called for every output
            double diff = output - target;
            return diff * diff;
        }

        /// <inheritdoc/>
        public override double DerivativeOfLossFunction(double output, double target)
        {
            return 2*(output - target);
        }

    }
}
