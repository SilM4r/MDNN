namespace MDNN.Optimizers
{
    /// <summary>
    /// Plain stochastic gradient descent: <c>w ← w − learningRate · gradient</c>.
    /// </summary>
    public class SGD : Optimizer
    {
        private readonly double learningRate;

        /// <inheritdoc/>
        public override string Name => "SGD";
        /// <inheritdoc/>
        public override double[] Hyperparameters => new[] { learningRate };

        /// <summary>
        /// Creates an SGD optimizer with the given learning rate.
        /// </summary>
        public SGD(double learningRate)
        {
            this.learningRate = learningRate;
        }

        /// <inheritdoc/>
        public override Optimizer CreateFresh() => new SGD(learningRate);

        /// <inheritdoc/>
        public override double Update(double w, double gradient, int paramIndex)
        {
            return w - learningRate * gradient;
        }
    }
}
