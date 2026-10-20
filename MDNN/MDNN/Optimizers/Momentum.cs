namespace MDNN.Optimizers
{
    /// <summary>
    /// SGD with momentum: <c>v ← momentum · v + learningRate · gradient</c>, <c>w ← w − v</c>.
    /// </summary>
    public class Momentum : Optimizer
    {
        private readonly double learningRate;
        private readonly double momentum;

        private double[] velocity = new double[0];

        /// <inheritdoc/>
        public override double[] Hyperparameters => new[] { learningRate, momentum };
        /// <inheritdoc/>
        public override string Name => "Momentum";

        /// <summary>
        /// Creates a momentum optimizer.
        /// </summary>
        /// <param name="learningRate">Step size.</param>
        /// <param name="momentum">Decay factor of the velocity (typically 0.9).</param>
        public Momentum(double learningRate, double momentum)
        {
            this.learningRate = learningRate;
            this.momentum = momentum;
        }

        /// <inheritdoc/>
        public override Optimizer CreateFresh() => new Momentum(learningRate, momentum);

        /// <inheritdoc/>
        public override Dictionary<string, double[]> ExportState() => new()
        {
            ["velocity"] = (double[])velocity.Clone(),
        };

        /// <inheritdoc/>
        public override void ImportState(Dictionary<string, double[]> state)
        {
            velocity = RequireState(state, "velocity", null);
        }

        // Exact size — see Adam.EnsureCapacity.
        /// <inheritdoc/>
        public override void EnsureCapacity(int parameterCount)
        {
            if (velocity.Length < parameterCount) Array.Resize(ref velocity, parameterCount);
        }

        /// <inheritdoc/>
        public override double Update(double w, double gradient, int paramIndex)
        {
            // Safeguard for calls without EnsureCapacity — see Adam.Update.
            if (velocity.Length <= paramIndex) EnsureCapacity(paramIndex + 1);
            velocity[paramIndex] = momentum * velocity[paramIndex] + learningRate * gradient;
            return w - velocity[paramIndex];
        }
    }
}
