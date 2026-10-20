namespace MDNN.Losses
{
    /// <summary>
    /// Base class for loss functions. Also accumulates the loss over iterations for reporting.
    /// </summary>
    public abstract class Loss
    {
        /// <summary>
        /// Per-output loss values from the last <see cref="CalculateLoss"/> call, or <c>null</c> if none yet.
        /// </summary>
        public double[]? Losses
        {
            get { return losses; }
        }
        private int iteration = 0;
        private double totalLossSum = 0;
        private double[]? losses;
        /// <summary>
        /// Name used in the model file; built-in losses are reconstructed from it on load.
        /// </summary>
        public abstract string Name { get; }

        /// <summary>
        /// Loss of a single output value against its target.
        /// </summary>
        public abstract double LossFunction(double output, double target);

        /// <summary>
        /// Derivative of the loss with respect to the output value.
        /// </summary>
        public abstract double DerivativeOfLossFunction(double output, double target);

        /// <summary>
        /// True if <see cref="DerivativeOfLossFunction"/> already returns the fused gradient dL/dz for a
        /// softmax output (i.e. s − t); backprop then skips the multiplication by the softmax derivative.
        /// </summary>
        public virtual bool RequiresSoftmax => false;

        /// <summary>
        /// Computes the loss for one sample, stores it in <see cref="Losses"/> and adds it to the running total.
        /// </summary>
        public void CalculateLoss(double[] outputs, double[] targets)
        {
            int n = outputs.Length;

            losses = new double[n];

            for (int i = 0; i < n; i++)
            {
                double lossValue = LossFunction(outputs[i], targets[i]);
                losses[i] = lossValue;
                totalLossSum += lossValue;
            }

            iteration++;

        }

        /// <summary>
        /// Returns the summed loss for one sample without affecting the running total.
        /// </summary>
        public double CalculateAndGetLoss(double[] outputs, double[] targets)
        {
            int n = outputs.Length;

            double lossSum = 0;

            for (int i = 0; i < n; i++)
            {
                lossSum += LossFunction(outputs[i], targets[i]);
            }

            return lossSum;
        }

        /// <summary>
        /// Returns the average loss per iteration since the last reset, then resets the running total.
        /// </summary>
        public double GetResetAverageLossPerIteration()
        {
            if (iteration == 0)
            {
                return 0;
            }

            double output = totalLossSum / iteration;
            iteration = 0;
            totalLossSum = 0;

            return output;
        }

        /// <summary>
        /// Returns the average loss per iteration since the last reset (0 if there were none).
        /// </summary>
        public double GetAverageLossPerIteration()
        {  
            if (iteration == 0)
            {
                return 0;
            }

            return totalLossSum / iteration;
        }

        /// <summary>
        /// Resets the running loss total.
        /// </summary>
        public void ResetAverageLossPerIteration()
        {
            iteration = 0;
            totalLossSum = 0;
        }

        // Replica for parallel training: the same function, its own (empty) accumulators.
        // Apart from the accumulators a loss is stateless, so a shallow copy suffices and works
        // for custom loss functions too — unlike constructing by name.
        internal Loss CreateReplica()
        {
            Loss replica = (Loss)MemberwiseClone();
            replica.iteration = 0;
            replica.totalLossSum = 0;
            replica.losses = null;
            return replica;
        }

        // Adds the replica's accumulated loss to this one and resets the replica. `Losses` (per element,
        // from the last sample) is taken over only if the replica computed any sample.
        internal void TakeAccumulatedLossFrom(Loss replica)
        {
            totalLossSum += replica.totalLossSum;
            iteration += replica.iteration;

            if (replica.losses != null)
            {
                losses = replica.losses;
            }

            replica.totalLossSum = 0;
            replica.iteration = 0;
            replica.losses = null;
        }

        internal static bool IsBuiltIn(string? name) => name is "Cross Entropy" or "MSE";

        // Loss for a model being loaded — same rules as Optimizer.ResolveForLoad.
        internal static Loss ResolveForLoad(string? savedName, Loss? own, bool ownWins)
        {
            if (own != null && own.Name == savedName && (ownWins || !IsBuiltIn(savedName)))
            {
                return own;
            }

            if (own != null && ownWins)
            {
                throw new ArgumentException(
                    $"The model was saved with loss function '{savedName}', but '{own.Name}' was passed in.");
            }

            if (!IsBuiltIn(savedName) && !string.IsNullOrEmpty(savedName))
            {
                throw new ArgumentException(
                    $"Loss function '{savedName}' is not built in, so loading cannot create it by name. " +
                    "Pass an instance: NeuralNetwork.LoadModel(path, loss: new …).");
            }

            return FromName(savedName);
        }

        // Loss by the name stored in the model file. Previously an unknown name silently fell back to
        // MSE: a model with a custom loss loaded without error and continued training with a completely
        // different objective. Same bug as with the optimizer, just without an exception. Now it throws.
        internal static Loss FromName(string? name)
        {
            switch (name)
            {
                case "Cross Entropy":
                    return new CrossEntropy();
                case "MSE":
                    return new MSE();
                case null or "":
                    throw new ArgumentException("The model file is missing the name of the loss function.");
                default:
                    throw new ArgumentException(
                        $"Loss function '{name}' is not built in. A model with a custom " +
                        "loss cannot be loaded from a file this way — loading looks it up by name " +
                        "and only knows MSE and Cross Entropy.");
            }
        }

    }
}
