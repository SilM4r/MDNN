// NOTE on the namespace: the file lives in Exceptions/, but the namespace is `MDNN` (root), not
// `MDNN.Exceptions`. Elsewhere in the repo folder = namespace; here it differs on purpose:
//   - The folder is for OUR organization (there will be more of them), the namespace is the caller's API.
//   - Whoever catches `TrainingDivergedException` already has `using MDNN;` (otherwise they would
//     not have NeuralNetwork). Forcing a second using is friction with no benefit.
//   - It matches the BCL: `FileNotFoundException` is in `System.IO`, not `System.IO.Exceptions`
//     — .NET puts exceptions where they belong by meaning, not into a namespace of their own.
// Aligning it with the rest of the repo is a one-line change.
namespace MDNN
{
    // Previously `Environment.Exit(0)` was called here: the library killed the host process, with
    // exit code 0 = "success". The caller could not react and a supervising tool (an AutoML runner,
    // a CI script) never learned that anything failed. With an exception, whoever wants to can catch
    // it and move on to the next candidate; everyone else gets a normal stack trace.
    /// <summary>
    /// Thrown when training diverges: the loss is NaN or infinite, so further steps would only
    /// propagate garbage and the weights are unusable.
    /// </summary>
    public class TrainingDivergedException : Exception
    {
        /// <summary>
        /// The epoch in which the divergence was first detected.
        /// </summary>
        public uint Epoch { get; }

        /// <summary>
        /// The loss value that revealed the divergence (NaN or ±∞).
        /// </summary>
        public double Loss { get; }

        /// <summary>
        /// Creates the exception for the given epoch and loss value.
        /// </summary>
        /// <param name="epoch">Epoch in which the divergence was detected.</param>
        /// <param name="loss">The diverged loss value.</param>
        public TrainingDivergedException(uint epoch, double loss)
            : base($"Training diverged in epoch {epoch}: loss = {loss}. " +
                   "The network output is too large or too small — try a lower learning rate, " +
                   "a different weight initialization or normalized inputs.")
        {
            Epoch = epoch;
            Loss = loss;
        }
    }
}
