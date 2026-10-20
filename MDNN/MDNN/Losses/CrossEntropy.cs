namespace MDNN.Losses
{
    /// <summary>
    /// Categorical cross-entropy <c>L = −Σ t_i · ln(s_i)</c>, used together with a softmax output.
    /// </summary>
    public class CrossEntropy: Loss
    {
        /// <inheritdoc/>
        public override string Name => "Cross Entropy";
        // The guard against ln(0) is a CLAMP, not an added epsilon. Previously it was `value + 1e-15`,
        // which silently CAPPED the loss at −ln(1e-15) ≈ 34.5: once softmax output a probability below
        // epsilon, the epsilon dominated and the loss stopped growing. For a saturated softmax
        // (s ≈ 1e-181, "confident and wrong") it gave 34.5 instead of 414.7 — and since the best model
        // and early stopping are chosen by validation loss, the bug was invisible. The gradient takes a
        // different path (fused s − t) and was unaffected, so the model learned correctly and only
        // REPORTED wrongly.
        //
        // Clamping to 1e-300 keeps the true value up to where double underflows to zero while still
        // preventing −Infinity. Found by a parity test against PyTorch (fixture dense_softmax_ce_saturated).
        /// <inheritdoc/>
        public override double LossFunction(double output, double target)
        {
            return -target * Math.Log(Math.Max(output, 1e-300));
        }

        // Valid only for a softmax output; thanks to RequiresSoftmax, backprop takes this value as the
        // finished delta and does NOT multiply it by the softmax derivative.
        /// <summary>
        /// Fused gradient of softmax + categorical cross-entropy with respect to the logits: <c>dL/dz = s − t</c>.
        /// </summary>
        public override double DerivativeOfLossFunction(double output, double target)
        {
            return output - target;
        }

        /// <inheritdoc/>
        public override bool RequiresSoftmax => true;
    }
}
