namespace MDNN.Activations
{
    /// <summary>
    /// Base class for activations that act on the whole layer output at once (e.g. softmax).
    /// </summary>
    public abstract class LayerActivationFunction : ActivationFunction
    {
        /// <inheritdoc/>
        public override bool AppliesToLayer => true;

        /// <summary>
        /// Applies the activation to the whole vector of raw layer outputs.
        /// </summary>
        public abstract double[] ApplyToLayer(double[] values);

        // Previously DerivativeForLayer() returned only the DIAGONAL of the Jacobian, applied element-wise
        // via Derivative(double). That is wrong for softmax: ∂L/∂z_i = Σ_j (∂L/∂s_j)·J[j,i], a product with
        // the FULL Jacobian, not just J[i,i]. (Measured against numerical gradients: ~24 % error for
        // softmax+MSE. The fused softmax+CE path does not go through here, which is why it went unnoticed.)
        /// <summary>
        /// Backward pass for a layer-wide activation: given the raw layer outputs and the gradient from above
        /// (∂L/∂output), returns ∂L/∂raw_output.
        /// </summary>
        public abstract double[] BackwardForLayer(double[] rawOutput, double[] gradFromAbove);
    }
}
