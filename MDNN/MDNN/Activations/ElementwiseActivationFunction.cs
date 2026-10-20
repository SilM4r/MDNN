namespace MDNN.Activations
{
    /// <summary>
    /// Base class for activations applied to each value independently.
    /// </summary>
    public abstract class ElementwiseActivationFunction: ActivationFunction
    {
        /// <inheritdoc/>
        public override bool AppliesToLayer => false;

    }
}
