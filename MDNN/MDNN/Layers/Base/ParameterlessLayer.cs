using MDNN.Activations;

namespace MDNN.Layers
{
    /// <summary>
    /// Base class of layers without trainable parameters (e.g. pooling); no-op update and a fixed linear activation.
    /// </summary>
    public abstract class ParameterlessLayer: Layer
    {
        /// <summary>Does nothing — the layer has no parameters to update.</summary>
        public override void UpdateParams() { return; }
        /// <summary>Always <see cref="Linear"/>; setting it has no effect.</summary>
        public override ActivationFunction Activation
        { 
            get => new Linear(); 
            set { } 
        }
    }
}
