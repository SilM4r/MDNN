

namespace mdnn.Activation_functions.classes
{
    public abstract class LayerActivationFunc : Activation_func
    {
        public override bool Apply_to_layer => true;
        public abstract override string Name { get; }
        public abstract override double Apply(double value);
        public abstract override double Derivative(double value);

        public abstract double[] ApplyToLayer(double[] values);
        public abstract double[] DerivativeForLayer(double[] values);
    }
}
