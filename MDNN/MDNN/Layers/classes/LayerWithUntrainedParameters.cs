
using mdnn.Activation_functions.classes;

namespace mdnn.Layers.classes
{
    public abstract class LayerWithUntrainedParameters: Layer
    {
        public override void BackPropagation(Tensor TensorE){ return; }
        public override void UpdateParams() { return; }
        public override Activation_func Activation_Func 
        { 
            get => Activation_func.Linear(); 
            set { } 
        }
    }
}
