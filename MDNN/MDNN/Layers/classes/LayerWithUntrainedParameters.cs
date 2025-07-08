using My_DNN.Activation_functions;

namespace My_DNN.Layers.classes
{
    public abstract class LayerWithUntrainedParameters: Layer
    {
        public override void BackPropagation(Tensor TensorE){ return; }
        public override void UpdateParams() { return; }
        public override Activation_func Activation_Func 
        { 
            get => new Linear(); 
            set { } 
        }
    }
}
