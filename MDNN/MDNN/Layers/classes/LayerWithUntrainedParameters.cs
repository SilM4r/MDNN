using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace My_DNN.Layers.classes
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
