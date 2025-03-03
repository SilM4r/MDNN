using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace My_DNN.Layers.classes
{
    public abstract class LayerBasedOnNeurons : Layer
    {
        abstract public List<Neuron> Neurons { get; }
    }
}
