using My_DNN.Activation_functions;
using ScottPlot;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace My_DNN.Layers.classes
{
    public abstract class Layer
    {
        abstract public string Name { get; }

        abstract public int[] Input_size_and_shape { get; }
        abstract public int[] Output_size_and_shape { get; }
        abstract public Tensor? Layer_output { get; }
        abstract public Tensor? Layer_raw_output { get; }
        abstract public Activation_func Activation_Func { get; set; }
        abstract public Tensor FeedForward(Tensor input_values);
        abstract public Tensor CalculateLayerGradients(Tensor next_layer_e, Layer next_layer);
        abstract public void BackPropagation(Tensor de);
        abstract public void UpdateParams();
        abstract public void LayerAdjustment(int? number_of_elements = null, int[]? number_of_input = null);

        virtual public async Task<Tensor> FeedForwardAsync(Tensor input_values)
        {
            return FeedForward(input_values);
        }
        virtual public async Task<Tensor> CalculateLayerGradientsAsync(Tensor next_layer_e, Layer next_layer)
        {
            return CalculateLayerGradients(next_layer_e,next_layer);
        }
        virtual public async Task BackPropagationAsync(Tensor de)
        {
            BackPropagation(de);
        }
        virtual public async Task UpdateParamsAsync()
        {
            UpdateParams();
        }

        public static Layer Dense(int number_of_neuron, Activation_func? activation_func = null)
        {
            return new Dense(number_of_neuron, activation_func);
        }
    }
}
