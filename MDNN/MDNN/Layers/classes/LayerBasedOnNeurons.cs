namespace My_DNN.Layers.classes
{
    public abstract class LayerBasedOnNeurons : Layer
    {
        abstract public List<Neuron> Neurons { get; }
    }
}
