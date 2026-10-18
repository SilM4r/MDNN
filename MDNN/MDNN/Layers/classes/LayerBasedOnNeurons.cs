namespace My_DNN.Layers.classes
{
    public abstract class LayerBasedOnNeurons : Layer
    {
        abstract public List<Neuron> Neurons { get; }

        // Společné pro Dense i RNN: váhy replika sdílí odkazem, takže se přenáší jen bias.
        internal override void SyncReplica(Layer replica)
        {
            List<Neuron> replicaNeurons = ((LayerBasedOnNeurons)replica).Neurons;
            for (int i = 0; i < Neurons.Count; i++)
            {
                replicaNeurons[i].SyncBiasFrom(Neurons[i]);
            }
        }

        internal override void TakeGradientsFrom(Layer replica)
        {
            List<Neuron> replicaNeurons = ((LayerBasedOnNeurons)replica).Neurons;
            for (int i = 0; i < Neurons.Count; i++)
            {
                Neurons[i].TakeGradientsFrom(replicaNeurons[i]);
            }
        }
    }
}
