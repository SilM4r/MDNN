namespace MDNN.Layers
{
    /// <summary>
    /// Base class of layers built from <see cref="Neuron"/> objects (Dense, RNN).
    /// </summary>
    public abstract class NeuronLayer : Layer
    {
        /// <summary>Neurons of the layer, each holding its own weights, bias and optimizer state.</summary>
        abstract public List<Neuron> Neurons { get; }

        // Shared by Dense and RNN: the replica shares weights by reference, so only the bias is copied.
        internal override void SyncReplica(Layer replica)
        {
            List<Neuron> replicaNeurons = ((NeuronLayer)replica).Neurons;
            for (int i = 0; i < Neurons.Count; i++)
            {
                replicaNeurons[i].SyncBiasFrom(Neurons[i]);
            }
        }

        internal override void ReplaceOptimizer(MDNN.Optimizers.Optimizer prototype)
        {
            foreach (Neuron neuron in Neurons)
            {
                neuron.Optimizer = prototype.CreateFresh();
            }
        }

        internal override void TakeGradientsFrom(Layer replica)
        {
            List<Neuron> replicaNeurons = ((NeuronLayer)replica).Neurons;
            for (int i = 0; i < Neurons.Count; i++)
            {
                Neurons[i].TakeGradientsFrom(replicaNeurons[i]);
            }
        }
    }
}
