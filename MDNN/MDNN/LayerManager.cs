using MDNN.Layers;
using MDNN.Serialization;
using MDNN.Activations;


namespace MDNN
{
    /// <summary>
    /// The ordered list of a model's layers; the last layer is the output layer.
    /// </summary>
    public class LayerManager
    {
        private List<Layer> layersList = new List<Layer>();

        // Read-only view: layers are added and removed only through Add/Insert/RemoveAt,
        // which also rewire the shapes of the chain. A layer added straight into the list
        // would stay unwired.
        /// <summary>
        /// The layers in order, from input to output (read-only; use Add/Insert/RemoveAt to change).
        /// </summary>
        public IReadOnlyList<Layer> Layers => layersList;

        // Does any layer have its shape wired but no parameters yet? Signals the model
        // that materialization must finish before the forward pass.
        internal bool HasUnmaterializedLayers => layersList.Any(layer => !layer.IsMaterialized);
        private NetworkContext context;

        internal LayerManager(Layer outputLayer, NetworkContext context)
        {
            this.context = context;
            context.OptimizerChanged = ReplaceOptimizer;
            outputLayer.Context = context;
            layersList = new List<Layer> { outputLayer };
            MarkLayerRoles();
        }
        internal LayerManager(List<BaseExportLayer> exportLayers, NetworkContext context)
        {
            this.context = context;
            context.OptimizerChanged = ReplaceOptimizer;
            layersList = new List<Layer>();

            // The model's optimizer is passed to the layers directly. Previously the global
            // GeneralNeuralNetworkSettings.DefaultOptimizer was overwritten for this and the constructors
            // cloned from it — two loads on two threads could swap each other's optimizer.
            MDNN.Optimizers.Optimizer optimizer = context.Optimizer;

            foreach (BaseExportLayer layer in exportLayers)
            {

                if (layer.LayerType == "Dense")
                {
                    layersList.Add(new Dense((ExportDenseLayer)layer, optimizer));
                }
                else if (layer.LayerType == "RNN")
                {
                    layersList.Add(new RNN((ExportRnnLayer)layer, optimizer));
                }
                else if (layer.LayerType == "Conv")
                {
                    layersList.Add(new Conv((ExportCNNLayer)layer, optimizer));
                }
                else if (layer.LayerType == "MaxPool")
                {
                    layersList.Add(new MaxPool((ExportMaxPoolLayer)layer));
                }
                else
                {
                    throw new NotSupportedException($"Loading of layer type '{layer.LayerType}' is not implemented yet.");
                }
            }

            foreach (Layer l in layersList) l.Context = context;
            MarkLayerRoles();
        }
        // Manager of ready-made layers (replicas) — no wiring or materialization, the layers
        // inherit those from the original. Only sets their context and roles.
        internal static LayerManager ForReplica(List<Layer> layers, NetworkContext context)
        {
            return new LayerManager(context, layers);
        }

        private LayerManager(NetworkContext context, List<Layer> layers)
        {
            this.context = context;
            layersList = layers;
            foreach (Layer layer in layers) layer.Context = context;
            MarkLayerRoles();
        }

        // The model got a different optimizer — each layer takes a fresh clone of it.
        private void ReplaceOptimizer(MDNN.Optimizers.Optimizer prototype)
        {
            foreach (Layer layer in layersList)
            {
                layer.ReplaceOptimizer(prototype);
            }
        }

        /// <summary>
        /// Sets the input shape (or uses the one already in the context), wires all layer shapes
        /// and initializes their parameters.
        /// </summary>
        /// <param name="inputShape">Shape of one input sample; <c>null</c> keeps the context's current shape.</param>
        public void SetInputShape(int[]? inputShape = null)
        {
            if (inputShape != null)
            {
                context.InputShape = inputShape;
            }

            foreach (int size in context.InputShape)
            {
                if (size <= 0)
                {
                    throw new ArgumentException("The number of inputs to the first layer must be greater than zero.");
                }
            }

            RewireChain();
            MaterializeAll();
        }

        // Which layer is the output layer. Must be recomputed on EVERY chain change — after Insert
        // or RemoveAt a different layer may become last, and a layer keeping a stale flag would get
        // the global dropout (see Layer.IsOutputLayer).
        private void MarkLayerRoles()
        {
            for (int i = 0; i < layersList.Count; i++)
            {
                layersList[i].IsOutputLayer = (i == layersList.Count - 1);
            }
        }

        // Wires shapes left to right. Draws nothing random, so it may be called any number of
        // times — after every structural change. If the model input is not known yet it does
        // nothing and shapes are computed on the first forward pass.
        internal void RewireChain()
        {
            // DELIBERATELY before the early return below: layer roles can be determined even
            // when the input shape is not known yet and wiring is skipped.
            MarkLayerRoles();

            int[] inputShape = EffectiveInputShape();

            if (inputShape.Length == 0 || inputShape[0] <= 0)
            {
                return;
            }

            layersList[0].WireShapes(null, inputShape);

            for (int i = 1; i < layersList.Count; i++)
            {
                layersList[i].WireShapes(null, layersList[i - 1].OutputShape);
            }
        }

        // Materialization in ONE left-to-right pass. The draw order is thus a function of the
        // architecture, not of the build history — two identical networks with the same seed come
        // out the same regardless of whether they were built via Add() or Insert().
        internal void MaterializeAll()
        {
            foreach (Layer layer in layersList)
            {
                layer.MaterializeParameters();
            }
        }

        // The model's input shape. Context takes precedence; for a LOADED model, however, Context
        // does not know it while the first layer does (from the file) — without this branch a layer
        // added via Add() to a loaded model would stay unwired and the forward pass would throw IndexOutOfRange.
        private int[] EffectiveInputShape()
        {
            if (context.InputShape.Length > 0 && context.InputShape[0] > 0)
            {
                return context.InputShape;
            }

            if (layersList.Count > 0)
            {
                int[] firstLayerInput = layersList[0].InputShape;
                if (firstLayerInput.Length > 0 && firstLayerInput[0] > 0)
                {
                    return firstLayerInput;
                }
            }

            return new int[] { 0 };
        }

        /// <summary>
        /// Adds a hidden layer just before the output layer.
        /// </summary>
        public void Add(Layer layer)
        {
            layer.Context = context;
            layersList.Insert(layersList.Count() - 1, layer);
            RewireChain();
        }
        /// <summary>
        /// Inserts a layer at <paramref name="position"/> (0 = first layer).
        /// </summary>
        public void Insert(int position, Layer layer)
        {
            if (position < 0 || position > layersList.Count())
            {
                throw new ArgumentOutOfRangeException(nameof(position), position, "Position must be in the range 0 ≤ position ≤ Layers.Count.");
            }

            layer.Context = context;
            layersList.Insert(position, layer);
            RewireChain();
        }
        /// <summary>
        /// Removes the layer at <paramref name="position"/>. The only remaining layer cannot be removed.
        /// </summary>
        public void RemoveAt(int position)
        {
            if (position < 0 || position >= layersList.Count())
            {
                throw new ArgumentOutOfRangeException(nameof(position), position, "Position must be in the range 0 ≤ position < Layers.Count.");
            }

            // A model always has an output layer; to replace it use Clear(outputLayer).
            if (layersList.Count == 1)
            {
                throw new InvalidOperationException("The only layer of the model cannot be removed. Use Clear(outputLayer) to replace it.");
            }

            layersList.RemoveAt(position);
            RewireChain();
        }
        /// <summary>
        /// Sets the activation function of the output layer.
        /// </summary>
        public void SetOutputActivation(ActivationFunction activation)
        {
            layersList[layersList.Count() - 1].Activation = activation;
        }
        /// <summary>
        /// Removes all layers and starts over with <paramref name="outputLayer"/> as the only layer.
        /// </summary>
        public void Clear(Layer outputLayer)
        {
            outputLayer.Context = context;
            layersList = new List<Layer> { outputLayer };
            RewireChain();
        }
    }
}
