using MDNN.Layers;
using MDNN.Serialization;
using MDNN.Activations;


namespace MDNN
{
    /// <summary>
    /// The ordered list of a model's layers; the last layer is the output layer.
    /// </summary>
    /// <remarks>
    /// Layers are added as DESCRIPTIONS and built only once the shape of the input is known — on the
    /// first pass of data, or earlier via <see cref="SetInputShape"/>. Building goes in one pass from
    /// left to right, so the random draws depend only on the architecture and the seed, not on the
    /// order in which the model was assembled.
    /// </remarks>
    public class LayerManager
    {
        private List<Layer> layersList = new List<Layer>();

        // Read-only view: layers are added and removed only through Add/Insert/RemoveAt/Clear,
        // which also decide which built layers no longer fit.
        /// <summary>
        /// The layers in order, from input to output (read-only; use Add/Insert/RemoveAt to change).
        /// </summary>
        public IReadOnlyList<Layer> Layers => layersList;

        private NetworkContext context;

        // RebuildIncompatible() was called: the next sample may have a different shape, and the
        // layers it no longer fits are rebuilt.
        private bool rebuildIncompatibleOnNextPass;

        // Does anything have to be built before the next forward pass? A plain loop rather than
        // LINQ — it is asked on every GetResults and must not allocate.
        internal bool NeedsBuild
        {
            get
            {
                if (rebuildIncompatibleOnNextPass || context.InputShape == null) return true;

                for (int i = 0; i < layersList.Count; i++)
                {
                    if (!layersList[i].IsBuilt) return true;
                }
                return false;
            }
        }

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

            // A loaded model is built — the input it is built for is the one of its first layer.
            // (Previously Context did not know it at all and EffectiveInputShape() had to fall back
            // to the first layer: two sources of truth for one piece of information.)
            //
            // Overwritten even when the context already knows a shape: LoadWeightsInPlace with a file
            // of a DIFFERENT architecture ends up here, and the old shape would then be a lie.
            context.InputShape = layersList.Count > 0 && layersList[0].InputShape != null
                ? (int[])layersList[0].InputShape!.Clone()
                : null;
            rebuildIncompatibleOnNextPass = false;
        }
        // Manager of ready-made layers (replicas) — no building, the layers inherit that from the
        // original. Only sets their context and roles.
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
        internal void ReplaceOptimizer(MDNN.Optimizers.Optimizer prototype)
        {
            foreach (Layer layer in layersList)
            {
                layer.ReplaceOptimizer(prototype);
            }
        }

        // Loads saved parameters INTO the existing layers when the file describes exactly this
        // architecture (layer by layer: type, sizes, activation). All layers are checked and
        // prepared first and only then written, so a mismatch or a damaged file leaves the model
        // untouched. false = the architecture differs and the caller has to build new layers from the file.
        internal bool TryImportParameters(List<BaseExportLayer> saved, MDNN.Optimizers.Optimizer optimizer)
        {
            if (saved.Count != layersList.Count)
            {
                return false;
            }

            for (int i = 0; i < saved.Count; i++)
            {
                if (!layersList[i].CanImport(saved[i]))
                {
                    return false;
                }
            }

            // everything that can fail, for ALL layers, before anything is written — see Layer.PrepareImport
            var commits = new List<Action>(saved.Count);
            for (int i = 0; i < saved.Count; i++)
            {
                commits.Add(layersList[i].PrepareImport(saved[i], optimizer));
            }

            foreach (Action commit in commits)
            {
                commit();
            }

            // The loaded weights are the model's state now. A pending RebuildIncompatible() would
            // otherwise throw them away on the next differently shaped sample — loading through new
            // layers (the other branch in ApplyLoaded) clears it too.
            rebuildIncompatibleOnNextPass = false;
            return true;
        }

        // Re-attaches this manager to the context's optimizer change notification (after it was
        // temporarily detached while a snapshot was being loaded into the layers).
        internal void ReconnectOptimizerChanges()
        {
            context.OptimizerChanged = ReplaceOptimizer;
        }

        // ---------------------------------------------------------------- building

        /// <summary>
        /// Builds the model for the given input shape now, instead of on the first pass of data.
        /// </summary>
        /// <param name="inputShape">Shape of one input sample; <c>null</c> = the shape the model is already built for.</param>
        /// <exception cref="ArgumentException">The model is built for a different input shape — see <see cref="Rebuild"/>.</exception>
        public void SetInputShape(int[]? inputShape = null)
        {
            int[] shape = inputShape ?? context.InputShape ?? throw new InvalidOperationException(
                "The input shape is not known yet. Pass it: model.Layers.SetInputShape(new int[] { … }).");

            EnsureBuilt(shape);
        }

        /// <summary>
        /// Throws away the weights of ALL layers; the model is built again — with new random weights —
        /// for the next data it gets, whatever their shape.
        /// </summary>
        /// <remarks>
        /// For training the same architecture on differently shaped data from scratch. The optimizer
        /// state goes with the weights; the training counters (epochs, optimizer steps) stay.
        /// </remarks>
        /// <param name="inputShape">Build right away for this shape; <c>null</c> = on the first pass of data.</param>
        public void Rebuild(int[]? inputShape = null)
        {
            foreach (Layer layer in layersList)
            {
                layer.Invalidate();
            }

            context.InputShape = null;
            rebuildIncompatibleOnNextPass = false;

            if (inputShape != null)
            {
                EnsureBuilt(inputShape);
            }
        }

        /// <summary>
        /// Lets the next data have a different shape: the layers that no longer fit it get new random
        /// weights, the others KEEP theirs.
        /// </summary>
        /// <remarks>
        /// Typically only the first layer changes (its input got wider or narrower) and the rest of
        /// the trained network stays. Which layers are rebuilt is decided by the shapes alone: a
        /// layer whose input shape stays the same keeps its weights.
        /// </remarks>
        /// <param name="inputShape">Rebuild right away for this shape; <c>null</c> = on the first pass of data.</param>
        public void RebuildIncompatible(int[]? inputShape = null)
        {
            rebuildIncompatibleOnNextPass = true;

            if (inputShape != null)
            {
                EnsureBuilt(inputShape);
            }
        }

        // The ONE place that decides whether the model can take a sample of this shape and builds
        // whatever is missing. Previously five places did their own version of it (GetResults, Fit,
        // Evaluate, TrainLoop, SimpleTrainLoop), each inferring the shape a little differently, and
        // SimpleTrainLoop silently re-randomized the first layer when the data got wider.
        internal void EnsureBuilt(int[] sampleShape)
        {
            RequirePositive(sampleShape);

            if (rebuildIncompatibleOnNextPass)
            {
                BuildChain(sampleShape, invalidateIncompatible: true);
                rebuildIncompatibleOnNextPass = false;
                return;
            }

            CheckInput(sampleShape);

            // Also when every layer is built but the model has never recorded its input — the
            // layers were built on their own (Build) before going into the model. BuildChain then
            // only checks them against the plan and records the shape; previously the model stayed
            // "without input" and training refused to start.
            if (NeedsBuild)
            {
                BuildChain(sampleShape, invalidateIncompatible: false);
            }
        }

        // Training setup (TrainLoop): checks the data against the model, but builds only a model
        // that has never been built. A built model that got new layers (Add/Insert) builds them on
        // its first pass, exactly as before — building them here would move their random draws in
        // front of the epoch shuffle and change the result of a seeded run.
        internal void PrepareFor(int[] sampleShape)
        {
            if (context.InputShape == null || rebuildIncompatibleOnNextPass)
            {
                EnsureBuilt(sampleShape);
            }
            else
            {
                RequirePositive(sampleShape);
                CheckInput(sampleShape);
            }
        }

        // A model that is (or was) built for some input takes only data that fit it.
        private void CheckInput(int[] sampleShape)
        {
            int[]? known = context.InputShape;
            if (known == null)
            {
                return;   // never built — any shape is fine, the model is built for it
            }

            Layer first = layersList[0];

            // The first layer decides: Dense takes [28,28] and [784] alike, Conv compares [h,w,c].
            // When the first layer is new (Insert(0, …) into a built model), it has no shape yet,
            // so at least the size of a sample must stay the same.
            bool fits = first.IsBuilt
                ? first.Accepts(sampleShape)
                : Dense.Flatten(sampleShape) == Dense.Flatten(known);

            if (!fits)
            {
                throw new ArgumentException(
                    $"The model is built for input {SpatialShape.Describe(known)}, but got " +
                    $"{SpatialShape.Describe(sampleShape)}. {Layer.ShapeChangeHint}");
            }
        }

        private static void RequirePositive(int[] shape) => Layer.RequirePositiveShape(shape);

        // Builds the unbuilt layers for `modelInput`, left to right.
        //
        // First the whole chain is PLANNED — every layer's input shape computed without changing
        // anything. A layer that cannot take its input (e.g. a kernel that does not fit) throws
        // there, before anything is drawn, so the model never ends up half built.
        private void BuildChain(int[] modelInput, bool invalidateIncompatible)
        {
            List<int[]> plan = Plan(layersList, modelInput);

            List<int> incompatible = IncompatibleLayers(layersList, plan);

            if (incompatible.Count > 0 && !invalidateIncompatible)
            {
                Layer layer = layersList[incompatible[0]];
                throw new ArgumentException(
                    $"Layer {incompatible[0] + 1} ({layer.Name}) is built for input " +
                    $"{SpatialShape.Describe(layer.InputShape!)}, but in this model it gets " +
                    $"{SpatialShape.Describe(plan[incompatible[0]])}. {Layer.ShapeChangeHint}");
            }

            foreach (int i in incompatible)
            {
                layersList[i].Invalidate();
            }

            // One pass left to right: the draw order is a function of the architecture, not of the
            // build history — two identical networks with the same seed come out the same whether
            // they were assembled with Add() or Insert().
            for (int i = 0; i < layersList.Count; i++)
            {
                if (!layersList[i].IsBuilt)
                {
                    layersList[i].Build(plan[i]);
                }
            }

            context.InputShape = (int[])modelInput.Clone();
        }

        // The input shape of every layer of `layers` when the model gets `modelInput`. Changes nothing.
        private static List<int[]> Plan(IReadOnlyList<Layer> layers, int[] modelInput)
        {
            var inputs = new List<int[]>(layers.Count);
            int[] shape = modelInput;

            foreach (Layer layer in layers)
            {
                int[] normalized = layer.NormalizeForPlan(shape);
                inputs.Add(normalized);
                shape = layer.OutputShapeFor(normalized);
            }

            return inputs;
        }

        // Built layers whose input in the plan differs from the one they are built for.
        private static List<int> IncompatibleLayers(IReadOnlyList<Layer> layers, List<int[]> plan)
        {
            var result = new List<int>();
            for (int i = 0; i < layers.Count; i++)
            {
                if (layers[i].IsBuilt && !SpatialShape.SameAs(layers[i].InputShape!, plan[i]))
                {
                    result.Add(i);
                }
            }
            return result;
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

        // ---------------------------------------------------------------- structure

        // Every change of the structure goes through here. The change is made on a COPY of the list,
        // planned and only then committed — a layer that does not fit throws before the model has
        // changed (previously the layer was already in the list when the error came).
        //
        // If the model is built, the built layers whose input the change shifts (typically the layer
        // right after an inserted or removed one) go back to being descriptions and are built again
        // on the next pass. That is the user's decision — they changed the structure — unlike data of
        // a different shape, which is an error (see EnsureBuilt).
        private void ChangeStructure(Action<List<Layer>> change, Layer? added)
        {
            var candidate = new List<Layer>(layersList);
            change(candidate);

            List<int> stale = context.InputShape != null
                ? IncompatibleLayers(candidate, Plan(candidate, context.InputShape))
                : new List<int>();

            if (added != null)
            {
                added.Context = context;
            }

            layersList = candidate;

            foreach (int i in stale)
            {
                layersList[i].Invalidate();
            }

            // No built layer left (Clear, or a change that shifted every layer's input): there are
            // no weights to protect, so the model forgets the input it was built for and takes any
            // shape again. Otherwise it would reject data of a new size, claiming to be "built" for
            // the old one.
            if (!layersList.Any(layer => layer.IsBuilt))
            {
                context.InputShape = null;
                rebuildIncompatibleOnNextPass = false;
            }

            MarkLayerRoles();
        }

        /// <summary>
        /// Adds a hidden layer just before the output layer.
        /// </summary>
        public void Add(Layer layer)
        {
            ArgumentNullException.ThrowIfNull(layer);
            ChangeStructure(list => list.Insert(list.Count - 1, layer), layer);
        }
        /// <summary>
        /// Inserts a layer at <paramref name="position"/> (0 = first layer).
        /// </summary>
        public void Insert(int position, Layer layer)
        {
            ArgumentNullException.ThrowIfNull(layer);

            if (position < 0 || position > layersList.Count())
            {
                throw new ArgumentOutOfRangeException(nameof(position), position, "Position must be in the range 0 ≤ position ≤ Layers.Count.");
            }

            ChangeStructure(list => list.Insert(position, layer), layer);
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

            ChangeStructure(list => list.RemoveAt(position), null);
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
            ArgumentNullException.ThrowIfNull(outputLayer);
            ChangeStructure(list => { list.Clear(); list.Add(outputLayer); }, outputLayer);
        }
    }
}
