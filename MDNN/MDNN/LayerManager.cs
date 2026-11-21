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

        // Read-only view: layers are added and removed only through Hidden, ReplaceOutput and
        // ReplaceAll, which also decide which built layers no longer fit.
        /// <summary>
        /// All layers in order, from input to output (read-only; change them via <see cref="Hidden"/>,
        /// <see cref="ReplaceOutput"/> or <see cref="ReplaceAll"/>).
        /// </summary>
        public IReadOnlyList<Layer> Layers => layersList;

        // The model is split into its hidden layers and the output layer, and each part has its own
        // API — so it is clear from the call alone which of the two it touches. (The input needs no
        // layer of its own: the first layer adapts to the shape of the data.)
        /// <summary>
        /// The hidden layers (all except the output layer), with their own Add/Insert/RemoveAt/Clear/Replace.
        /// </summary>
        public HiddenLayers Hidden => hidden ??= new HiddenLayers(this);
        private HiddenLayers? hidden;

        /// <summary>
        /// The output layer (always the last one). Replace it with <see cref="ReplaceOutput"/>.
        /// </summary>
        public Layer Output => layersList[^1];

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

        // `layers` in order from input to output, the last one is the output layer. The model's
        // constructors use this both for a single output layer and for a whole list.
        internal LayerManager(IEnumerable<Layer> layers, NetworkContext context)
        {
            this.context = context;
            List<Layer> incoming = RequireUsableLayers(layers, nameof(layers));

            RequireSizedByTargetsOnlyAsOutput(incoming);
            foreach (Layer layer in incoming) layer.Context = context;
            context.OptimizerChanged = ReplaceOptimizer;
            layersList = incoming;
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

            // Builds what it can: an output layer whose size comes from the targets waits for them.
            EnsureBuilt(shape, targetShape: null, requireOutput: false);
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

            // an output layer sized by the targets takes its size from the next targets again
            AutomaticOutput?.ForgetTargetSize();

            context.InputShape = null;
            rebuildIncompatibleOnNextPass = false;

            if (inputShape != null)
            {
                EnsureBuilt(inputShape, targetShape: null, requireOutput: false);
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
                EnsureBuilt(inputShape, targetShape: null, requireOutput: false);
            }
        }

        // The ONE place that decides whether the model can take a sample of this shape and builds
        // whatever is missing. Previously five places did their own version of it (GetResults, Fit,
        // Evaluate, TrainLoop, SimpleTrainLoop), each inferring the shape a little differently, and
        // SimpleTrainLoop silently re-randomized the first layer when the data got wider.
        //
        // `targetShape` = shape of one target, when the caller has targets (Fit, TrainLoop, Evaluate);
        // an output layer sized by the targets gets its size from it. `requireOutput` = false builds
        // what can be built without targets (SetInputShape, Rebuild with a shape) and leaves such an
        // output layer for later; true (a forward pass) needs the whole model.
        internal void EnsureBuilt(int[] sampleShape, int[]? targetShape = null, bool requireOutput = true)
        {
            RequirePositive(sampleShape);

            if (targetShape != null)
            {
                PrepareOutputFor(targetShape);
            }

            if (rebuildIncompatibleOnNextPass)
            {
                BuildChain(sampleShape, invalidateIncompatible: true, requireOutput);
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
                BuildChain(sampleShape, invalidateIncompatible: false, requireOutput);
            }
        }

        // Training setup (TrainLoop): checks the data against the model, but builds only a model
        // that has never been built. A built model that got new layers (Add/Insert) builds them on
        // its first pass, exactly as before — building them here would move their random draws in
        // front of the epoch shuffle and change the result of a seeded run.
        internal void PrepareFor(int[] sampleShape, int[] targetShape)
        {
            if (context.InputShape == null || rebuildIncompatibleOnNextPass)
            {
                EnsureBuilt(sampleShape, targetShape);
            }
            else
            {
                RequirePositive(sampleShape);
                PrepareOutputFor(targetShape);
                CheckInput(sampleShape);
            }
        }

        // ---------------------------------------------------------------- output sized by the targets

        // The output layer when its size comes from the targets (Dense.FromTargets — the default), else null.
        private Dense? AutomaticOutput => Output is Dense { SizeFromTargets: true } dense ? dense : null;

        // Is it still waiting for the targets?
        private bool OutputWaitsForTargets => AutomaticOutput is { NeuronCount: 0 };

        // The model got targets of this shape. An output layer sized by the targets gets its size from
        // them; once it has one, targets of another size are an error — the same rule as for the input
        // (a model trained on 10 classes does not silently become one for 12). RebuildIncompatible()
        // allows it: the output layer is rebuilt for the new size, the hidden layers keep their weights.
        // A fixed output layer is not checked here — the backward pass reports a target of the wrong size.
        private void PrepareOutputFor(int[] targetShape)
        {
            Dense? output = AutomaticOutput;
            if (output == null)
            {
                return;
            }

            int count = Dense.Flatten(targetShape);
            if (count == output.NeuronCount)
            {
                return;
            }

            if (output.NeuronCount == 0)
            {
                output.ResolveFromTargets(count, context.Loss);
                return;
            }

            if (!rebuildIncompatibleOnNextPass)
            {
                throw new ArgumentException(
                    $"The output layer is built for {output.NeuronCount} values per sample (the size of the targets " +
                    $"it was trained on), but the targets have {count}. {Layer.ShapeChangeHint}");
            }

            output.Invalidate();
            output.ForgetTargetSize();
            output.ResolveFromTargets(count, context.Loss);
        }

        private const string OutputWaitsForTargetsMessage =
            "The output layer takes its size from the targets, and the model has not seen any yet — so it cannot " +
            "predict before it is trained. Train it first (Fit, TrainLoop), or give it an output layer of a fixed " +
            "size: model.Layers.ReplaceOutput(new Dense(n, …)).";

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
            // When the first layer is new (Hidden.Insert(0, …) into a built model), it has no shape yet,
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
        private void BuildChain(int[] modelInput, bool invalidateIncompatible, bool requireOutput)
        {
            // An output layer that still waits for the targets is left for later — unless the caller
            // needs the whole model (a forward pass), which then cannot run.
            bool outputWaits = OutputWaitsForTargets;
            if (outputWaits && requireOutput)
            {
                throw new InvalidOperationException(OutputWaitsForTargetsMessage);
            }

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
            // they were assembled with Add() or Hidden.Insert().
            int toBuild = outputWaits ? layersList.Count - 1 : layersList.Count;
            for (int i = 0; i < toBuild; i++)
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

            for (int i = 0; i < layers.Count; i++)
            {
                Layer layer = layers[i];
                int[] normalized = layer.NormalizeForPlan(shape);
                inputs.Add(normalized);

                // The output shape of the LAST layer feeds no other layer. It is still computed — that
                // validates the layer before anything is built — except for an output layer whose size
                // is not known until the targets come.
                bool lastWaitingForTargets = i == layers.Count - 1 && layer is Dense { SizeFromTargets: true, NeuronCount: 0 };
                if (!lastWaitingForTargets)
                {
                    shape = layer.OutputShapeFor(normalized);
                }
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

        // Which layer is the output layer. Must be recomputed on EVERY chain change — after
        // ReplaceOutput or ReplaceAll a different layer is last, and a layer keeping a stale flag would get
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
        private void ChangeStructure(Action<List<Layer>> change, IReadOnlyCollection<Layer> added)
        {
            var candidate = new List<Layer>(layersList);
            change(candidate);
            RequireSizedByTargetsOnlyAsOutput(candidate);

            List<int> stale = context.InputShape != null
                ? IncompatibleLayers(candidate, Plan(candidate, context.InputShape))
                : new List<int>();

            foreach (Layer layer in added)
            {
                layer.Context = context;
            }

            // Layers that left the model are detached from it, so they can be put into another
            // model (see RequireUsableLayer) — a layer keeps its weights, it just no longer belongs here.
            foreach (Layer layer in layersList)
            {
                if (!candidate.Contains(layer))
                {
                    layer.Context = null;
                }
            }

            layersList = candidate;

            foreach (int i in stale)
            {
                layersList[i].Invalidate();
            }

            // No built layer left (ReplaceAll, or a change that shifted every layer's input): there are
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

        // ---------------------------------------------------------------- one layer instance, one place
        //
        // A layer keeps the state of ONE pass (its outputs, the inputs in its neurons, the dropout
        // mask, the accumulated gradients). The same instance at two positions — or in two models —
        // shares all of it: the second pass overwrites what the first one left for the backward
        // pass. Measured with the same Dense at two positions: the gradient of its weight came out
        // −0.0063 instead of 0.0736 — the wrong SIGN, so the model would silently learn in the
        // wrong direction. Sharing weights between positions is a real technique (tied weights),
        // but it needs its own support; until then an error beats a silently wrong gradient.

        // `chain` = the layers the new one would join (the model's list, or the part of a new list
        // checked so far). A layer of THIS model that is not in `chain` is fine — Replace may reuse it.
        private void RequireUsableLayer(Layer layer, IReadOnlyList<Layer> chain, string parameterName)
        {
            int existing = -1;
            for (int i = 0; i < chain.Count; i++)
            {
                if (ReferenceEquals(chain[i], layer))
                {
                    existing = i;
                    break;
                }
            }

            if (existing >= 0)
            {
                throw new ArgumentException(
                    $"This {layer.Name} instance is already in the model (layer {existing + 1}). Create a new layer " +
                    "instead — the same instance at two positions would share its weights and buffers, and the " +
                    "gradients would silently come out wrong.", parameterName);
            }

            if (layer.Context != null && !ReferenceEquals(layer.Context, context))
            {
                throw new ArgumentException(
                    $"This {layer.Name} instance already belongs to another model. Create a new layer, or remove it " +
                    "from the other model first — two models would overwrite each other's state in it.", parameterName);
            }
        }

        // Dense.FromTargets() gets its size from the targets — which only the OUTPUT layer is compared to.
        private static void RequireSizedByTargetsOnlyAsOutput(List<Layer> chain)
        {
            for (int i = 0; i < chain.Count - 1; i++)
            {
                if (chain[i] is Dense { SizeFromTargets: true })
                {
                    throw new ArgumentException(
                        $"Layer {i + 1} is Dense.FromTargets(), which can only be the output layer — its size comes from " +
                        "the targets. Use new Dense(n, …) for a hidden layer.");
                }
            }
        }

        // A list of layers: no null, every instance usable once; empty only where allowed (hidden layers).
        private List<Layer> RequireUsableLayers(IEnumerable<Layer> layers, string parameterName, bool allowEmpty = false)
        {
            ArgumentNullException.ThrowIfNull(layers, parameterName);

            var result = new List<Layer>();
            foreach (Layer? layer in layers)
            {
                if (layer == null)
                {
                    throw new ArgumentException($"Layer {result.Count + 1} in the list is null.", parameterName);
                }

                RequireUsableLayer(layer, result, parameterName);
                result.Add(layer);
            }

            if (result.Count == 0 && !allowEmpty)
            {
                throw new ArgumentException(
                    "The list of layers is empty — a model needs at least its output layer.", parameterName);
            }

            return result;
        }

        // ---------------------------------------------------------------- hidden layers (via Hidden)

        // Positions are positions among the hidden layers; HiddenLayers has checked their range.
        // The hidden layers are the start of the chain, so a hidden position is also the chain position.

        internal void InsertHidden(int position, Layer layer)
        {
            RequireUsableLayer(layer, layersList, "layer");
            ChangeStructure(list => list.Insert(position, layer), [layer]);
        }

        // Hidden[position] = layer. The layer being replaced is left out of the duplicate check, so
        // putting the same instance back is a no-op; any OTHER layer of the model (hidden or output) is a duplicate.
        internal void SetHidden(int position, Layer layer)
        {
            List<Layer> others = layersList.Where((_, i) => i != position).ToList();
            RequireUsableLayer(layer, others, "value");
            ChangeStructure(list => list[position] = layer, [layer]);
        }

        internal void RemoveHidden(int position)
        {
            ChangeStructure(list => list.RemoveAt(position), []);
        }

        internal void ReplaceHidden(IEnumerable<Layer> layers)
        {
            List<Layer> incoming = RequireUsableLayers(layers, nameof(layers), allowEmpty: true);

            if (incoming.Any(layer => ReferenceEquals(layer, Output)))
            {
                throw new ArgumentException(
                    "The output layer cannot also be a hidden layer — the same instance at two positions would " +
                    "share its weights and buffers.", nameof(layers));
            }

            Layer output = Output;
            ChangeStructure(list => { list.Clear(); list.AddRange(incoming); list.Add(output); }, incoming);
        }

        // ---------------------------------------------------------------- public API

        /// <summary>
        /// Adds a hidden layer just before the output layer — a shortcut for <c>Hidden.Add(layer)</c>.
        /// </summary>
        /// <exception cref="ArgumentException">The layer instance is already in this or another model.</exception>
        public void Add(Layer layer) => Hidden.Add(layer);

        /// <summary>
        /// Replaces the output layer. The only way to change it — the methods of <see cref="Hidden"/>
        /// never touch it.
        /// </summary>
        /// <remarks>
        /// The hidden layers keep their weights; the new output layer is built on the next pass and the
        /// old one is detached from the model (it can be used elsewhere). Passing the current output
        /// layer does nothing.
        /// </remarks>
        /// <exception cref="ArgumentException">The layer is one of the hidden layers, or belongs to another model.</exception>
        public void ReplaceOutput(Layer outputLayer)
        {
            ArgumentNullException.ThrowIfNull(outputLayer);

            // The current output layer passes the check (it is not among the hidden layers) and is
            // put back in its place — a no-op without a special case.
            RequireUsableLayer(outputLayer, Hidden.ToList(), nameof(outputLayer));
            ChangeStructure(list => list[^1] = outputLayer, [outputLayer]);
        }

        /// <summary>
        /// Replaces the whole architecture: the model gets exactly <paramref name="layers"/>, in order from
        /// input to output; the last one is the output layer.
        /// </summary>
        /// <remarks>
        /// Optimizer, loss, seed, datasets and training settings stay. A layer of the current model may be
        /// reused in the new list (it keeps its weights if its input does not change); layers left out are
        /// detached from the model and can be used elsewhere. To replace only one part, use
        /// <c>Hidden.Replace</c> or <see cref="ReplaceOutput"/>.
        /// </remarks>
        /// <exception cref="ArgumentException">The list is empty, contains null, or a layer instance twice
        /// or from another model.</exception>
        public void ReplaceAll(IEnumerable<Layer> layers)
        {
            List<Layer> incoming = RequireUsableLayers(layers, nameof(layers));
            ChangeStructure(list => { list.Clear(); list.AddRange(incoming); }, incoming);
        }
    }
}
