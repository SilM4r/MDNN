using MDNN.Optimizers;
using MDNN.Activations;
using MDNN.Serialization;

namespace MDNN.Layers
{
    /// <summary>
    /// Base class of all network layers: building for an input shape, forward and backward pass.
    /// </summary>
    /// <remarks>
    /// A layer has two states. A freshly constructed layer is only a DESCRIPTION — how many
    /// neurons or filters, which activation — without any weights, because it does not know its
    /// input yet. <see cref="Build"/> turns it into a BUILT layer for a concrete input shape:
    /// buffers are allocated and the parameters randomly initialized. Inside a model the
    /// <see cref="LayerManager"/> does this on the first pass, when the shape of the data is known.
    /// </remarks>
    public abstract class Layer
    {
        /// <summary>
        /// Context of the model this layer belongs to (set by LayerManager when the layer is attached).
        /// </summary>
        public MDNN.NetworkContext? Context { get; set; }

        /// <summary>
        /// True if this is the last layer of the model. LayerManager keeps it up to date on every chain change.
        /// </summary>
        /// <remarks>
        /// Currently needed for one thing only: GLOBAL dropout must NOT be applied to the output
        /// layer — dropping network outputs is not regularization, it corrupts the prediction.
        /// Dropout set explicitly on a specific layer overrides this rule (it is a deliberate choice).
        /// </remarks>
        public bool IsOutputLayer { get; internal set; }

        /// <summary>Layer type name (also used as the type key in saved models).</summary>
        abstract public string Name { get; }

        /// <summary>
        /// True once the layer has been built for an input shape (it has its parameters and buffers).
        /// False = the layer is only a description, see <see cref="Build"/>.
        /// </summary>
        public bool IsBuilt { get; private set; }

        /// <summary>
        /// Shape of the input the layer is built for, in the form the layer consumes it — flattened
        /// to <c>[n]</c> for Dense and RNN, <c>[height, width, channels]</c> for Conv and MaxPool.
        /// <c>null</c> until the layer is built.
        /// </summary>
        public int[]? InputShape => inputShape;

        /// <summary>Shape of the layer output; <c>null</c> until the layer is built.</summary>
        public int[]? OutputShape => outputShape;

        // Previously every layer kept its own shape fields with {0} meaning "not known yet", and
        // that sentinel was checked in ten places, each a little differently (<= 0, == 0,
        // Length == 0, Any(d <= 0)). Now the shapes live here, null means "not built" and only
        // Build/Invalidate change them.
        private int[]? inputShape;
        private int[]? outputShape;

        /// <summary>Output of the last forward pass, after activation.</summary>
        abstract public Tensor? Output { get; }
        /// <summary>Output of the last forward pass, before activation.</summary>
        abstract public Tensor? RawOutput { get; }
        /// <summary>Activation function applied to the layer output.</summary>
        abstract public ActivationFunction Activation { get; set; }
        /// <summary>Runs the forward pass and returns the layer output.</summary>
        abstract public Tensor FeedForward(Tensor input);
        /// <summary>Applies the accumulated gradients to the parameters via the optimizer.</summary>
        abstract public void UpdateParams();

        // ---------------------------------------------------------------- building

        /// <summary>
        /// Builds the layer for <paramref name="inputShape"/>: allocates its buffers and randomly
        /// initializes its parameters. Inside a model this is done by the model itself on the first
        /// pass — call it directly only for a standalone layer.
        /// </summary>
        /// <remarks>
        /// Calling it again with the same shape does nothing (trained weights are kept). Calling it
        /// with a DIFFERENT shape is an error: that would silently replace trained weights with
        /// random ones. Inside a model, use <c>model.Layers.Rebuild()</c> or
        /// <c>model.Layers.RebuildIncompatible()</c> to rebuild on purpose.
        /// </remarks>
        /// <param name="inputShape">Shape of one input sample.</param>
        /// <exception cref="InvalidOperationException">The layer is already built for a different shape.</exception>
        public void Build(int[] inputShape)
        {
            ArgumentNullException.ThrowIfNull(inputShape);

            // Checked here, not only in LayerManager: a standalone Build([0]) used to give a "built"
            // Dense with zero weights per neuron.
            RequirePositiveShape(inputShape);

            int[] normalized = NormalizeInputShape(inputShape);

            if (IsBuilt)
            {
                if (SpatialShape.SameAs(normalized, this.inputShape!))
                {
                    return;
                }

                throw new InvalidOperationException(
                    $"{Name} is already built for input {SpatialShape.Describe(this.inputShape!)} and cannot be " +
                    $"rebuilt for {SpatialShape.Describe(normalized)} — that would silently replace its weights " +
                    "with random ones. Inside a model use model.Layers.Rebuild() or model.Layers.RebuildIncompatible().");
            }

            // The output shape is computed (and the input validated) BEFORE anything is allocated
            // or drawn, so an input that does not fit leaves the layer untouched.
            int[] output = ComputeOutputShape(normalized);

            // The random source and the optimizer come from the model; a standalone layer falls back
            // to the global defaults. Read only here, when something is really built — reading
            // Context.Random marks the generator as used (see NetworkContext.Seed).
            BuildCore(normalized,
                      Context?.Random ?? GeneralNeuralNetworkSettings.Random,
                      Context?.Optimizer ?? GeneralNeuralNetworkSettings.DefaultOptimizer);

            this.inputShape = normalized;
            outputShape = output;
            IsBuilt = true;
        }

        /// <summary>
        /// Converts an incoming shape to the form the layer works with. The default keeps it as it
        /// is; Dense flattens it, Conv and MaxPool turn it into <c>[height, width, channels]</c>.
        /// </summary>
        protected virtual int[] NormalizeInputShape(int[] inputShape) => (int[])inputShape.Clone();

        /// <summary>
        /// Computes the output shape for an (already normalized) input shape and throws if the
        /// layer cannot take such an input. Must not change the layer — the model calls it to plan
        /// the whole chain before building any of it.
        /// </summary>
        protected abstract int[] ComputeOutputShape(int[] inputShape);

        /// <summary>
        /// Allocates the buffers and creates the randomly initialized parameters for
        /// <paramref name="inputShape"/> (already normalized and validated by <see cref="ComputeOutputShape"/>).
        /// </summary>
        /// <param name="inputShape">Normalized input shape.</param>
        /// <param name="random">Random source for the initialization.</param>
        /// <param name="optimizerPrototype">Optimizer whose fresh clone the parameters get.</param>
        protected abstract void BuildCore(int[] inputShape, Random random, Optimizer optimizerPrototype);

        /// <summary>
        /// Called when the layer goes back to being a description (its input shape changed).
        /// Release the parameters and buffers here.
        /// </summary>
        protected virtual void OnInvalidated() { }

        // For LayerManager: planning the chain and checking incoming data.
        internal int[] NormalizeForPlan(int[] inputShape) => NormalizeInputShape(inputShape);
        internal int[] OutputShapeFor(int[] normalizedInputShape) => ComputeOutputShape(normalizedInputShape);

        // Would the built layer take a sample of this shape as it is?
        internal bool Accepts(int[] shape) =>
            IsBuilt && SpatialShape.SameAs(NormalizeInputShape(shape), inputShape!);

        // Back to a description: the layer's input changed (the user changed the structure of the
        // model, or asked for a rebuild) and it will be built again on the next pass.
        internal void Invalidate()
        {
            IsBuilt = false;
            inputShape = null;
            outputShape = null;
            OnInvalidated();
        }

        // A layer loaded from a file is built straight away — its parameters come from the file,
        // so nothing is drawn.
        private protected void MarkBuiltFromFile(int[] normalizedInputShape)
        {
            outputShape = ComputeOutputShape(normalizedInputShape);
            inputShape = normalizedInputShape;
            IsBuilt = true;
        }

        internal static void RequirePositiveShape(int[] shape)
        {
            if (shape.Length == 0 || shape.Any(size => size <= 0))
            {
                throw new ArgumentException(
                    $"The number of inputs to the first layer must be greater than zero (got shape [{string.Join(", ", shape)}]).");
            }
        }

        // For the error message of a layer that gets a sample of a different shape than it is built for.
        internal const string ShapeChangeHint =
            "The input shape must not change once the model is built. To use the model with differently " +
            "shaped data, call model.Layers.Rebuild() (new weights for all layers) or " +
            "model.Layers.RebuildIncompatible() (new weights only for the layers that no longer fit).";

        // For the error message when the TARGETS change size under an output layer sized by them.
        internal const string TargetSizeChangeHint =
            "The size of the targets must not change once the output layer is built. To train the model on targets " +
            "of another size, call model.Layers.RebuildIncompatible() (a new output layer, the hidden layers keep " +
            "their weights) or model.Layers.Rebuild() (new weights for all layers).";

        // ---------------------------------------------------------------- loading into an existing layer
        //
        // Restoring a snapshot (the best model at the end of TrainLoop) or LoadWeightsInPlace used to
        // build NEW layer objects from the file. A reference the user held to a layer then pointed
        // outside the model, and a layer with a custom activation could not be rebuilt at all — the
        // file only has its name. When the saved layer matches this one, the parameters are copied
        // into it instead.

        // Does the saved layer have exactly this layer's architecture? Must not change anything.
        internal virtual bool CanImport(BaseExportLayer saved) => false;

        // Loading in two steps, so that a damaged file cannot leave the model half loaded:
        //
        //   PrepareImport  — everything that can FAIL (validating the dropout rate, loading the
        //                    optimizer state into fresh optimizer instances); changes nothing
        //   the returned action — only copies the prepared values into the layer; cannot fail
        //
        // The model prepares ALL layers first and only then runs the actions. Previously a broken
        // optimizer state in the third layer threw after the first two already had the file's weights.
        internal virtual Action PrepareImport(BaseExportLayer saved, Optimizer optimizerPrototype)
        {
            throw new NotSupportedException($"Layer {GetType().Name} cannot load parameters.");
        }

        // ---------------------------------------------------------------- convention B
        //
        /// <summary>
        /// The single entry point of the backward pass: takes ∂L/∂output of this layer, accumulates
        /// the gradients of its OWN parameters and returns ∂L/∂input (∂L/∂output of the layer below).
        /// </summary>
        /// <remarks>
        /// <para>
        /// The layer propagates the gradient through ITS OWN weights and does not need to know about
        /// its neighbour. Previously the backward pass was split across several methods with an
        /// off-by-one: Conv and MaxPool had their own "next layer is Dense" branches and hidden state
        /// (Conv.dOutput, RNN.gImm) existed only because of that split; this method removed both.
        /// </para>
        /// <para>
        /// <paramref name="preActivation"/>: gradOutput ALREADY IS ∂L/∂z, so the activation derivative is
        /// skipped. The only case is the fused softmax+CrossEntropy, where the loss returns <c>s − t</c>
        /// directly: softmax logically belongs to the loss there, not to the layer (just as PyTorch's
        /// CrossEntropyLoss contains log-softmax). This keeps a single entry point instead of two.
        /// </para>
        /// <para>
        /// <paramref name="computeInputGradient"/> = false means "nobody needs ∂L/∂input, do not compute it".
        /// This applies to the FIRST layer of the model: there is nothing below it, so its input gradient
        /// would be thrown away. For Conv that is a full correlation over the whole image — the most
        /// expensive backward operation. Analogous to <c>needs_input_grad</c> in PyTorch.
        /// The default is true, which is always correct; only a caller that KNOWS it will not use the
        /// return value may turn it off. With false an EMPTY tensor is returned, not a meaningless one.
        /// </para>
        /// </remarks>
        /// <param name="gradOutput">∂L/∂output of this layer.</param>
        /// <param name="preActivation">True if <paramref name="gradOutput"/> is already the gradient before activation (∂L/∂z).</param>
        /// <param name="computeInputGradient">False to skip computing ∂L/∂input (an empty tensor is returned).</param>
        /// <returns>∂L/∂input of this layer, or an empty tensor when <paramref name="computeInputGradient"/> is false.</returns>
        abstract public Tensor Backward(Tensor gradOutput, bool preActivation = false,
                                        bool computeInputGradient = true);

        // ---------------------------------------------------------------- replicas
        //
        // Data-parallel training (Train.Workers > 1): each thread has its own model replica
        // and processes its part of the batch. A replica SHARES parameters with the original
        // (they are only read during the batch) and OWNS everything that changes during a pass —
        // activations, masks, gradients.
        //
        // null = the layer does not support parallel training. The default is deliberately null,
        // not MemberwiseClone: a shallow copy would share working buffers and two threads would
        // overwrite each other's activations — silently, without a crash. Training rejects it instead.
        internal virtual Layer? CreateReplica() => null;

        // Before each batch: copy into the replica the parameters that cannot be shared by reference.
        internal virtual void SyncReplica(Layer replica) { }

        // After the batch: add the replica's gradients to our own and reset the replica.
        internal virtual void TakeGradientsFrom(Layer replica) { }

        // The model got a different optimizer (NetworkContext.Optimizer) — the layer takes a
        // fresh clone of it. Layers without parameters have nothing to change.
        internal virtual void ReplaceOptimizer(Optimizer prototype) { }

        // A replica is a copy of a BUILT layer; an unbuilt one has no parameters to share.
        private protected void ThrowIfNotBuiltForReplica()
        {
            if (!IsBuilt)
            {
                throw new InvalidOperationException(
                    $"{Name}: a replica for parallel training can only be made of a built layer.");
            }
        }
    }
}
