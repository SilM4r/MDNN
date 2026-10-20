using MDNN.Optimizers;
using MDNN.Activations;

namespace MDNN.Layers
{
    /// <summary>
    /// Base class of all network layers: shape wiring, parameter materialization, forward and backward pass.
    /// </summary>
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

        /// <summary>Shape of the layer input.</summary>
        abstract public int[] InputShape { get; }
        /// <summary>Shape of the layer output.</summary>
        abstract public int[] OutputShape { get; }
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
        /// <summary>
        /// Wires the layer shapes and then materializes its parameters.
        /// </summary>
        /// <remarks>
        /// Building a layer is split into TWO independent phases:
        /// <list type="bullet">
        /// <item><see cref="WireShapes"/> computes the shapes (input, output, buffers). LayerManager
        /// needs it to chain the layers; it may be called any number of times — it draws no random numbers.</item>
        /// <item><see cref="MaterializeParameters"/> creates and RANDOMLY INITIALIZES the parameters.
        /// It is called exactly once, in a single left-to-right pass, before the first forward.</item>
        /// </list>
        /// Why split: while it was a single method, materialization already ran while the model was
        /// being assembled. The random draw order thus depended on the HISTORY of Add/Insert calls,
        /// not on the final architecture — two identical networks with the same seed got different
        /// weights depending on how they were assembled. The output layer was also materialized three
        /// times when two layers were added, and an inserted layer was not wired at all, which ended
        /// with IndexOutOfRange in the forward pass of a loaded model.
        /// </remarks>
        public void Build(int? outputSize = null, int[]? inputShape = null)
        {
            WireShapes(outputSize, inputShape);
            MaterializeParameters();
        }

        /// <summary>
        /// Computes the input/output shapes and buffers. Safe to call repeatedly; draws no random numbers.
        /// </summary>
        abstract public void WireShapes(int? outputSize = null, int[]? inputShape = null);

        /// <summary>
        /// Creates and randomly initializes the parameters for the current shape.
        /// </summary>
        /// <remarks>
        /// Idempotent: if parameters for the current shape already exist, they are left alone
        /// (otherwise trained weights would be overwritten).
        /// </remarks>
        abstract public void MaterializeParameters();

        /// <summary>
        /// True if the layer has parameters matching its current shape. False means
        /// "shape is wired but parameters do not exist yet" or "shape is not known yet".
        /// </summary>
        abstract public bool IsMaterialized { get; }

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

    }
}
