using MDNN.Activations;
using MDNN.Layers;
using MDNN.Serialization;
using MDNN.Optimizers;

namespace MDNN.Layers
{

    /// <summary>
    /// Fully connected layer: every neuron sees the whole (flattened) input. Supports dropout.
    /// </summary>
    public class Dense : NeuronLayer
    {
        /// <inheritdoc/>
        public override List<Neuron> Neurons => neurons;
        /// <inheritdoc/>
        public override string Name => "Dense";
        // A snapshot, not a window into the working array — see Tensor.FromVector.
        /// <inheritdoc/>
        public override Tensor Output => Tensor.FromVector(output);
        /// <inheritdoc/>
        public override Tensor RawOutput => Tensor.FromVector(raw_output);
        /// <inheritdoc/>
        public override ActivationFunction Activation
        {
            get
            {
                return activation_func ?? throw new InvalidOperationException(
                    "The activation of this output layer is chosen from the loss when the layer is built " +
                    "(CrossEntropy → Softmax, MSE → Linear) — it is not known yet.");
            }
            set
            {
                ArgumentNullException.ThrowIfNull(value);
                activation_func = value;
                ActivationFromLoss = false;   // set explicitly — no longer chosen from the loss
                foreach (Neuron neuron in neurons)
                {
                    neuron.Activation = activation_func;
                }
            }
        }

        // Number of inputs (the flattened input shape); 0 until the layer is built.
        private int inputCount;

        // Output buffers have the size of the layer (NeuronCount), which is known from the
        // description, so they are allocated right in the constructor.
        private double[] output;
        private double[] raw_output;
        private List<Neuron> neurons;

        // null only for Dense.FromTargets() without an activation, until it is chosen from the loss
        private ActivationFunction? activation_func;

        // ---------------------------------------------------------------- dropout

        private double? dropout;

        // Mask from the last forward pass: 0 = neuron was dropped, 1/(1-p) = it survived,
        // 1 = dropout did not apply (prediction or rate 0). Backward MUST use the SAME
        // mask, so it is kept between forward and backward instead of being regenerated.
        private double[]? dropoutMask;

        /// <summary>
        /// Dropout rate for THIS layer. <c>null</c> (default) = use the global
        /// <see cref="NetworkContext.Dropout"/>. A layer-specific value overrides the global one —
        /// including zero, which turns dropout off for this layer.
        /// </summary>
        public double? DropoutRate
        {
            get => dropout;
            set
            {
                if (value != null) ValidateDropout(value.Value);
                dropout = value;
            }
        }

        private static void ValidateDropout(double value)
        {
            if (double.IsNaN(value) || value < 0 || value >= 1)
                throw new ArgumentOutOfRangeException(nameof(DropoutRate), value,
                    "Dropout must be in the range [0, 1). 0 = off; 1 would drop " +
                    "every neuron and the scaling factor 1/(1-p) would divide by zero.");
        }

        /// <summary>The rate actually applied: the layer's own value, otherwise the global one —
        /// which is DELIBERATELY ignored on the output layer and on a softmax layer
        /// (see NetworkContext.Dropout and PrepareDropoutMask).</summary>
        private double EffectiveDropout
        {
            get
            {
                if (dropout != null) return dropout.Value;
                if (IsOutputLayer || activation_func!.AppliesToLayer) return 0.0;
                return Context?.Dropout ?? 0.0;
            }
        }

        /// <summary>
        /// Creates a dense layer. This is only a description — the weights are created when the
        /// layer is built for its input (inside a model: on the first pass).
        /// </summary>
        /// <param name="neuronCount">Number of neurons (= size of the output).</param>
        /// <param name="activation">Activation function; <c>null</c> = the default hidden activation.</param>
        /// <param name="dropout">Fraction of neurons dropped during training, 0 ≤ p &lt; 1.
        /// null = inherit the global <see cref="NetworkContext.Dropout"/>.</param>
        public Dense(int neuronCount, ActivationFunction? activation = null, double? dropout = null)
        {
            if (neuronCount <= 0)
            {
                throw new ArgumentException("The number of neurons in a layer must be greater than 0");
            }

            if (dropout != null) ValidateDropout(dropout.Value);
            this.dropout = dropout;

            this.activation_func = activation ?? GeneralNeuralNetworkSettings.DefaultHiddenActivation;

            // Previously the constructor also created `neuronCount` placeholder neurons with ZERO
            // weights (the real ones were built later). The layer then looked built while it was
            // not, and the size the user asked for could only be read from the placeholders.
            // Now the description is NeuronCount and the neurons exist only once built.
            NeuronCount = neuronCount;
            neurons = new List<Neuron>();

            output = new double[neuronCount];
            raw_output = new double[neuronCount];
        }
        // ---------------------------------------------------------------- size from the targets
        //
        // The output layer a model gets by default. The input of a model needs no layer — the first
        // layer adapts to the shape of the data — and the output can follow the data the same way:
        // its size is the size of one target, and its activation follows from the loss
        // (CrossEntropy needs Softmax, MSE a Linear output). The layer stays an ordinary Dense; it only
        // learns its size later, when the model first sees targets (Fit, TrainLoop, Evaluate).

        /// <summary>
        /// An output layer whose number of neurons comes from the targets (the size of one target) — the
        /// output layer a model gets when none is given. The activation is chosen from the loss
        /// (<c>CrossEntropy</c> → <c>Softmax</c>, <c>MSE</c> → <c>Linear</c>) unless given here.
        /// </summary>
        /// <remarks>
        /// Can only be the output layer. Until the model has seen targets the size is not known, so a
        /// model with such an output cannot predict before it is trained.
        /// Use it to go back to the automatic output: <c>model.Layers.ReplaceOutput(Dense.FromTargets())</c>.
        /// </remarks>
        /// <param name="activation">The activation; <c>null</c> = chosen from the loss.</param>
        public static Dense FromTargets(ActivationFunction? activation = null) => new Dense(activation, sizeFromTargets: true);

        private Dense(ActivationFunction? activation, bool sizeFromTargets)
        {
            SizeFromTargets = sizeFromTargets;
            activation_func = activation;
            ActivationFromLoss = activation == null;
            NeuronCount = 0;   // learned from the targets (ResolveFromTargets)
            neurons = new List<Neuron>();
            output = Array.Empty<double>();
            raw_output = Array.Empty<double>();
        }

        /// <summary>True if the number of neurons comes from the targets (<see cref="FromTargets"/>).</summary>
        public bool SizeFromTargets { get; private set; }

        /// <summary>True if the activation is chosen from the loss (<see cref="FromTargets"/> without an activation).</summary>
        public bool ActivationFromLoss { get; private set; }

        // The model has seen targets: the layer gets its size (and, if it has none, its activation).
        internal void ResolveFromTargets(int targetCount, MDNN.Losses.Loss loss)
        {
            if (targetCount <= 0)
            {
                throw new ArgumentException($"A target must have at least one value, got {targetCount}.");
            }

            ActivationFunction activation = activation_func ?? ActivationForLoss(loss);

            activation_func = activation;
            NeuronCount = targetCount;
            output = new double[targetCount];
            raw_output = new double[targetCount];
        }

        // Rebuild(): the next targets decide the size (and the activation, if it came from the loss) again.
        internal void ForgetTargetSize()
        {
            if (!SizeFromTargets) return;

            NeuronCount = 0;
            output = Array.Empty<double>();
            raw_output = Array.Empty<double>();
            if (ActivationFromLoss) activation_func = null;
        }

        // Which activation the output needs for this loss. A custom loss does not say, and a silent
        // guess (e.g. Linear) would be exactly the kind of substitution the library avoids elsewhere.
        private static ActivationFunction ActivationForLoss(MDNN.Losses.Loss loss)
        {
            if (loss.RequiresSoftmax) return new Softmax();
            if (loss is MDNN.Losses.MSE) return new Linear();

            throw new InvalidOperationException(
                $"The output layer takes its activation from the loss, but for loss '{loss.Name}' it cannot tell " +
                "which one. Give it: model.Layers.ReplaceOutput(Dense.FromTargets(new Sigmoid())), or pass the " +
                "output layer to the constructor: outputLayer: new Dense(n, …).");
        }

        // `optimizerPrototype` = optimizer of the model being loaded; each neuron gets its own clone
        // (with the state from the file). null = global default, see Neuron.
        internal Dense(ExportDenseLayer layer, Optimizer? optimizerPrototype = null)
        {
            activation_func = ActivationFunction.FromName(layer.Name_of_activation_function);

            // Older files lack this key → null → the layer inherits the global value,
            // which is also the default for a new model. Backward compatible.
            if (layer.DropoutRate != null) ValidateDropout(layer.DropoutRate.Value);
            dropout = layer.DropoutRate;

            List<ExportNeuron> saved = BaseExportLayer.Required(layer.Neurons, Name, nameof(layer.Neurons));
            BaseExportLayer.RequireNotEmpty(saved, Name, nameof(layer.Neurons));

            NeuronCount = saved.Count;
            output = new double[saved.Count];
            raw_output = new double[saved.Count];

            neurons = new List<Neuron>();

            foreach (ExportNeuron neuron in saved)
            {
                neurons.Add(new Neuron(neuron, activation_func, optimizerPrototype));
            }

            inputCount = neurons[0].Weights.Length;

            // older files lack these keys → false = an ordinary fixed-size layer
            SizeFromTargets = layer.SizeFromTargets;
            ActivationFromLoss = layer.ActivationFromLoss;

            MarkBuiltFromFile(new int[] { inputCount });
        }

        // ---------------------------------------------------------------- building

        /// <summary>Dense flattens any input to a vector: <c>[28, 28]</c> becomes <c>[784]</c>.</summary>
        protected override int[] NormalizeInputShape(int[] inputShape) => new int[] { Flatten(inputShape) };

        internal static int Flatten(int[] shape)
        {
            int size = 1;
            foreach (int dimension in shape)
            {
                size *= dimension;
            }
            return size;
        }

        /// <inheritdoc/>
        protected override int[] ComputeOutputShape(int[] inputShape)
        {
            if (NeuronCount == 0)
            {
                throw new InvalidOperationException(
                    "The size of this output layer comes from the targets, which are not known yet.");
            }

            return new int[] { NeuronCount };
        }

        /// <inheritdoc/>
        protected override void BuildCore(int[] inputShape, Random random, Optimizer optimizerPrototype)
        {
            // Built into a local list and assigned to `neurons` only when COMPLETE: if cloning the
            // optimizer throws, the layer stays unbuilt instead of keeping new neurons with the
            // constructor's GLOBAL optimizer.
            var built = new List<Neuron>(NeuronCount);

            // All neurons are drawn first, then the optimizers are cloned — the same order of random
            // draws as before, so a seeded model gets bit-for-bit the same weights.
            for (int i = 0; i < NeuronCount; i++)
            {
                built.Add(new Neuron(inputShape[0], activation_func!, random));
            }

            // Per-model optimizer: each neuron gets ITS OWN clone (the optimizer state is per parameter).
            foreach (Neuron n in built)
            {
                n.Optimizer = optimizerPrototype.CreateFresh();
            }

            neurons = built;
            inputCount = inputShape[0];
            dropoutMask = null;
        }

        /// <inheritdoc/>
        protected override void OnInvalidated()
        {
            neurons = new List<Neuron>();
            inputCount = 0;
            dropoutMask = null;
            Array.Clear(output);
            Array.Clear(raw_output);
        }

        internal override bool CanImport(BaseExportLayer saved)
        {
            return IsBuilt
                   && saved is ExportDenseLayer dense
                   && dense.Name_of_activation_function == activation_func?.Name
                   && dense.Neurons != null
                   && dense.Neurons.Count == NeuronCount
                   && dense.Neurons.All(n => n.Weights.Length == inputCount);
        }

        internal override Action PrepareImport(BaseExportLayer saved, Optimizer optimizerPrototype)
        {
            var dense = (ExportDenseLayer)saved;
            List<ExportNeuron> savedNeurons = dense.Neurons!;

            // what can fail — nothing is written yet
            if (dense.DropoutRate != null) ValidateDropout(dense.DropoutRate.Value);
            Optimizer[] optimizers = savedNeurons.Select(n => Neuron.PrepareOptimizer(n, optimizerPrototype)).ToArray();

            return () =>
            {
                dropout = dense.DropoutRate;

                for (int i = 0; i < neurons.Count; i++)
                {
                    neurons[i].ImportFrom(savedNeurons[i], optimizers[i]);
                }

                Array.Clear(output);
                Array.Clear(raw_output);
                dropoutMask = null;
            };
        }

        // Will this layer draw a dropout mask during training? Replicas use this to decide
        // whether they need their own random generator — see ReplicaPool.
        internal bool UsesDropout => EffectiveDropout > 0.0;

        // Replica for parallel training — see Layer.CreateReplica. Shared: weights (via
        // neuron replicas), activation, dropout settings. Own: outputs and the dropout mask,
        // which change with every sample.
        internal override Layer CreateReplica()
        {
            ThrowIfNotBuiltForReplica();

            Dense replica = (Dense)MemberwiseClone();
            replica.output = new double[output.Length];
            replica.raw_output = new double[raw_output.Length];
            replica.dropoutMask = null;
            replica.neurons = neurons.Select(n => n.CreateReplica()).ToList();
            return replica;
        }

        /// <inheritdoc/>
        public override Tensor FeedForward(Tensor input)
        {
            if (!IsBuilt)
            {
                throw new InvalidOperationException(
                    $"{Name} is not built yet — add it to a model and run a sample through it, or call Build(inputShape).");
            }

            double[] values = input.Data;

            // Previously nothing checked this: a SHORTER input was silently multiplied with only the
            // first weights and gave a meaningless output, a LONGER one crashed with
            // IndexOutOfRange deep in Neuron.FeedForward. Conv and MaxPool already checked it.
            if (values.Length != inputCount)
            {
                throw new ArgumentException(
                    $"{Name} is built for {inputCount} inputs, but got {values.Length} " +
                    $"(shape {SpatialShape.Describe(input.Shape)}). {ShapeChangeHint}");
            }

            // The mask must be known BEFORE the computation so dropped neurons can be
            // skipped — at p=0.5 that saves half of this layer's dot products.
            // A softmax layer never has a mask (PrepareDropoutMask rejects it).
            PrepareDropoutMask();

            for (int i = 0; i < neurons.Count; i++)
            {
                if (dropoutMask != null && dropoutMask[i] == 0.0)
                {
                    // A dropped neuron is NOT computed. Zeros are set explicitly so the arrays
                    // don't keep values from the previous sample.
                    output[i] = 0;
                    raw_output[i] = 0;
                    neurons[i].Output = 0;
                    neurons[i].RawOutput = 0;
                    continue;
                }

                output[i] = neurons[i].FeedForward(values);
                raw_output[i] = neurons[i].RawOutput;
            }

            if (activation_func!.AppliesToLayer)
            {
                LayerActivationFunction? layerActivationFunc = activation_func as LayerActivationFunction;

                if (layerActivationFunc == null)
                {
                    throw new ArgumentException("Bad activation func");
                }

                output = layerActivationFunc.ApplyToLayer(raw_output);

                for (int i = 0; i < neurons.Count; i++)
                {
                    neurons[i].Output = output[i];
                }
            }

            ApplyDropoutMask();

            return Tensor.FromVector(output);
        }

        // Test hook. A gradient check calls forward repeatedly and compares how the loss
        // changes — that only makes sense if it is the SAME function every time. With a
        // freshly drawn mask it would keep changing and the check would measure nothing.
        // Freezing turns dropout into plain multiplication by a constant vector, which can be verified.
        internal bool FreezeDropoutMaskForTests { get; set; }

        internal double[]? DropoutMaskForTests => dropoutMask;

        // Draws the mask for this sample. null = dropout does not apply at all (off,
        // or prediction is running) — then nothing is multiplied or allocated.
        private void PrepareDropoutMask()
        {
            double rate = EffectiveDropout;
            bool training = Context?.IsTraining ?? false;

            if (rate == 0.0 || !training)
            {
                dropoutMask = null;
                return;
            }

            // Softmax normalizes over the WHOLE layer, so even a dropped neuron affects the
            // other outputs and its weight gradient is not zero — skipping it in backward
            // (as with other activations) would give a wrong gradient. And with the softmax + CE
            // fusion the loss would see an output with some probabilities missing. Dropout on
            // softmax makes no sense anyway (it drops the prediction), hence an error rather than
            // silent nonsense. Global dropout never reaches a softmax layer (EffectiveDropout);
            // only an explicitly set rate gets here.
            if (activation_func!.AppliesToLayer)
            {
                throw new NotSupportedException(
                    $"Dropout cannot be used on a layer with a layer-wide activation ({activation_func.Name}). " +
                    "Remove DropoutRate from this layer, or put the dropout on the layer before it.");
            }

            if (FreezeDropoutMaskForTests && dropoutMask != null
                && dropoutMask.Length == neurons.Count)
            {
                return;
            }

            if (dropoutMask == null || dropoutMask.Length != neurons.Count)
            {
                dropoutMask = new double[neurons.Count];
            }

            // Scaling up the survivors keeps the expected output the same as without dropout
            // ("inverted dropout"), so prediction needs no correction.
            double scale = 1.0 / (1.0 - rate);
            Random random = Context!.Random;

            for (int i = 0; i < dropoutMask.Length; i++)
            {
                dropoutMask[i] = random.NextDouble() < rate ? 0.0 : scale;
            }
        }

        // Post-processing of the computed outputs (skipped neurons are already zero):
        // zero the dropped ones and scale up the survivors.
        private void ApplyDropoutMask()
        {
            if (dropoutMask == null) return;

            for (int i = 0; i < output.Length; i++)
            {
                output[i] *= dropoutMask[i];
                neurons[i].Output = output[i];
            }
        }
        // ---------------------------------------------------------------- backward

        /// <inheritdoc/>
        public override Tensor Backward(Tensor gradOutput, bool preActivation = false,
                                        bool computeInputGradient = true)
        {
            // Copied ONLY when something will be overwritten. The only mutation of `grad` is
            // the dropout-mask multiplication below; without dropout the array is only read, so
            // a defensive copy would be an extra allocation per layer per sample — which adds up
            // over hundreds of thousands of training steps.
            double[] grad = dropoutMask != null
                ? (double[])gradOutput.Data.Clone()
                : gradOutput.Data;

            if (grad.Length != neurons.Count)
                throw new ArgumentException(
                    $"Dense has {neurons.Count} neurons, but the gradient from above has {grad.Length} values.");

            // 1) dropout — the SAME mask as in forward, and BEFORE the activation derivative,
            //    because the mask acts on the layer's output.
            if (dropoutMask != null)
            {
                for (int i = 0; i < grad.Length; i++) grad[i] *= dropoutMask[i];
            }

            // 2) ∂L/∂output → ∂L/∂z. With the softmax+CE fusion `grad` is already ∂L/∂z (s − t)
            //    and the activation is skipped, otherwise the softmax derivative would be applied twice.
            double[] delta = preActivation ? grad : ApplyActivationBackward(grad);

            // 3) gradients of the layer's own parameters
            for (int i = 0; i < neurons.Count; i++)
            {
                if (dropoutMask != null && dropoutMask[i] == 0.0)
                {
                    neurons[i].SkipSample();
                    continue;
                }

                neurons[i].CalculateGradients(delta[i]);
            }

            // 4) ∂L/∂input = Wᵀ · delta. Each layer computes it from its OWN weights, so the
            //    layer below never has to reach into another layer's weights.
            if (!computeInputGradient) return Tensor.Empty;

            double[] gradInput = new double[inputCount];

            for (int i = 0; i < neurons.Count; i++)
            {
                double d = delta[i];
                if (d == 0.0) continue;          // dropped by dropout → contributes nothing

                double[] w = neurons[i].Weights;
                for (int j = 0; j < inputCount; j++) gradInput[j] += d * w[j];
            }

            return new Tensor(gradInput);
        }

        // A layer-wide activation (softmax) needs the WHOLE gradient vector — element-wise
        // multiplication by Derivative() would use only the Jacobian's diagonal, which is wrong.
        private double[] ApplyActivationBackward(double[] gradFromAbove)
        {
            if (activation_func!.AppliesToLayer)
            {
                LayerActivationFunction? layerActivationFunc = activation_func as LayerActivationFunction;

                if (layerActivationFunc == null)
                {
                    throw new ArgumentException("Bad activation func");
                }

                return layerActivationFunc.BackwardForLayer(raw_output, gradFromAbove);
            }

            double[] e = new double[Neurons.Count];
            for (int j = 0; j < Neurons.Count; j++)
            {
                Neuron neuron = Neurons[j];
                e[j] = gradFromAbove[j] * neuron.Activation.Derivative(neuron.RawOutput);
            }
            return e;
        }
        /// <inheritdoc/>
        public override void UpdateParams()
        {
            foreach (Neuron neuron in neurons)
            {
                neuron.UpdateWeightsAndBias(Context?.L2 ?? 0);
            }
        }
    }
}
