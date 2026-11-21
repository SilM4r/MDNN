using MDNN.Activations;
using MDNN.Layers;
using MDNN.Serialization;
using MDNN.Optimizers;

namespace MDNN.Layers
{
    /// <summary>
    /// Simple recurrent layer: each neuron gets the input plus its own previous output h(t-1).
    /// Trained with real-time recurrent learning (RTRL).
    /// </summary>
    public class RNN : NeuronLayer
    {
        /// <inheritdoc/>
        public override List<Neuron> Neurons => neurons;
        /// <inheritdoc/>
        public override string Name => "RNN";

        // A snapshot, not a window into the working array — see Tensor.FromVector.
        /// <inheritdoc/>
        public override Tensor Output => Tensor.FromVector(output);
        /// <inheritdoc/>
        public override Tensor RawOutput => Tensor.FromVector(raw_output);
        /// <inheritdoc/>
        public override ActivationFunction Activation
        {
            get { return activation_func; }
            set
            {
                activation_func = value;
                CheckIsActivationFuncIsNotApplyToLayer();
            }
        }

        // Number of inputs WITHOUT the recurrent one (each neuron has inputCount + 1 weights);
        // 0 until the layer is built.
        private int inputCount;

        private double[] output;
        private double[] raw_output;
        private List<Neuron> neurons;
        private ActivationFunction activation_func;

        // RTRL: sensitivities ∂h_i/∂param, accumulated FORWARD in time.
        // Allocated lazily (EnsureSensitivities) once the weight count is known; empty array = not yet.
        private double[][] sW = Array.Empty<double[]>();   // sW[neuron][weight]  (last weight = recurrent)
        private double[] sB = Array.Empty<double>();       // sB[neuron]          (bias)


        // `optimizerPrototype` — see Dense(ExportDenseLayer, …)
        internal RNN(ExportRnnLayer layer, Optimizer? optimizerPrototype = null)
        {
            activation_func = ActivationFunction.FromName(layer.Name_of_activation_function);
            CheckIsActivationFuncIsNotApplyToLayer();
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

            inputCount = neurons[0].Weights.Length - 1;
            MarkBuiltFromFile(new int[] { inputCount });
        }

        /// <summary>
        /// Creates a recurrent layer. This is only a description — the weights are created when
        /// the layer is built for its input (inside a model: on the first pass).
        /// </summary>
        /// <param name="neuronCount">Number of neurons (= size of the hidden state and of the output).</param>
        /// <param name="activation">Activation function; <c>null</c> = the default hidden activation.
        /// Layer-wide activations such as softmax are not supported.</param>
        public RNN(int neuronCount, ActivationFunction? activation = null)
        {
            if (neuronCount <= 0)
            {
                throw new ArgumentException("The number of neurons in a layer must be greater than 0");
            }

            this.activation_func = activation ?? GeneralNeuralNetworkSettings.DefaultHiddenActivation;
            CheckIsActivationFuncIsNotApplyToLayer();

            // no placeholder neurons — see the Dense constructor
            NeuronCount = neuronCount;
            neurons = new List<Neuron>();

            output = new double[neuronCount];
            raw_output = new double[neuronCount];
        }

        // A layer-wide activation (softmax) on RNN used to be broken: ApplyToLayer was applied to
        // `output`, i.e. to values that had ALREADY gone through the per-neuron activation (double
        // activation), while raw_output and the RTRL sensitivities stayed on the pre-softmax path.
        // Until it gets a proper design, an explicit error beats silent nonsense.
        // Same policy as Conv (CheckIsActivationFuncIsNotApplyToLayer).
        private void CheckIsActivationFuncIsNotApplyToLayer()
        {
            if (activation_func.AppliesToLayer)
            {
                throw new ArgumentException(
                    "An RNN layer cannot use a layer-wide activation (e.g. Softmax) yet. " +
                    "Use e.g. Tanh() or ReLU(), and put the softmax on a following Dense layer.");
            }
        }

        /// <summary>
        /// Starts a new sequence: clears the hidden state and the RTRL sensitivities.
        /// </summary>
        public void ResetSequence()
        {
            // new sequence → zero the hidden state and the RTRL sensitivities (reallocation = zeroing)
            output = new double[NeuronCount];
            AllocateSensitivities();
        }

        // allocates (and thereby zeroes) the sensitivities for the current neurons
        private void AllocateSensitivities()
        {
            sW = new double[neurons.Count][];
            for (int i = 0; i < neurons.Count; i++)
                sW[i] = new double[neurons[i].Weights.Length];
            sB = new double[neurons.Count];
        }

        // safeguard: the sensitivities must match both the neuron count AND the weight count
        // (neurons are rebuilt only when the layer is built, after ResetSequence)
        private void EnsureSensitivities()
        {
            if (sW.Length != neurons.Count
                || (neurons.Count > 0 && sW[0].Length != neurons[0].Weights.Length))
                AllocateSensitivities();
        }

        // ---------------------------------------------------------------- building

        /// <summary>RNN flattens any input to a vector, like Dense.</summary>
        protected override int[] NormalizeInputShape(int[] inputShape) => new int[] { Dense.Flatten(inputShape) };

        /// <inheritdoc/>
        protected override int[] ComputeOutputShape(int[] inputShape) => new int[] { NeuronCount };

        // Note: an RNN neuron has one extra weight (the recurrent input h(t-1)).
        /// <inheritdoc/>
        protected override void BuildCore(int[] inputShape, Random random, Optimizer optimizerPrototype)
        {
            // assigned to `neurons` only when complete, same draw order — see Dense.BuildCore
            var built = new List<Neuron>(NeuronCount);

            for (int i = 0; i < NeuronCount; i++)
            {
                built.Add(new Neuron(inputShape[0] + 1, activation_func, random));
            }

            foreach (Neuron n in built)
            {
                n.Optimizer = optimizerPrototype.CreateFresh();
            }

            neurons = built;
            inputCount = inputShape[0];
        }

        /// <inheritdoc/>
        protected override void OnInvalidated()
        {
            neurons = new List<Neuron>();
            inputCount = 0;

            // Including the hidden state: `output` is h(t-1) of the next step, and after a rebuild
            // it would still hold the last output of the DISCARDED weights.
            ClearSequenceState();
        }

        internal override bool CanImport(BaseExportLayer saved)
        {
            return IsBuilt
                   && saved is ExportRnnLayer rnn
                   && rnn.Name_of_activation_function == activation_func.Name
                   && rnn.Neurons != null
                   && rnn.Neurons.Count == NeuronCount
                   && rnn.Neurons.All(n => n.Weights.Length == inputCount + 1);
        }

        internal override Action PrepareImport(BaseExportLayer saved, Optimizer optimizerPrototype)
        {
            var rnn = (ExportRnnLayer)saved;
            List<ExportNeuron> savedNeurons = rnn.Neurons!;

            // what can fail — nothing is written yet (see Layer.PrepareImport)
            Optimizer[] optimizers = savedNeurons.Select(n => Neuron.PrepareOptimizer(n, optimizerPrototype)).ToArray();

            return () =>
            {
                for (int i = 0; i < neurons.Count; i++)
                {
                    neurons[i].ImportFrom(savedNeurons[i], optimizers[i]);
                }

                ClearSequenceState();
            };
        }

        // The same state as a layer freshly loaded from the file or freshly built: no hidden state
        // h(t-1), no RTRL sensitivities.
        private void ClearSequenceState()
        {
            output = new double[NeuronCount];
            Array.Clear(raw_output);
            sW = Array.Empty<double[]>();
            sB = Array.Empty<double>();
        }

        // Replica for parallel training — see Layer.CreateReplica. Besides outputs it also owns
        // the hidden state and RTRL sensitivities: a replica processes WHOLE sequences, so the
        // state between steps stays in one replica and never mixes with another sequence.
        internal override Layer CreateReplica()
        {
            ThrowIfNotBuiltForReplica();

            RNN replica = (RNN)MemberwiseClone();
            replica.output = new double[output.Length];
            replica.raw_output = new double[raw_output.Length];
            replica.neurons = neurons.Select(n => n.CreateReplica()).ToList();
            replica.AllocateSensitivities();
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

            // same check as in Dense.FeedForward
            if (values.Length != inputCount)
            {
                throw new ArgumentException(
                    $"{Name} is built for {inputCount} inputs, but got {values.Length} " +
                    $"(shape {SpatialShape.Describe(input.Shape)}). {ShapeChangeHint}");
            }

            double[] newValues = new double[values.Length + 1];


            for (int i = 0; i < values.Length; i++)
            {
                newValues[i] = values[i];
            }

            EnsureSensitivities();

            for (int i = 0; i < neurons.Count; i++)
            {
                newValues[values.Length] = output[i];   // recurrent input = h_i(t-1) = previous output

                output[i] = neurons[i].FeedForward(newValues);
                raw_output[i] = neurons[i].RawOutput;   // z_i(t)

                // RTRL: advance the sensitivities by one step
                double actD = neurons[i].Activation.Derivative(raw_output[i]);   // act'(z_i)
                double r = neurons[i].Weights[newValues.Length - 1];                  // recurrent weight
                for (int k = 0; k < newValues.Length; k++)                            // ∂z/∂w_k = newValues[k]
                    sW[i][k] = actD * (newValues[k] + r * sW[i][k]);
                sB[i] = actD * (1.0 + r * sB[i]);                                     // ∂z/∂bias = 1
            }

            return Tensor.FromVector(output);
        }

        // ---------------------------------------------------------------- backward

        /// <inheritdoc/>
        public override Tensor Backward(Tensor gradOutput, bool preActivation = false,
                                        bool computeInputGradient = true)
        {
            if (preActivation)
                throw new NotSupportedException(
                    "RNN cannot be an output layer with softmax (guarded in Gradient), " +
                    "so the softmax+CE fusion does not apply to it.");

            EnsureSensitivities();

            double[] gradOut = gradOutput.Data;

            if (gradOut.Length != neurons.Count)
                throw new ArgumentException(
                    $"The gradient from above has {gradOut.Length} elements, the layer {neurons.Count} neurons.");

            // `gradOutput` is exactly the RTRL immediate gradient g_imm = ∂L/∂h_j, WITHOUT the
            // activation derivative, so it is used directly. (Previously it had to be stored in
            // the field `gImm` because two separate methods computed the backward pass.)
            //
            // Note: the recurrent weight does not feed into the input gradient — RTRL does not
            // capture the contribution back in time. A known limitation, not an oversight (see README).

            // When the caller does not want the input gradient, it is not even allocated — and,
            // more importantly, an EMPTY tensor is returned, not an array of zeros. The contract in
            // Layer.cs says so explicitly: a mistake should fail on the length, not as a silently
            // wrong result. Previously RNN was the only one of the four layers that deviated.
            double[] gradInput = computeInputGradient ? new double[inputCount] : Array.Empty<double>();

            for (int j = 0; j < neurons.Count; j++)
            {
                double gImmJ = gradOut[j];

                // RTRL: parameter gradient = g_imm(t) · sensitivity(t), summed over time.
                // The sW/sB sensitivities ALREADY include the activation derivative (they are
                // advanced in forward), so act' is NOT applied here — unlike Dense.
                neurons[j].AccumulateGradient(gImmJ, sW[j], sB[j]);

                // For the layer below, act' does belong: ∂L/∂z_j = g_imm_j · act'(z_j).
                if (!computeInputGradient) continue;

                double delta = gImmJ * neurons[j].Activation.Derivative(neurons[j].RawOutput);
                if (delta == 0.0) continue;

                double[] w = neurons[j].Weights;
                for (int k = 0; k < inputCount; k++) gradInput[k] += delta * w[k];   // the last weight is recurrent
            }

            return computeInputGradient ? new Tensor(gradInput) : Tensor.Empty;
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
