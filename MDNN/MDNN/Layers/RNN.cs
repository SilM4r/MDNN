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
        public override int[] InputShape => input_size;
        /// <inheritdoc/>
        public override int[] OutputShape => new int[] { output.Length };
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

        private int[] input_size;

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

            output = new double[saved.Count];
            raw_output = new double[saved.Count];

            neurons = new List<Neuron>();

            foreach (ExportNeuron neuron in saved)
            {
                neurons.Add(new Neuron(neuron, activation_func, optimizerPrototype));
            }

            input_size = new int[] { neurons[0].Weights.Length - 1 };

        }

        /// <summary>
        /// Creates a recurrent layer. The input size and the weights are set up later by
        /// <c>Build</c>, when the layer is attached to a model.
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

            // activation: default when not given; input size + neurons are set by Build (when attached to a model)
            this.activation_func = activation ?? GeneralNeuralNetworkSettings.DefaultHiddenActivation;
            CheckIsActivationFuncIsNotApplyToLayer();
            input_size = new int[] { 0 };

            output = new double[neuronCount];
            raw_output = new double[neuronCount];

            neurons = new List<Neuron>();

            for (int i = 0; i < neuronCount; i++)
            {
                neurons.Add(new Neuron(input_size[0] + 1, this.activation_func));   // placeholder, replaced by Build
            }

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
            output = new double[OutputShape[0]];
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
        // (neurons are rebuilt only in Fit → SetInputShape, after ResetSequence)
        private void EnsureSensitivities()
        {
            if (sW.Length != neurons.Count
                || (neurons.Count > 0 && sW[0].Length != neurons[0].Weights.Length))
                AllocateSensitivities();
        }

        /// <inheritdoc/>
        public override void WireShapes(int? outputSize = null, int[]? inputShape = null)
        {
            if (inputShape != null)
            {
                if (inputShape.Length == 1)
                {
                    input_size = inputShape;
                }

                else
                {
                    input_size = new int[] { 1 };
                    foreach (int input in inputShape)
                    {
                        input_size[0] *= input;
                    }
                }
            }

            if (outputSize != null)
            {
                output = new double[(int)outputSize];
                raw_output = new double[(int)outputSize];
            }

        }

        // Note: an RNN neuron has one extra weight (the recurrent input h(t-1)).
        /// <inheritdoc/>
        public override bool IsMaterialized =>
            input_size[0] > 0 && NeuronsMatchShape(output.Length, input_size[0] + 1);

        /// <inheritdoc/>
        public override void MaterializeParameters()
        {
            if (input_size[0] <= 0)
            {
                return;
            }

            if (!NeuronsMatchShape(output.Length, input_size[0] + 1))
            {
                // assigned to `neurons` only when complete — see Dense.MaterializeParameters
                var rebuilt = new List<Neuron>();

                for (int i = 0; i < output.Length; i++)
                {
                    rebuilt.Add(new Neuron(input_size[0] + 1, activation_func, Context?.Random));
                }

                // per-model optimizer, only for fresh neurons — see Dense.MaterializeParameters
                if (Context != null)
                {
                    foreach (Neuron n in rebuilt)
                    {
                        n.Optimizer = Context.Optimizer.CreateFresh();
                    }
                }

                neurons = rebuilt;
            }
        }

        // Replica for parallel training — see Layer.CreateReplica. Besides outputs it also owns
        // the hidden state and RTRL sensitivities: a replica processes WHOLE sequences, so the
        // state between steps stays in one replica and never mixes with another sequence.
        internal override Layer CreateReplica()
        {
            RNN replica = (RNN)MemberwiseClone();
            replica.output = new double[output.Length];
            replica.raw_output = new double[raw_output.Length];
            replica.neurons = neurons.Select(n => n.CreateReplica()).ToList();
            replica.AllocateSensitivities();
            return replica;
        }

        private bool NeuronsMatchShape(int neuronCount, int weightsPerNeuron)
        {
            return neurons.Count == neuronCount
                   && neurons.Count > 0
                   && neurons[0].Weights.Length == weightsPerNeuron;
        }

        /// <inheritdoc/>
        public override Tensor FeedForward(Tensor input)
        {

            double[] values = input.Data;

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
            int inputCount = neurons[0].Weights.Length - 1;   // the last weight is recurrent

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
                for (int k = 0; k < inputCount; k++) gradInput[k] += delta * w[k];
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
