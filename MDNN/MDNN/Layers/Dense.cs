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
        public override int[] InputShape => input_size;
        /// <inheritdoc/>
        public override int[] OutputShape => new int[] { output.Length };
        /// <inheritdoc/>
        public override ActivationFunction Activation
        {
            get { return activation_func; }
            set
            {
                activation_func = value;
                foreach (Neuron neuron in neurons)
                {
                    neuron.Activation = activation_func;
                }
            }
        }

        private int[] input_size;

        private double[] output;
        private double[] raw_output;
        private List<Neuron> neurons;
        private ActivationFunction activation_func;

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
                if (IsOutputLayer || activation_func.AppliesToLayer) return 0.0;
                return Context?.Dropout ?? 0.0;
            }
        }

        /// <summary>
        /// Creates a dense layer. The input size and the weights are set up later by
        /// <c>Build</c>, when the layer is attached to a model.
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

            // activation: default when not given; the correct input size + neurons are set by Build (when attached to a model)
            this.activation_func = activation ?? GeneralNeuralNetworkSettings.DefaultHiddenActivation;
            input_size = new int[] { 0 };

            output = new double[neuronCount];
            raw_output = new double[neuronCount];

            neurons = new List<Neuron>();

            for (int i = 0; i < neuronCount; i++)
            {
                neurons.Add(new Neuron(input_size[0], this.activation_func));   // placeholder (0 weights), replaced by Build
            }
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

            output = new double[saved.Count];
            raw_output = new double[saved.Count];

            neurons = new List<Neuron>();

            foreach (ExportNeuron neuron in saved)
            {
                neurons.Add(new Neuron(neuron, activation_func, optimizerPrototype));
            }

            input_size = new int[] { neurons[0].Weights.Length };

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

        /// <inheritdoc/>
        public override bool IsMaterialized =>
            input_size[0] > 0 && NeuronsMatchShape(output.Length, input_size[0]);

        /// <inheritdoc/>
        public override void MaterializeParameters()
        {
            if (input_size[0] <= 0)
            {
                return;   // input shape not known yet, nothing to build from
            }

            // Rebuild ONLY when the shape actually changes. Otherwise even a no-op call would
            // silently wipe trained weights. When the shape DOES change, a rebuild is unavoidable —
            // the weight vector length changes and the old values have nowhere to go.
            if (!NeuronsMatchShape(output.Length, input_size[0]))
            {
                // Built into a local list and assigned to `neurons` only when COMPLETE. Previously
                // `neurons` was overwritten first and the Context optimizer cloned afterwards — if
                // cloning threw, the layer kept new neurons with the constructor's GLOBAL optimizer,
                // later calls saw the shape as done, and whoever caught the exception silently
                // trained with a different optimizer.
                var rebuilt = new List<Neuron>();

                for (int i = 0; i < output.Length; i++)
                {
                    rebuilt.Add(new Neuron(input_size[0], activation_func, Context?.Random));
                }

                // Per-model optimizer: each neuron gets ITS OWN clone of the Context optimizer
                // (models are independent of each other). The neuron constructor takes a clone of
                // the GLOBAL optimizer, so without this a model with Adam would train with global SGD.
                //
                // Only for freshly built neurons. Previously this ran on EVERY materialization, and
                // the clone starts with empty state, so Adam's moments were thrown away. Adding a
                // layer was enough — MaterializeAll walks the whole chain, so unrelated layers lost
                // their accumulated state too.
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

        // Will this layer draw a dropout mask during training? Replicas use this to decide
        // whether they need their own random generator — see ReplicaPool.
        internal bool UsesDropout => EffectiveDropout > 0.0;

        // Replica for parallel training — see Layer.CreateReplica. Shared: weights (via
        // neuron replicas), activation, dropout settings. Own: outputs and the dropout mask,
        // which change with every sample.
        internal override Layer CreateReplica()
        {
            Dense replica = (Dense)MemberwiseClone();
            replica.output = new double[output.Length];
            replica.raw_output = new double[raw_output.Length];
            replica.dropoutMask = null;
            replica.neurons = neurons.Select(n => n.CreateReplica()).ToList();
            return replica;
        }

        // Do the already built neurons match the requested shape exactly?
        private bool NeuronsMatchShape(int neuronCount, int inputSize)
        {
            return neurons.Count == neuronCount
                   && neurons.Count > 0
                   && neurons[0].Weights.Length == inputSize;
        }
        /// <inheritdoc/>
        public override Tensor FeedForward(Tensor input)
        {

            double[] values = input.Data;

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

            if (activation_func.AppliesToLayer)
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
            if (activation_func.AppliesToLayer)
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

            int inputs = neurons[0].Weights.Length;
            double[] gradInput = new double[inputs];

            for (int i = 0; i < neurons.Count; i++)
            {
                double d = delta[i];
                if (d == 0.0) continue;          // dropped by dropout → contributes nothing

                double[] w = neurons[i].Weights;
                for (int j = 0; j < inputs; j++) gradInput[j] += d * w[j];
            }

            return new Tensor(gradInput);
        }

        // A layer-wide activation (softmax) needs the WHOLE gradient vector — element-wise
        // multiplication by Derivative() would use only the Jacobian's diagonal, which is wrong.
        private double[] ApplyActivationBackward(double[] gradFromAbove)
        {
            if (activation_func.AppliesToLayer)
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
