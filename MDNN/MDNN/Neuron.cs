using MDNN.Serialization;
using MDNN.Optimizers;
using MDNN.Activations;

namespace MDNN
{

    /// <summary>
    /// A single neuron: weights, bias, activation and its own optimizer state.
    /// Used by <c>Dense</c> and <c>RNN</c> layers.
    /// </summary>
    public class Neuron
    {
        /// <summary>The neuron's weights, one per input.</summary>
        public double[] Weights
        {
            get { return weights; }
        }
        /// <summary>The neuron's bias.</summary>
        public double Bias
        {
            get { return bias; }
        }


        // The neuron's state can be READ from outside (inspecting weights, outputs… to learn
        // how the network computes), but only the library writes to it. Previously these were
        // public fields, and setting e.g. `inputs` or `optimizer` out of the order the layer
        // expects silently gave a wrong computation.
        //
        // Output and inputs are from the LAST pass through the neuron.
        /// <summary>Output of the last forward pass (after activation).</summary>
        public double Output { get; internal set; }
        /// <summary>Output of the last forward pass before activation (w · x + b).</summary>
        public double RawOutput { get; internal set; }   // before activation (w · x + b)
        /// <summary>Inputs of the last forward pass.</summary>
        public double[] Inputs { get; internal set; }

        /// <summary>The neuron's activation function.</summary>
        public ActivationFunction Activation { get; internal set; }
        /// <summary>The optimizer that updates this neuron's weights and bias (holds its own state).</summary>
        public Optimizer Optimizer { get; internal set; }

        private double[] weights;
        private double bias;

        internal double[] gradientsW;
        // internal for the parity tests against PyTorch — without reading the bias gradient
        // only the weight gradients could be compared. The public API stays unchanged.
        internal double gradientsB;

        private int mini_batch_size = 0;

        // `random` = randomness source for weight initialization. The layer passes it from its
        // NetworkContext (for seed reproducibility); null = a standalone neuron without a model,
        // falling back to the shared global generator.
        /// <summary>
        /// Creates a neuron with randomly initialized weights (uniform in ±√(6 / inputs)) and zero bias.
        /// </summary>
        /// <param name="numberOfInputs">Number of inputs (= number of weights).</param>
        /// <param name="activation">Activation function.</param>
        /// <param name="random">Random source for initialization; <c>null</c> = the global generator.</param>
        public Neuron(int numberOfInputs, ActivationFunction activation, Random? random = null)
        {
            Random rnd = random ?? GeneralNeuralNetworkSettings.Random;

            weights = new double[numberOfInputs];
            Inputs = new double[numberOfInputs];
            gradientsW = new double[numberOfInputs];

            Output = 0;

            // the limit is the same for all weights, so it is computed once outside the loop
            double limit = numberOfInputs > 0 ? Math.Sqrt(6.0 / numberOfInputs) : 0;

            for (int i = 0; i < numberOfInputs; i++)
            {
                Weights[i] = (rnd.NextDouble() * 2 - 1) * limit;
            }

            bias = 0;

            ResetGradients();
            Optimizer = GeneralNeuralNetworkSettings.DefaultOptimizer.CreateFresh();
            Activation = activation;
        }

        // `optimizerPrototype` = prototype whose fresh clone the neuron gets (passed by the layer
        // from the context of the model being loaded). null = standalone neuron without a model →
        // the global default optimizer, as in the constructor above.
        internal Neuron(ExportNeuron exportNeuron, ActivationFunction activation, Optimizer? optimizerPrototype = null)
        {

            weights = exportNeuron.Weights;
            Inputs = new double[Weights.Length];
            gradientsW = new double[Weights.Length];

            bias = exportNeuron.Bias;

            Output = 0;

            ResetGradients();
            Optimizer = (optimizerPrototype ?? GeneralNeuralNetworkSettings.DefaultOptimizer).CreateFresh();
            Activation = activation;

            // Resuming training: the optimizer continues with the state the model was saved with.
            if (exportNeuron.OptimizerState != null)
            {
                Optimizer.ImportState(exportNeuron.OptimizerState);
            }
        }


        // Sets specific weights and bias from outside. Needed for the parity tests against
        // PyTorch: both sides must start from BIT-identical parameters, otherwise the tests would
        // compare random initialization, not the computation. `Weights` is public but read-only
        // (it returns the array, so it could be mutated element by element) and `Bias` is a
        // scalar without a setter — this handles both in one place.
        internal void SetParamsForTests(double[] newWeights, double newBias)
        {
            if (newWeights.Length != weights.Length)
                throw new ArgumentException(
                    $"Expected {weights.Length} weights, got {newWeights.Length}.");

            Array.Copy(newWeights, weights, weights.Length);
            bias = newBias;
        }

        /// <summary>
        /// Computes the neuron's output for the given inputs: activation(w · x + b).
        /// Also stores <see cref="Inputs"/>, <see cref="RawOutput"/> and <see cref="Output"/>.
        /// </summary>
        public double FeedForward(double[] values)
        {
            Inputs = values;

            // Dot product w · x. Two things, both for speed:
            //
            // 1) It sums into LOCAL variables, and RawOutput is assigned only once at the end.
            //    Accumulating into a property/field of the object would mean a memory write on
            //    every iteration; a local sum stays in a register (measured: 3.75× faster).
            //
            // 2) Four independent sums instead of one. With a single sum each addition waits for
            //    the previous one (floating-point addition has a latency of several cycles), so the
            //    CPU stalls. Four chains run in parallel (measured: 2.45× faster; SIMD adds nothing
            //    on top). Mathematically it is the same sum in a different order — so the result may
            //    differ in the last bit (~1e-16) from left-to-right summation.
            double s0 = 0, s1 = 0, s2 = 0, s3 = 0;
            int i = 0;
            for (; i <= values.Length - 4; i += 4)
            {
                s0 += values[i] * weights[i];
                s1 += values[i + 1] * weights[i + 1];
                s2 += values[i + 2] * weights[i + 2];
                s3 += values[i + 3] * weights[i + 3];
            }

            double sum = (s0 + s1) + (s2 + s3);
            for (; i < values.Length; i++)   // remainder when the input count is not divisible by 4
            {
                sum += values[i] * weights[i];
            }

            RawOutput = sum + bias;
            Output = Activation.Apply(RawOutput);

            return Output;
        }

        // ---------------------------------------------------------------- replicas
        //
        // Replica for data-parallel training (Train.Workers > 1). It SHARES the weight array with
        // the original — during a batch the weights are only read; they are changed by the
        // original's UpdateParams after the batch, when the replicas are idle. So weights need not
        // be copied between batches. The replica owns everything that changes during a pass:
        // gradients, the batch counter, inputs and outputs. The bias is a scalar and cannot be
        // shared — SyncBiasFrom copies it before each batch.
        internal Neuron CreateReplica()
        {
            Neuron replica = (Neuron)MemberwiseClone();
            replica.Inputs = new double[weights.Length];
            replica.Output = 0;
            replica.RawOutput = 0;
            replica.gradientsW = new double[weights.Length];
            replica.gradientsB = 0;
            replica.mini_batch_size = 0;
            return replica;
        }

        internal void SyncBiasFrom(Neuron original)
        {
            bias = original.bias;
        }

        // Adds the replica's gradients and counter to this neuron and resets the replica. The
        // counter MUST be added — UpdateWeightsAndBias divides by it, so without it the whole
        // batch's gradient would be divided only by the samples that went through the original (zero).
        internal void TakeGradientsFrom(Neuron replica)
        {
            for (int i = 0; i < gradientsW.Length; i++)
            {
                gradientsW[i] += replica.gradientsW[i];
                replica.gradientsW[i] = 0;
            }
            gradientsB += replica.gradientsB;
            replica.gradientsB = 0;

            mini_batch_size += replica.mini_batch_size;
            replica.mini_batch_size = 0;
        }

        // The neuron was dropped by dropout in this sample: it was not computed in forward and
        // its gradient is zero, so there is nothing to accumulate. The batch counter MUST still be
        // incremented — gradients are divided by it, and skipping dropped samples would divide by
        // a smaller number and inflate the gradient.
        internal void SkipSample()
        {
            mini_batch_size++;
        }

        /// <summary>
        /// Accumulates the weight and bias gradients for one sample, given
        /// <paramref name="e"/> = ∂L/∂z (the gradient with respect to the pre-activation output).
        /// </summary>
        public void CalculateGradients(double e)
        {

            for (int i = 0; i < Weights.Length; i++)
            {
                gradientsW[i] += e * Inputs[i];
            }
            gradientsB += e ;

            mini_batch_size++;
        }

        // RTRL: parameter gradient = immediate gradient from above (gImm) × that parameter's sensitivity.
        // sensW[i] = ∂h/∂Weights[i], sensB = ∂h/∂bias (both kept and advanced by the RNN layer).
        /// <summary>
        /// RTRL gradient accumulation for one time step: adds <paramref name="gImm"/> × sensitivity
        /// to each parameter's gradient.
        /// </summary>
        /// <param name="gImm">Immediate gradient ∂L/∂h from above.</param>
        /// <param name="sensW">Sensitivities ∂h/∂Weights[i].</param>
        /// <param name="sensB">Sensitivity ∂h/∂bias.</param>
        public void AccumulateGradient(double gImm, double[] sensW, double sensB)
        {
            for (int i = 0; i < Weights.Length; i++)
            {
                gradientsW[i] += gImm * sensW[i];
            }
            gradientsB += gImm * sensB;

            mini_batch_size++;
        }

        // weightDecay = L2 coefficient. The layer takes it from NetworkContext.L2; 0 = off.
        /// <summary>
        /// Applies the accumulated (batch-averaged) gradients through the optimizer and resets them.
        /// Does nothing if no gradients were accumulated.
        /// </summary>
        /// <param name="weightDecay">L2 regularization coefficient (not applied to the bias); 0 = off.</param>
        public void UpdateWeightsAndBias(double weightDecay = 0)
        {
            // An empty batch is a legitimate state (UpdateParams without a preceding backprop).
            // Without this it divided by zero → 0/0 = NaN → silently the whole model became NaN.
            if (mini_batch_size == 0)
            {
                return;
            }

            // weights + bias (the bias has index Weights.Length, see below)
            Optimizer.EnsureCapacity(Weights.Length + 1);

            for (int i = 0; i < Weights.Length; i++)
            {
                // L2 is added to the ALREADY AVERAGED gradient, not to the batch sum —
                // otherwise the regularization strength would depend on the mini-batch size.
                double gradient = gradientsW[i] / mini_batch_size + weightDecay * Weights[i];
                Weights[i] = Optimizer.Update(Weights[i], gradient, i);
            }

            // The bias is NOT regularized: it does not shape the decision boundary, only shifts it,
            // so pulling it towards zero does not simplify the model — it only makes it worse.
            bias = Optimizer.Update(bias, gradientsB / mini_batch_size,Weights.Length);

            mini_batch_size = 0;

            ResetGradients();
        }

        // `randomSource`: same logic as in the constructor — if the caller does not supply one,
        // the global generator is used. (The neuron itself cannot see the Context, so whoever
        // runs the mutation must pass it in.)
        /// <summary>
        /// Randomly perturbs the weights (e.g. for evolutionary training): each weight changes with
        /// probability <paramref name="chanceOfMutation"/> % by a random relative amount
        /// within ±<paramref name="percentMutation"/>/2 %.
        /// </summary>
        /// <param name="chanceOfMutation">Chance (0–100) that a given weight is mutated.</param>
        /// <param name="percentMutation">Width of the relative change range, in percent.</param>
        /// <param name="randomSource">Random source; <c>null</c> = the global generator.</param>
        public void MutateParams(int chanceOfMutation, int percentMutation, Random? randomSource = null)
        {
            Random random = randomSource ?? GeneralNeuralNetworkSettings.Random;

            for (int i = 0; i < Weights.Length; i++)
            {
                int randomValueBetween0And100 = random.Next(101);
                if (randomValueBetween0And100 < chanceOfMutation)
                {
                    // The range is -percentage/2 to +percentage/2
                    double minPercentage = -percentMutation / 100.0 / 2.0;
                    double maxPercentage = percentMutation / 100.0 / 2.0;

                    // Random value between minPercentage and maxPercentage
                    double randomPercentage = minPercentage + (random.NextDouble() * (maxPercentage - minPercentage));

                    // Change the weight by that random percentage
                    Weights[i] += Weights[i] * randomPercentage;
                }
            }
        }

        private void ResetGradients()
        {
            for (int i = 0; i < gradientsW.Length; i++)
            {
                gradientsW[i] = 0;
            }
            gradientsB = 0;
        }
    }
}