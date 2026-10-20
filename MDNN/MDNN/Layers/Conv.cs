using MDNN.Layers;
using MDNN.Optimizers;
using MDNN.Serialization;
using MDNN.Activations;

namespace MDNN.Layers
{
    /// <summary>
    /// 2D convolution layer (stride 1) with "valid" or "same" padding; input and output are <c>[height, width, channels]</c>.
    /// </summary>
    public class Conv : Layer
    {
        /// <inheritdoc/>
        public override string Name => "Conv";

        /// <inheritdoc/>
        public override int[] InputShape => inputsShape;

        /// <inheritdoc/>
        public override int[] OutputShape => outputShape;

        /// <summary>Snapshot of the last output (after activation) with shape <c>[h,w,filters]</c>.</summary>
        // Same shape as FeedForward returns. Previously this was `new Tensor(output)` over the
        // jagged array: `double[][][]` has Rank 1, so Shape came out as [outH] instead of
        // [h,w,channels]. The data were right, the Shape was not — visible only when Conv is the
        // output layer and someone reads Shape. It was also a view into the working array, not a snapshot.
        public override Tensor Output => Tensor.FromJagged3D(output, outputShape);

        /// <summary>Snapshot of the last output before activation, shape <c>[h,w,filters]</c>.</summary>
        public override Tensor RawOutput => Tensor.FromJagged3D(raw_output, outputShape);

        /// <summary>Element-wise activation applied to each output value; layer-wide activations (e.g. Softmax) are rejected.</summary>
        public override ActivationFunction Activation
        {
            get { return activation_func; }
            set 
            {
                activation_func = value;
                CheckIsActivationFuncIsNotApplyToLayer();
            }
        }

        /// <summary>Kernel weights indexed <c>[filter][kernelRow][kernelCol][inputChannel]</c>.</summary>
        public double[][][][] Kernels  // [numFilters][kernelRows][kernelCols][inputChannels]
        {
            get { return kernels; }
        }
        /// <summary>One bias per filter.</summary>
        public double[] Biases
        {
            get { return biases; }
        }
        /// <summary>Padding mode, lower-case: <c>"valid"</c> or <c>"same"</c>.</summary>
        public string Padding
        {
            get { return padding; }
        }

        // Empty defaults: buffers are allocated during wiring (WireShapes), kernels during
        // materialization. All constructors overwrite them; the empty array is here only so
        // the fields are never null.
        private double[][][][] kernels = Array.Empty<double[][][]>();
        internal double[][][][] dKernels = Array.Empty<double[][][]>();   // internal for gradient-check tests
        private double[] biases = Array.Empty<double>();
        internal double[] dBiases = Array.Empty<double>();                // internal for gradient-check tests
        private double[][][] dOutput = Array.Empty<double[][]>();
        private string padding;
        private double[][][] output = Array.Empty<double[][]>();
        private double[][][] raw_output = Array.Empty<double[][]>();
        private double[][][] inputs = Array.Empty<double[][]>();

        private int[] inputsShape;
        private int[] outputShape;

        private int mini_batch_size;
        private Optimizer optimizer;

        // For saving the optimizer state to a file (ExportCNNLayer).
        internal Optimizer OptimizerForSave => optimizer;

        internal override void ReplaceOptimizer(Optimizer prototype)
        {
            optimizer = prototype.CreateFresh();
        }

        // Number of filters requested by the last WireShapes; MaterializeParameters builds accordingly.
        private int? requestedKernelCount;

        private ActivationFunction activation_func;

        // `optimizerPrototype` — see Dense(ExportDenseLayer, …)
        internal Conv(ExportCNNLayer layer, Optimizer? optimizerPrototype = null)
        {
            kernels = BaseExportLayer.Required(layer.Kernel, Name, nameof(layer.Kernel));
            biases = BaseExportLayer.Required(layer.Biases, Name, nameof(layer.Biases));
            activation_func = ActivationFunction.FromName(layer.Name_of_activation_function);
            CheckIsActivationFuncIsNotApplyToLayer();
            padding = NormalizePadding(BaseExportLayer.Required(layer.Padding, Name, nameof(layer.Padding)));

            optimizer = (optimizerPrototype ?? GeneralNeuralNetworkSettings.DefaultOptimizer).CreateFresh();

            // resuming training — see Neuron(ExportNeuron, …)
            if (layer.OptimizerState != null)
            {
                optimizer.ImportState(layer.OptimizerState);
            }

            inputsShape = BaseExportLayer.Required(layer.InputShape, Name, nameof(layer.InputShape));

            if (padding == "same")
            {
                outputShape = new int[] { inputsShape[0], inputsShape[1], kernels.Length };
            }
            else
            {
                outputShape = new int[] { inputsShape[0] - kernels[0].Length + 1, inputsShape[1] - kernels[0][0].Length + 1, kernels.Length };
            }

            inputs = new double[inputsShape[0]][][];
            for (int i = 0; i < inputsShape[0]; i++)
            {
                inputs[i] = new double[inputsShape[1]][];
                for (int j = 0; j < inputsShape[1]; j++)
                {
                    inputs[i][j] = new double[inputsShape[2]];
                }
            }

            output = new double[outputShape[0]][][];
            dOutput = new double[outputShape[0]][][];
            raw_output = new double[outputShape[0]][][];
            for (int i = 0; i < outputShape[0]; i++)
            {
                output[i] = new double[outputShape[1]][];
                raw_output[i] = new double[outputShape[1]][];
                dOutput[i] = new double[outputShape[1]][];
                for (int j = 0; j < outputShape[1]; j++)
                {
                    output[i][j] = new double[outputShape[2]];
                    dOutput[i][j] = new double[outputShape[2]];
                    raw_output[i][j] = new double[outputShape[2]];
                }
            }
            dBiases = new double[kernels.Count()];
            dKernels = new double[kernels.Count()][][][];

            for (int i = 0; i < kernels.Count(); i++)
            {
                dKernels[i] = new double[kernels[0].Count()][][];
                for (int j = 0; j < kernels[0].Count(); j++)
                {
                    dKernels[i][j] = new double[kernels[0][0].Count()][];
                    for (int k = 0; k < kernels[0][0].Count(); k++)
                    {
                        dKernels[i][j][k] = new double[kernels[0][0][0].Count()];
                    }
                }
            }

            inicializationK_B_dK_dB();

            mini_batch_size = 0;
        }

        /// <summary>Creates a convolution layer; the input shape is wired when the layer is added to a model.</summary>
        /// <param name="filters">Number of filters (output channels); must be greater than 0.</param>
        /// <param name="kernelSize">Size of the square kernel; must be greater than 0.</param>
        /// <param name="activation">Element-wise activation; defaults to <c>GeneralNeuralNetworkSettings.DefaultHiddenActivation</c>.</param>
        /// <param name="padding"><c>"valid"</c> (no padding) or <c>"same"</c> (output has the input's height and width); case-insensitive.</param>
        public Conv(int filters,int kernelSize, ActivationFunction? activation = null, string padding = "valid")
        {
            if (filters <= 0)
                throw new ArgumentOutOfRangeException(nameof(filters), filters,
                    "The number of kernels must be greater than 0.");
            if (kernelSize <= 0)
                throw new ArgumentOutOfRangeException(nameof(kernelSize), kernelSize,
                    "The kernel size must be greater than 0.");

            this.activation_func = activation ?? GeneralNeuralNetworkSettings.DefaultHiddenActivation;
            CheckIsActivationFuncIsNotApplyToLayer();

            inputsShape = new int[] { 0 };
            outputShape = new int[] { 0 };
            this.padding = NormalizePadding(padding);

            // input channels = 0 (placeholder); Build fills in the real channels + kernels
            inicializationK_B_dK_dB(filters, new int[] { kernelSize, kernelSize, 0 }, true);

            optimizer = GeneralNeuralNetworkSettings.DefaultOptimizer.CreateFresh();
            mini_batch_size = 0;

        }

        /// <inheritdoc/>
        /// <remarks>For Conv, <paramref name="outputSize"/> sets the number of filters.</remarks>
        public override void WireShapes(int? outputSize = null, int[]? inputShape = null)
        {
            // The input shape is not known yet (Insert/Add before SetInputShape passes the
            // placeholder {0}). Previously it was pushed into ConvertTo2D → rows = ceil(sqrt(0)) = 0
            // → DivideByZeroException deep inside the library. The layer stays a placeholder;
            // SetInputShape completes it once the shape is known.
            if (outputSize != null)
            {
                requestedKernelCount = outputSize;
            }

            if (inputShape != null && inputShape.Any(d => d <= 0))
            {
                inputsShape = new int[] { 0 };
                outputShape = new int[] { 0 };
                return;
            }

            if (inputShape != null)
            {
                // The same normalizer FeedForward uses. Previously there were two different
                // implementations: this one handled rank 1 and 3, the forward also rank 2 — so a
                // model with 2D input could not be wired even though the pass would handle it.
                inputsShape = SpatialShape.Normalize(inputShape, Name);

                // Without padding the kernel must fit into the input — otherwise the output would
                // have a negative or zero dimension and it would only crash on buffer allocation.
                if (padding == "valid"
                    && (inputsShape[0] < kernels[0].Length || inputsShape[1] < kernels[0][0].Length))
                {
                    throw new ArgumentException(
                        $"{Name}: a {kernels[0].Length}×{kernels[0][0].Length} kernel does not fit the input " +
                        $"{SpatialShape.Describe(inputsShape)} with \"valid\" padding. Use a smaller kernel, " +
                        "or padding \"same\".");
                }

                if (padding == "same")
                {
                    outputShape = new int[] { inputsShape[0], inputsShape[1], kernels.Length };
                }
                else
                {
                    outputShape = new int[] { inputsShape[0] - kernels[0].Length + 1, inputsShape[1] - kernels[0][0].Length + 1, kernels.Length };
                }

                inputs = new double[inputsShape[0]][][];
                for (int i = 0; i < inputsShape[0]; i++)
                {
                    inputs[i] = new double[inputsShape[1]][];
                    for (int j = 0; j < inputsShape[1]; j++)
                    {
                        inputs[i][j] = new double[inputsShape[2]];
                    }
                }

                output = new double[outputShape[0]][][];
                dOutput = new double[outputShape[0]][][];
                raw_output = new double[outputShape[0]][][];
                for (int i = 0; i < outputShape[0]; i++)
                {
                    output[i] = new double[outputShape[1]][];
                    raw_output[i] = new double[outputShape[1]][];
                    dOutput[i] = new double[outputShape[1]][];
                    for (int j = 0; j < outputShape[1]; j++)
                    {
                        output[i][j] = new double[outputShape[2]];
                        dOutput[i][j] = new double[outputShape[2]];
                        raw_output[i][j] = new double[outputShape[2]];
                    }
                }
            }


        }

        /// <inheritdoc/>
        public override bool IsMaterialized =>
            inputsShape.Length == 3
            && inputsShape[2] > 0
            && kernels.Length == (requestedKernelCount ?? kernels.Length)
            && kernels[0][0][0].Length == inputsShape[2];

        /// <inheritdoc/>
        public override void MaterializeParameters()
        {
            if (inputsShape.Length != 3 || inputsShape[2] <= 0)
            {
                return;   // input shape not known yet
            }

            int filters = requestedKernelCount ?? kernels.Length;

            // Re-initialize kernels ONLY when their shape changes (number of filters or
            // number of input channels). Otherwise even a no-op call would overwrite trained
            // kernels with random numbers. ChangeKernel: false only zeroes
            // dKernels/dBiases and leaves kernels+biases alone.
            bool kernelShapeMatches = kernels.Length == filters
                                      && kernels[0][0][0].Length == inputsShape[2];

            // per-model optimizer only when the kernels were actually rebuilt — see
            // Dense.MaterializeParameters. Cloning on every materialization discarded Adam's
            // moments even for layers the rewiring did not affect.
            //
            // The clone is made BEFORE rebuilding the kernels: if cloning throws, the layer stays
            // as it was, rather than with new kernels and the old optimizer.
            Optimizer? fresh = Context != null && !kernelShapeMatches
                ? Context.Optimizer.CreateFresh()
                : null;

            inicializationK_B_dK_dB(
                filters,
                new int[] { kernels[0].Count(), kernels[0][0].Count(), inputsShape[2] },
                !kernelShapeMatches);

            if (fresh != null)
            {
                optimizer = fresh;
            }
        }

        // Replica for parallel training — see Layer.CreateReplica. Kernels and biases are SHARED
        // by reference (the original's optimizer updates them in place, after the batch), so
        // nothing is copied between batches. Owned: input and output buffers (allocated by
        // WireShapes just as during wiring) and gradient accumulators.
        internal override Layer CreateReplica()
        {
            Conv replica = (Conv)MemberwiseClone();
            replica.WireShapes(null, inputsShape);

            replica.dBiases = new double[kernels.Length];
            replica.dKernels = new double[kernels.Length][][][];
            for (int f = 0; f < kernels.Length; f++)
            {
                replica.dKernels[f] = new double[kernels[f].Length][][];
                for (int i = 0; i < kernels[f].Length; i++)
                {
                    replica.dKernels[f][i] = new double[kernels[f][i].Length][];
                    for (int j = 0; j < kernels[f][i].Length; j++)
                    {
                        replica.dKernels[f][i][j] = new double[kernels[f][i][j].Length];
                    }
                }
            }

            replica.mini_batch_size = 0;
            return replica;
        }

        // Add the replica's gradients and reset the replica. The batch counter goes with them —
        // UpdateParams divides by it (see Neuron.TakeGradientsFrom).
        internal override void TakeGradientsFrom(Layer replica)
        {
            Conv other = (Conv)replica;

            for (int f = 0; f < dKernels.Length; f++)
            {
                for (int i = 0; i < dKernels[f].Length; i++)
                {
                    for (int j = 0; j < dKernels[f][i].Length; j++)
                    {
                        double[] mine = dKernels[f][i][j];
                        double[] theirs = other.dKernels[f][i][j];
                        for (int c = 0; c < mine.Length; c++)
                        {
                            mine[c] += theirs[c];
                            theirs[c] = 0;
                        }
                    }
                }

                dBiases[f] += other.dBiases[f];
                other.dBiases[f] = 0;
            }

            mini_batch_size += other.mini_batch_size;
            other.mini_batch_size = 0;
        }

        /// <inheritdoc/>
        public override Tensor FeedForward(Tensor input)
        {
            // The shape must match the one the layer is WIRED to. Previously this was never
            // compared: the copy loop below followed the incoming tensor's shape, but the
            // convolution then ran on inputsShape/outputShape. A smaller input overwrote only
            // its corner and the rest was computed from leftovers of the previous sample —
            // silently, without an exception.
            //
            // The shape is normalized only FOR THIS METHOD. Previously inputs.Reshape(...) was
            // called instead, so the layer changed the shape of the caller's tensor — after a pass
            // through the model the user held a different Shape than they had given.
            int[] shape = SpatialShape.RequireWiredShape(input.Shape, inputsShape, Name);

            // Reads straight from the flat data. Previously this was (double[,,])ToArray(),
            // which forced building a multi-dim copy via Array.CreateInstance + per-element
            // SetValue — reflection and boxing on every forward. The row-major order of Data
            // matches [i,j,k], so a single running index is enough.
            double[] inputValues = input.Data;

            int flatIndex = 0;
            for (int i = 0; i < shape[0]; i++)
            {
                for (int j = 0; j < shape[1]; j++)
                {
                    for (int k = 0; k < shape[2]; k++)
                    {
                        inputs[i][j][k] = inputValues[flatIndex++];
                    }
                }
            }

            int inputRows = inputs.Length;
            int inputCols = inputs[0].Length;
            int inputChannels = inputs[0][0].Length;
            int numFilters = kernels.Length;
            int kernelRows = kernels[0].Length;
            int kernelCols = kernels[0][0].Length;
            int kernelChannels = kernels[0][0][0].Length;

            if (kernelChannels != inputChannels)
            {
                throw new ArgumentException("The number of input channels and kernel channels must match.");
            }

            double[][][] paddedInput = PadForSame(inputs);
            

            int outH = outputShape[0];
            int outW = outputShape[1];

            // For each filter (output channel)
            //
            // Jagged-array rows are hoisted one level up (inRow, kRow, pixel, weights) instead of
            // being indexed in the innermost loop: `paddedInput[i + ki][j + kj][c]` means three heap
            // hops and three bounds checks on EVERY multiplication. The summation order is the
            // same, so the result is bit-for-bit identical.
            for (int f = 0; f < numFilters; f++)
            {
                double[][][] kernel = kernels[f];

                for (int i = 0; i < outH; i++)
                {
                    for (int j = 0; j < outW; j++)
                    {
                        double sum = 0;
                        for (int ki = 0; ki < kernelRows; ki++)
                        {
                            double[][] inRow = paddedInput[i + ki];
                            double[][] kRow = kernel[ki];

                            for (int kj = 0; kj < kernelCols; kj++)
                            {
                                double[] pixel = inRow[j + kj];
                                double[] weights = kRow[kj];

                                for (int c = 0; c < inputChannels; c++)
                                {
                                    sum += pixel[c] * weights[c];
                                }
                            }
                        }
                        // Add the bias of filter f
                        double result = sum + biases[f];
                        output[i][j][f] = activation_func.Apply(result);
                        raw_output[i][j][f] = result;
                    }
                }
            }

            return Tensor.FromJagged3D(output, outputShape);
        }

        // Accumulates kernel and bias gradients from the already filled `dOutput` (called by Backward).
        private void AccumulateKernelAndBiasGradients()
        {
            mini_batch_size++;

            int inChannels = inputs[0][0].Length;

            // Kernel dimensions
            int numFilters = kernels.Length;
            int kH = kernels[0].Length;
            int kW = kernels[0][0].Length;

            // Padding – in "same" mode the input is padded exactly as in the forward pass
            double[][][] paddedInput = PadForSame(inputs);

            // 1. dBiases: for each filter, sum all values of dOutput
            for (int f = 0; f < numFilters; f++)
            {
                for (int i = 0; i < outputShape[0]; i++)
                {
                    for (int j = 0; j < outputShape[1]; j++)
                    {
                        dBiases[f] += dOutput[i][j][f];
                    }
                }
            }

            int outH = outputShape[0];
            int outW = outputShape[1];

            // 2. dKernels: for each filter and each kernel element, multiply the matching values of paddedInput and dOutput
            for (int f = 0; f < numFilters; f++)
            {
                for (int ki = 0; ki < kH; ki++)
                {
                    for (int kj = 0; kj < kW; kj++)
                    {
                        for (int c = 0; c < inChannels; c++)
                        {
                            // rows hoisted one level up — see FeedForward
                            double sum = 0;
                            for (int i = 0; i < outH; i++)
                            {
                                double[][] inRow = paddedInput[i + ki];
                                double[][] dRow = dOutput[i];

                                for (int j = 0; j < outW; j++)
                                {
                                    sum += inRow[j + kj][c] * dRow[j][f];
                                }
                            }
                            dKernels[f][ki][kj][c] += sum;
                        }
                    }
                }
            }
        }

        /// <inheritdoc/>
        public override void UpdateParams()
        {
            // same safeguard as Neuron.UpdateWeightsAndBias — empty batch → NaN kernels
            if (mini_batch_size == 0)
            {
                return;
            }

            int numFilters = kernels.Length;
            int idx = 0;
            // all filters have the same shape; +1 = the filter's bias
            int perFilter = kernels[0].Length * kernels[0][0].Length * kernels[0][0][0].Length + 1;
            optimizer.EnsureCapacity(numFilters * perFilter);
            // L2 applies to kernels (the convolution weights), not to biases — see
            // NetworkContext.L2. It is added to the already averaged gradient.
            double weightDecay = Context?.L2 ?? 0;

            for (int f = 0; f < numFilters; f++)
            {
                int kernelHeight = kernels[f].Length;
                int kernelWidth = kernels[f][0].Length;
                int inChannels = kernels[f][0][0].Length;
                // Update the weights of this filter
                for (int i = 0; i < kernelHeight; i++)
                {
                    for (int j = 0; j < kernelWidth; j++)
                    {
                        for (int c = 0; c < inChannels; c++)
                        {
                            double gradient = dKernels[f][i][j][c] / mini_batch_size
                                              + weightDecay * kernels[f][i][j][c];
                            kernels[f][i][j][c] = optimizer.Update(kernels[f][i][j][c], gradient, idx++);
                        }
                    }
                }
                // Update the bias of this filter
                biases[f] = optimizer.Update(biases[f], dBiases[f] / mini_batch_size, idx++);
            }
            mini_batch_size = 0;
            inicializationK_B_dK_dB();
        }

        // ---------------------------------------------------------------- convention B

        /// <inheritdoc/>
        public override Tensor Backward(Tensor gradOutput, bool preActivation = false,
                                        bool computeInputGradient = true)
        {
            if (preActivation)
                throw new NotSupportedException(
                    "Conv cannot have a layer-wide activation (softmax), " +
                    "so the softmax+CE fusion does not apply to it.");

            double[] gradOut = gradOutput.Data;
            int expected = outputShape[0] * outputShape[1] * outputShape[2];

            if (gradOut.Length != expected)
                throw new ArgumentException(
                    $"Conv has output {outputShape[0]}×{outputShape[1]}×{outputShape[2]} " +
                    $"({expected} values), but the gradient from above has {gradOut.Length}.");

            // 1) ∂L/∂output → ∂L/∂z. Done in one place for both the output layer and hidden
            //    layers, because every layer receives its gradient the same way.
            int n = 0;
            for (int i = 0; i < outputShape[0]; i++)
            {
                for (int j = 0; j < outputShape[1]; j++)
                {
                    for (int f = 0; f < outputShape[2]; f++)
                    {
                        dOutput[i][j][f] = gradOut[n++] * activation_func.Derivative(raw_output[i][j][f]);
                    }
                }
            }

            // 2) kernel and bias gradients from the same dOutput
            AccumulateKernelAndBiasGradients();

            // 3) ∂L/∂input for the layer below — skipped for the first layer; it is a full
            //    correlation whose result nobody would use.
            if (!computeInputGradient) return Tensor.Empty;

            return ComputeInputGradientFromDOutput();
        }

        // ∂L/∂input from the already filled `dOutput` (step 3 of Backward).
        private Tensor ComputeInputGradientFromDOutput()
        {
            double[][][] dInput;
            // Input dimensions
            int H = inputs.Length;
            int W = inputs[0].Length;
            int inChannels = inputs[0][0].Length;

            // Kernel dimensions
            int numFilters = kernels.Length;
            int kH = kernels[0].Length;
            int kW = kernels[0][0].Length;

            // Padding – in "same" mode the input is padded exactly as in the forward pass.
            // The un-pad below uses padTop/padLeft (not total/2) — with an even kernel the padding
            // is asymmetric, so a symmetric offset would point at the wrong pixels.
            (int padTop, int padBottom, int padLeft, int padRight) = SamePaddingAmounts();
            double[][][] paddedInput = PadForSame(inputs);

            int paddedH = paddedInput.Length;
            int paddedW = paddedInput[0].Length;

            // Expected output dimensions (dOutput)
            int outH = dOutput.Length;
            int outW = dOutput[0].Length;


            // Gradient initialization:
            // dInput_padded has the same shape as paddedInput: [paddedH][paddedW][inChannels]
            double[][][] dInputPadded = new double[paddedH][][];
            for (int i = 0; i < paddedH; i++)
            {
                dInputPadded[i] = new double[paddedW][];
                for (int j = 0; j < paddedW; j++)
                {
                    dInputPadded[i][j] = new double[inChannels];
                }
            }

            // 3. dInput (gradient with respect to the padded input)
            // For each output point the gradient is "spread" back onto the input pixels through the kernel values
            for (int f = 0; f < numFilters; f++)
            {
                double[][][] kernel = kernels[f];

                for (int i = 0; i < outH; i++)
                {
                    for (int j = 0; j < outW; j++)
                    {
                        // rows hoisted one level up — see FeedForward
                        double g = dOutput[i][j][f];

                        for (int ki = 0; ki < kH; ki++)
                        {
                            double[][] dstRow = dInputPadded[i + ki];
                            double[][] kRow = kernel[ki];

                            for (int kj = 0; kj < kW; kj++)
                            {
                                double[] dst = dstRow[j + kj];
                                double[] weights = kRow[kj];

                                for (int c = 0; c < inChannels; c++)
                                {
                                    dst[c] += weights[c] * g;
                                }
                            }
                        }
                    }
                }
            }

            // With "same", strip the padding so dInput has the same dimensions as the original inputs
            if (padding == "same")
            {
                dInput = new double[H][][];
                for (int i = 0; i < H; i++)
                {
                    dInput[i] = new double[W][];
                    for (int j = 0; j < W; j++)
                    {
                        dInput[i][j] = new double[inChannels];
                        for (int c = 0; c < inChannels; c++)
                        {
                            dInput[i][j][c] = dInputPadded[i + padTop][j + padLeft][c];
                        }
                    }
                }
            }
            else
            {
                // In "valid" mode dInputPadded already has the dimensions of inputs
                dInput = dInputPadded;
            }

            return Tensor.FromJagged3D(dInput, new int[] { dInput.Length, dInput[0].Length, dInput[0][0].Length });
        }

        private void CheckIsActivationFuncIsNotApplyToLayer()
        {
            if (activation_func.AppliesToLayer)
            {
                throw new ArgumentException(
                    "A convolution layer cannot use a layer-wide activation (e.g. Softmax). " +
                    "Use e.g. ReLU() or Tanh(), and put the softmax on a following Dense layer.");
            }
        }

        // Padding is only "valid" or "same", case-insensitive; it is stored in lower case,
        // so a plain comparison is enough later on. Previously anything other than "same"
        // (e.g. the typo "smae" or "full") silently behaved as "valid".
        private static string NormalizePadding(string padding)
        {
            string normalized = padding.Trim().ToLowerInvariant();

            if (normalized != "valid" && normalized != "same")
            {
                throw new ArgumentException(
                    $"Unknown padding '{padding}'. Use \"valid\" (no padding) or \"same\" (output has the size of the input).",
                    nameof(padding));
            }

            return normalized;
        }
        // How many zero rows/columns to add in "same" mode.
        //
        // Previously this was symmetric, k/2 on each side. That works for an ODD kernel but not
        // an EVEN one: the padded size was in + 2·(k/2) = in + k, so the convolution should have
        // produced in + 1 rows — but the loop only ran to outputShape[0] = in, so the last row
        // and column silently disappeared (no exception, just missing data).
        //
        // Now the total is k-1 (exactly enough for the output to match the input size), split
        // before/after. For odd k it is symmetric as before, for even k asymmetric — the same
        // convention as TensorFlow/Keras/PyTorch.
        private (int top, int bottom, int left, int right) SamePaddingAmounts()
        {
            int totalRows = kernels[0].Length - 1;
            int totalCols = kernels[0][0].Length - 1;

            int top = totalRows / 2;
            int left = totalCols / 2;

            return (top, totalRows - top, left, totalCols - left);
        }

        // Returns the input padded according to the mode; for "valid" returns the original array unchanged.
        // Shared by the forward pass and both backward steps (kernel gradients, input gradient) so they cannot diverge.
        private double[][][] PadForSame(double[][][] input)
        {
            if (padding != "same")
            {
                return input;
            }

            (int top, int bottom, int left, int right) = SamePaddingAmounts();
            return Pad3D(input, top, bottom, left, right);
        }

        private double[][][] Pad3D(double[][][] input, int padTop, int padBottom, int padLeft, int padRight)
        {
            int originalRows = input.Length;
            int originalCols = input[0].Length;
            int channels = input[0][0].Length;
            int newRows = originalRows + padTop + padBottom;
            int newCols = originalCols + padLeft + padRight;

            double[][][] padded = new double[newRows][][];
            for (int i = 0; i < newRows; i++)
            {
                padded[i] = new double[newCols][];
                for (int j = 0; j < newCols; j++)
                {
                    padded[i][j] = new double[channels];
                }
            }

            for (int i = 0; i < originalRows; i++)
            {
                for (int j = 0; j < originalCols; j++)
                {
                    for (int c = 0; c < channels; c++)
                    {
                        padded[i + padTop][j + padLeft][c] = input[i][j][c];
                    }
                }
            }

            return padded;
        }

        private void inicializationK_B_dK_dB(int? number_of_kernels = null, int[]? finall_kernel_Shape = null, bool ChangeKernel = false)
        {
            int nkernels;
            int[] kernel_Shape;

            if (number_of_kernels == null)
            {
                nkernels = kernels.Count();
            }
            else
            {
                nkernels = (int)number_of_kernels;
            }

            if (finall_kernel_Shape == null)
            {
                kernel_Shape = new int[] { kernels[0].Length, kernels[0][0].Length, kernels[0][0][0].Length };
            }
            else
            {
                kernel_Shape = (int[])finall_kernel_Shape;
            }

            
            

            if (ChangeKernel)
            {
                dBiases = new double[nkernels];
                biases = new double[nkernels];
                kernels = new double[(int)nkernels][][][];
                dKernels = new double[(int)nkernels][][][];

                for (int i = 0; i < nkernels; i++)
                {
                    kernels[i] = new double[kernel_Shape[0]][][];
                    dKernels[i] = new double[kernel_Shape[0]][][];
                    for (int j = 0; j < kernel_Shape[0]; j++)
                    {
                        kernels[i][j] = new double[kernel_Shape[1]][];
                        dKernels[i][j] = new double[kernel_Shape[1]][];
                        for (int k = 0; k < kernel_Shape[1]; k++)
                        {
                            kernels[i][j][k] = new double[kernel_Shape[2]];
                            dKernels[i][j][k] = new double[kernel_Shape[2]];
                        }
                    }
                }

                // random source from the model (reproducible via seed); a standalone layer
                // without a model falls back to the global generator
                Random kernelRandom = Context?.Random ?? GeneralNeuralNetworkSettings.Random;

                for (int l = 0; l < nkernels; l++)
                {
                    biases[l] = 0;
                    dBiases[l] = 0;
                    for (int i = 0; i < kernel_Shape[0]; i++)
                    {
                        for (int j = 0; j < kernel_Shape[1]; j++)
                        {
                            for (int k = 0; k < kernel_Shape[2]; k++)
                            {
                                double fanIn = kernel_Shape[0] * kernel_Shape[1] * kernel_Shape[2];
                                double limit = Math.Sqrt(6.0 / fanIn);
                                kernels[l][i][j][k] = (kernelRandom.NextDouble() * 2 - 1) * limit;
                                dKernels[l][i][j][k] = 0;
                            }
                        }
                    }
                }

                

            }
            else
            {
                for (int l = 0; l < nkernels; l++)
                {
                    dBiases[l] = 0;
                    for (int i = 0; i < kernel_Shape[0]; i++)
                    {
                        for (int j = 0; j < kernel_Shape[1]; j++)
                        {
                            for (int k = 0; k < kernel_Shape[2]; k++)
                            {
                                dKernels[l][i][j][k] = 0;
                            }
                        }
                    }
                }
            }
        }
    }
}
