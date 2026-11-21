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

        /// <summary>Snapshot of the last output (after activation) with shape <c>[h,w,filters]</c>.</summary>
        // Same shape as FeedForward returns. Previously this was `new Tensor(output)` over the
        // jagged array: `double[][][]` has Rank 1, so Shape came out as [outH] instead of
        // [h,w,channels]. The data were right, the Shape was not — visible only when Conv is the
        // output layer and someone reads Shape. It was also a view into the working array, not a snapshot.
        public override Tensor Output => Tensor.FromJagged3D(output, OutputShape!);

        /// <summary>Snapshot of the last output before activation, shape <c>[h,w,filters]</c>.</summary>
        public override Tensor RawOutput => Tensor.FromJagged3D(raw_output, OutputShape!);

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

        /// <summary>Number of filters (output channels).</summary>
        public int Filters => filters;

        /// <summary>Size of the square kernel.</summary>
        public int KernelSize => kernelRows;

        /// <summary>Kernel weights indexed <c>[filter][kernelRow][kernelCol][inputChannel]</c>; empty until the layer is built.</summary>
        public double[][][][] Kernels  // [numFilters][kernelRows][kernelCols][inputChannels]
        {
            get { return kernels; }
        }
        /// <summary>One bias per filter; empty until the layer is built.</summary>
        public double[] Biases
        {
            get { return biases; }
        }
        /// <summary>Padding mode, lower-case: <c>"valid"</c> or <c>"same"</c>.</summary>
        public string Padding
        {
            get { return padding; }
        }

        // ---------------------------------------------------------------- description
        //
        // Previously the kernel size was not stored anywhere: the constructor created placeholder
        // kernels with ZERO input channels and the size was read back as kernels[0].Length. The
        // description is now explicit, and the kernels exist only once the layer is built.
        private readonly int filters;
        private readonly int kernelRows;
        private readonly int kernelCols;
        private readonly string padding;
        private ActivationFunction activation_func;

        // ---------------------------------------------------------------- built state
        //
        // Empty until built; BuildCore / the loading constructor fill them.
        private double[][][][] kernels = Array.Empty<double[][][]>();
        internal double[][][][] dKernels = Array.Empty<double[][][]>();   // internal for gradient-check tests
        private double[] biases = Array.Empty<double>();
        internal double[] dBiases = Array.Empty<double>();                // internal for gradient-check tests
        private double[][][] dOutput = Array.Empty<double[][]>();
        private double[][][] output = Array.Empty<double[][]>();
        private double[][][] raw_output = Array.Empty<double[][]>();
        private double[][][] inputs = Array.Empty<double[][]>();

        private int mini_batch_size;

        // null until the layer is built (an unbuilt layer has no parameters to optimize).
        private Optimizer? optimizer;

        // For saving the optimizer state to a file (ExportCNNLayer); only a built layer is saved.
        internal Optimizer OptimizerForSave => optimizer!;

        internal override void ReplaceOptimizer(Optimizer prototype)
        {
            if (IsBuilt)
            {
                optimizer = prototype.CreateFresh();
            }
        }

        // `optimizerPrototype` — see Dense(ExportDenseLayer, …)
        internal Conv(ExportCNNLayer layer, Optimizer? optimizerPrototype = null)
        {
            kernels = BaseExportLayer.Required(layer.Kernel, Name, nameof(layer.Kernel));
            biases = BaseExportLayer.Required(layer.Biases, Name, nameof(layer.Biases));
            int[] inputShape = BaseExportLayer.Required(layer.InputShape, Name, nameof(layer.InputShape));
            BaseExportLayer.RequireShape(inputShape, 3, Name, nameof(layer.InputShape));
            ValidateSavedKernels(kernels, biases, inputShape[2]);
            activation_func = ActivationFunction.FromName(layer.Name_of_activation_function);
            CheckIsActivationFuncIsNotApplyToLayer();
            padding = NormalizePadding(BaseExportLayer.Required(layer.Padding, Name, nameof(layer.Padding)));

            filters = kernels.Length;
            kernelRows = kernels[0].Length;
            kernelCols = kernels[0][0].Length;

            optimizer = (optimizerPrototype ?? GeneralNeuralNetworkSettings.DefaultOptimizer).CreateFresh();

            // resuming training — see Neuron(ExportNeuron, …)
            if (layer.OptimizerState != null)
            {
                optimizer.ImportState(layer.OptimizerState);
            }

            // The same allocation as when building — previously the loading constructor had its own
            // copy of it, a second path that could drift from the first.
            MarkBuiltFromFile(inputShape);
            AllocateBuffers(InputShape!, OutputShape!);
            dBiases = new double[filters];
            dKernels = AllocateKernelArray(inputShape[2]);

            mini_batch_size = 0;
        }

        // The kernels from a file must be a regular [filters][rows][cols][channels] block whose channel
        // count matches the input, with one bias per filter. Previously only their presence was
        // checked: "Kernel": [[]] crashed with IndexOutOfRange in this constructor, and a file whose
        // kernel channels differed from InputShape loaded fine and failed deep in forward or update.
        private void ValidateSavedKernels(double[][][][] saved, double[] savedBiases, int channels)
        {
            string Problem()
            {
                if (saved.Length == 0) return "no filters";
                int rows = saved[0]?.Length ?? 0;
                int cols = rows > 0 ? saved[0][0]?.Length ?? 0 : 0;
                if (rows == 0 || cols == 0) return "an empty kernel";

                foreach (double[][][]? filter in saved)
                {
                    if (filter == null || filter.Length != rows) return "filters of different sizes";
                    foreach (double[][]? row in filter)
                    {
                        if (row == null || row.Length != cols) return "filters of different sizes";
                        foreach (double[]? pixel in row)
                        {
                            if (pixel == null || pixel.Length != channels)
                                return $"kernels with a different channel count than the input ({channels})";
                        }
                    }
                }

                if (savedBiases.Length != saved.Length)
                    return $"{savedBiases.Length} biases for {saved.Length} filters";

                return "";
            }

            string problem = Problem();
            if (problem != "")
            {
                throw new ModelFileCorruptedException($"Layer {Name} in the model file has {problem}.");
            }
        }

        /// <summary>Creates a convolution layer. This is only a description — the kernels are created
        /// when the layer is built for its input (inside a model: on the first pass).</summary>
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

            this.filters = filters;
            kernelRows = kernelSize;
            kernelCols = kernelSize;
            this.padding = NormalizePadding(padding);
            mini_batch_size = 0;
        }

        // ---------------------------------------------------------------- building

        /// <summary>Conv works on <c>[height, width, channels]</c>; see SpatialShape.Normalize.</summary>
        protected override int[] NormalizeInputShape(int[] inputShape) => SpatialShape.Normalize(inputShape, Name);

        /// <inheritdoc/>
        protected override int[] ComputeOutputShape(int[] inputShape)
        {
            // Without padding the kernel must fit into the input — otherwise the output would
            // have a negative or zero dimension and it would only crash on buffer allocation.
            if (padding == "valid" && (inputShape[0] < kernelRows || inputShape[1] < kernelCols))
            {
                throw new ArgumentException(
                    $"{Name}: a {kernelRows}×{kernelCols} kernel does not fit the input " +
                    $"{SpatialShape.Describe(inputShape)} with \"valid\" padding. Use a smaller kernel, " +
                    "or padding \"same\".");
            }

            return padding == "same"
                ? new int[] { inputShape[0], inputShape[1], filters }
                : new int[] { inputShape[0] - kernelRows + 1, inputShape[1] - kernelCols + 1, filters };
        }

        /// <inheritdoc/>
        protected override void BuildCore(int[] inputShape, Random random, Optimizer optimizerPrototype)
        {
            // The clone is made BEFORE the kernels are drawn: if cloning throws, nothing has been drawn
            // and the layer stays unbuilt (the order of the random draws is unaffected by it).
            Optimizer fresh = optimizerPrototype.CreateFresh();

            AllocateBuffers(inputShape, ComputeOutputShape(inputShape));

            int channels = inputShape[2];
            kernels = AllocateKernelArray(channels);
            dKernels = AllocateKernelArray(channels);
            biases = new double[filters];
            dBiases = new double[filters];

            // He-uniform; the same loop order as before, so a seeded model gets the same kernels
            double fanIn = kernelRows * kernelCols * channels;
            double limit = Math.Sqrt(6.0 / fanIn);

            for (int l = 0; l < filters; l++)
            {
                for (int i = 0; i < kernelRows; i++)
                {
                    for (int j = 0; j < kernelCols; j++)
                    {
                        for (int k = 0; k < channels; k++)
                        {
                            kernels[l][i][j][k] = (random.NextDouble() * 2 - 1) * limit;
                        }
                    }
                }
            }

            optimizer = fresh;
            mini_batch_size = 0;
        }

        /// <inheritdoc/>
        protected override void OnInvalidated()
        {
            kernels = Array.Empty<double[][][]>();
            dKernels = Array.Empty<double[][][]>();
            biases = Array.Empty<double>();
            dBiases = Array.Empty<double>();
            dOutput = Array.Empty<double[][]>();
            output = Array.Empty<double[][]>();
            raw_output = Array.Empty<double[][]>();
            inputs = Array.Empty<double[][]>();
            optimizer = null;
            mini_batch_size = 0;
        }

        // [filters][kernelRows][kernelCols][channels], zero-filled
        private double[][][][] AllocateKernelArray(int channels)
        {
            var array = new double[filters][][][];
            for (int f = 0; f < filters; f++)
            {
                array[f] = new double[kernelRows][][];
                for (int i = 0; i < kernelRows; i++)
                {
                    array[f][i] = new double[kernelCols][];
                    for (int j = 0; j < kernelCols; j++)
                    {
                        array[f][i][j] = new double[channels];
                    }
                }
            }
            return array;
        }

        // Input and output buffers for one sample. Used when building, when loading and by replicas.
        private void AllocateBuffers(int[] inputShape, int[] outputShape)
        {
            inputs = Allocate3D(inputShape);
            output = Allocate3D(outputShape);
            raw_output = Allocate3D(outputShape);
            dOutput = Allocate3D(outputShape);
        }

        private static double[][][] Allocate3D(int[] shape)
        {
            var array = new double[shape[0]][][];
            for (int i = 0; i < shape[0]; i++)
            {
                array[i] = new double[shape[1]][];
                for (int j = 0; j < shape[1]; j++)
                {
                    array[i][j] = new double[shape[2]];
                }
            }
            return array;
        }

        internal override bool CanImport(BaseExportLayer saved)
        {
            return IsBuilt
                   && saved is ExportCNNLayer conv
                   && conv.Name_of_activation_function == activation_func.Name
                   && conv.Padding != null && NormalizePadding(conv.Padding) == padding
                   && conv.InputShape != null && SpatialShape.SameAs(conv.InputShape, InputShape!)
                   && conv.Kernel != null && conv.Biases != null
                   && SameKernelShape(conv.Kernel)
                   && conv.Biases.Length == filters;
        }

        private bool SameKernelShape(double[][][][] other)
        {
            if (other.Length != kernels.Length) return false;
            for (int f = 0; f < kernels.Length; f++)
            {
                if (other[f].Length != kernelRows) return false;
                for (int i = 0; i < kernelRows; i++)
                {
                    if (other[f][i].Length != kernelCols) return false;
                    for (int j = 0; j < kernelCols; j++)
                    {
                        if (other[f][i][j].Length != kernels[f][i][j].Length) return false;
                    }
                }
            }
            return true;
        }

        internal override Action PrepareImport(BaseExportLayer saved, Optimizer optimizerPrototype)
        {
            var conv = (ExportCNNLayer)saved;

            // what can fail — nothing is written yet (see Layer.PrepareImport)
            Optimizer prepared = optimizerPrototype.CreateFresh();
            if (conv.OptimizerState != null)
            {
                prepared.ImportState(conv.OptimizerState);
            }

            return () =>
            {
                // copied INTO the existing arrays — replicas share them by reference
                for (int f = 0; f < kernels.Length; f++)
                    for (int i = 0; i < kernelRows; i++)
                        for (int j = 0; j < kernelCols; j++)
                            Array.Copy(conv.Kernel![f][i][j], kernels[f][i][j], kernels[f][i][j].Length);
                Array.Copy(conv.Biases!, biases, filters);

                optimizer = prepared;

                // the same state as a layer freshly loaded from the file
                AllocateBuffers(InputShape!, OutputShape!);
                ResetGradients();
                mini_batch_size = 0;
            };
        }

        // Replica for parallel training — see Layer.CreateReplica. Kernels and biases are SHARED
        // by reference (the original's optimizer updates them in place, after the batch), so
        // nothing is copied between batches. Owned: input and output buffers and gradient accumulators.
        internal override Layer CreateReplica()
        {
            ThrowIfNotBuiltForReplica();

            Conv replica = (Conv)MemberwiseClone();
            replica.AllocateBuffers(InputShape!, OutputShape!);
            replica.dBiases = new double[filters];
            replica.dKernels = AllocateKernelArray(InputShape![2]);
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
            // The shape must match the one the layer is BUILT for. Previously this was never
            // compared: the copy loop below followed the incoming tensor's shape, but the
            // convolution then ran on the built shapes. A smaller input overwrote only
            // its corner and the rest was computed from leftovers of the previous sample —
            // silently, without an exception.
            //
            // The shape is normalized only FOR THIS METHOD. Previously inputs.Reshape(...) was
            // called instead, so the layer changed the shape of the caller's tensor — after a pass
            // through the model the user held a different Shape than they had given.
            int[] shape = SpatialShape.RequireBuiltShape(input.Shape, InputShape ?? Array.Empty<int>(), Name);

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

            int inputChannels = inputs[0][0].Length;
            int numFilters = kernels.Length;
            int kernelChannels = kernels[0][0][0].Length;

            if (kernelChannels != inputChannels)
            {
                throw new ArgumentException("The number of input channels and kernel channels must match.");
            }

            double[][][] paddedInput = PadForSame(inputs);


            int outH = OutputShape![0];
            int outW = OutputShape[1];

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

            return Tensor.FromJagged3D(output, OutputShape);
        }

        // Accumulates kernel and bias gradients from the already filled `dOutput` (called by Backward).
        private void AccumulateKernelAndBiasGradients()
        {
            mini_batch_size++;

            int inChannels = inputs[0][0].Length;

            int numFilters = kernels.Length;
            int outH = OutputShape![0];
            int outW = OutputShape[1];

            // Padding – in "same" mode the input is padded exactly as in the forward pass
            double[][][] paddedInput = PadForSame(inputs);

            // 1. dBiases: for each filter, sum all values of dOutput
            for (int f = 0; f < numFilters; f++)
            {
                for (int i = 0; i < outH; i++)
                {
                    for (int j = 0; j < outW; j++)
                    {
                        dBiases[f] += dOutput[i][j][f];
                    }
                }
            }

            // 2. dKernels: for each filter and each kernel element, multiply the matching values of paddedInput and dOutput
            for (int f = 0; f < numFilters; f++)
            {
                for (int ki = 0; ki < kernelRows; ki++)
                {
                    for (int kj = 0; kj < kernelCols; kj++)
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

            Optimizer opt = optimizer!;   // a gradient was accumulated, so the layer is built

            int numFilters = kernels.Length;
            int idx = 0;
            // all filters have the same shape; +1 = the filter's bias
            int perFilter = kernelRows * kernelCols * kernels[0][0][0].Length + 1;
            opt.EnsureCapacity(numFilters * perFilter);
            // L2 applies to kernels (the convolution weights), not to biases — see
            // NetworkContext.L2. It is added to the already averaged gradient.
            double weightDecay = Context?.L2 ?? 0;

            for (int f = 0; f < numFilters; f++)
            {
                int inChannels = kernels[f][0][0].Length;
                // Update the weights of this filter
                for (int i = 0; i < kernelRows; i++)
                {
                    for (int j = 0; j < kernelCols; j++)
                    {
                        for (int c = 0; c < inChannels; c++)
                        {
                            double gradient = dKernels[f][i][j][c] / mini_batch_size
                                              + weightDecay * kernels[f][i][j][c];
                            kernels[f][i][j][c] = opt.Update(kernels[f][i][j][c], gradient, idx++);
                        }
                    }
                }
                // Update the bias of this filter
                biases[f] = opt.Update(biases[f], dBiases[f] / mini_batch_size, idx++);
            }
            mini_batch_size = 0;
            ResetGradients();
        }

        private void ResetGradients()
        {
            for (int f = 0; f < dKernels.Length; f++)
            {
                dBiases[f] = 0;
                for (int i = 0; i < dKernels[f].Length; i++)
                    for (int j = 0; j < dKernels[f][i].Length; j++)
                        Array.Clear(dKernels[f][i][j]);
            }
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

            int[] outputShape = OutputShape!;
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

            int numFilters = kernels.Length;

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

                        for (int ki = 0; ki < kernelRows; ki++)
                        {
                            double[][] dstRow = dInputPadded[i + ki];
                            double[][] kRow = kernel[ki];

                            for (int kj = 0; kj < kernelCols; kj++)
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
            int totalRows = kernelRows - 1;
            int totalCols = kernelCols - 1;

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
    }
}
