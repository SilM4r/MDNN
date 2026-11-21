using MDNN.Layers;
using MDNN.Optimizers;
using MDNN.Serialization;
namespace MDNN.Layers
{
    /// <summary>
    /// 2D max pooling with a square window and stride equal to the window size (valid, no padding).
    /// </summary>
    public class MaxPool : ParameterlessLayer
    {
        /// <inheritdoc/>
        public override string Name => "MaxPool";

        /// <summary>Snapshot of the last output with shape <c>[h,w,channels]</c>, as returned by FeedForward.</summary>
        // Previously `new Tensor(output)` over the jagged array gave Shape [outH] instead of
        // [h,w,channels] (a jagged array has Rank 1) and pointed into the working array,
        // which is overwritten by the next pass.
        public override Tensor? Output
        {
            get
            {
                if (output == null)
                {
                    return null;
                }
                else
                {
                    return Tensor.FromJagged3D(output, OutputShape!);
                }
            }
        }

        /// <summary>Same as <see cref="Output"/> — MaxPool has no activation of its own.</summary>
        public override Tensor? RawOutput => Output;

        /// <summary>Size of the square pooling window (also the stride).</summary>
        public int PoolSize
        {
            get { return poolSize; }
        }

        // buffers — null until the layer is built
        private double[][][]? output;
        private double[][][]? inputs;
        private PoolingIndex[,,]? indexes;

        private readonly int poolSize;

        /// <summary>Creates a max-pooling layer; the input shape is set when the layer is built (inside a model: on the first pass).</summary>
        /// <param name="poolSize">Window size and stride; must be greater than 0.</param>
        public MaxPool(int poolSize)
        {
            // a window of 0 would divide by zero when computing the output size; negative makes no sense
            if (poolSize <= 0)
                throw new ArgumentOutOfRangeException(nameof(poolSize), poolSize,
                    "The pool size must be greater than 0.");

            this.poolSize = poolSize;
        }
        internal MaxPool(ExportMaxPoolLayer layer)
        {
            poolSize = BaseExportLayer.Required(layer.PoolSize, Name, nameof(layer.PoolSize));
            int[] inputShape = BaseExportLayer.Required(layer.InputsShape, Name, nameof(layer.InputsShape));

            // the same allocation as when building — see Conv(ExportCNNLayer, …)
            MarkBuiltFromFile(inputShape);
            AllocateBuffers(InputShape!, OutputShape!);
        }

        // ---------------------------------------------------------------- building

        /// <summary>MaxPool works on <c>[height, width, channels]</c>; see SpatialShape.Normalize.</summary>
        protected override int[] NormalizeInputShape(int[] inputShape) => SpatialShape.Normalize(inputShape, Name);

        /// <inheritdoc/>
        protected override int[] ComputeOutputShape(int[] inputShape)
        {
            // Output dimensions for valid pooling (no padding)
            int outHeight = (inputShape[0] - poolSize) / poolSize + 1;
            int outWidth = (inputShape[1] - poolSize) / poolSize + 1;

            return new int[] { outHeight, outWidth, inputShape[2] };
        }

        /// <inheritdoc/>
        protected override void BuildCore(int[] inputShape, Random random, Optimizer optimizerPrototype)
        {
            AllocateBuffers(inputShape, ComputeOutputShape(inputShape));
        }

        /// <inheritdoc/>
        protected override void OnInvalidated()
        {
            inputs = null;
            output = null;
            indexes = null;
        }

        private void AllocateBuffers(int[] inputShape, int[] outputShape)
        {
            inputs = new double[inputShape[0]][][];
            for (int i = 0; i < inputShape[0]; i++)
            {
                inputs[i] = new double[inputShape[1]][];
                for (int j = 0; j < inputShape[1]; j++)
                {
                    inputs[i][j] = new double[inputShape[2]];
                }
            }

            indexes = new PoolingIndex[outputShape[0], outputShape[1], outputShape[2]];

            output = new double[outputShape[0]][][];
            for (int i = 0; i < outputShape[0]; i++)
            {
                output[i] = new double[outputShape[1]][];
                for (int j = 0; j < outputShape[1]; j++)
                {
                    output[i][j] = new double[outputShape[2]];
                }
            }
        }

        // There are no parameters; the saved layer only has to describe the same pooling.
        internal override bool CanImport(BaseExportLayer saved)
        {
            return IsBuilt
                   && saved is ExportMaxPoolLayer pool
                   && pool.PoolSize == poolSize
                   && pool.InputsShape != null && SpatialShape.SameAs(pool.InputsShape, InputShape!);
        }

        internal override void ImportParameters(BaseExportLayer saved, Optimizer optimizerPrototype)
        {
            AllocateBuffers(InputShape!, OutputShape!);   // the same state as a freshly loaded layer
        }

        // ---------------------------------------------------------------- convention B

        /// <inheritdoc/>
        public override Tensor Backward(Tensor gradOutput, bool preActivation = false,
                                        bool computeInputGradient = true)
        {
            if (preActivation)
                throw new NotSupportedException(
                    "MaxPool has no activation, so preActivation makes no sense. " +
                    "The softmax+CE fusion only applies to the output Dense layer.");

            if (output == null || indexes == null)
                throw new InvalidOperationException(
                    "MaxPool: backward before forward — the positions of the maxima do not exist yet.");

            // The gradient arrives element-wise, with no reaching into the neighbour's weights.
            // Previously MaxPool needed its own "next layer is Dense" branch; now Dense computes
            // its ∂L/∂input itself and passes it here ready-made.
            double[] upstream = gradOutput.Data;

            int outHeight = output.Length;
            int outWidth = output[0].Length;
            int channels = output[0][0].Length;

            if (upstream.Length != outHeight * outWidth * channels)
                throw new ArgumentException(
                    $"MaxPool has output {outHeight}×{outWidth}×{channels} " +
                    $"({outHeight * outWidth * channels} values), but the gradient from above " +
                    $"has {upstream.Length}.");

            if (!computeInputGradient) return Tensor.Empty;

            int inputHeight = inputs!.Length;
            int inputWidth = inputs[0].Length;

            double[][][] dInput = new double[inputHeight][][];
            for (int i = 0; i < inputHeight; i++)
            {
                dInput[i] = new double[inputWidth][];
                for (int j = 0; j < inputWidth; j++) dInput[i][j] = new double[channels];
            }

            // The gradient flows ONLY to the position that won the window; the others did not
            // contribute to the output, so they get zero.
            int flat = 0;
            for (int i = 0; i < outHeight; i++)
            {
                for (int j = 0; j < outWidth; j++)
                {
                    for (int c = 0; c < channels; c++)
                    {
                        PoolingIndex idx = indexes[i, j, c];
                        dInput[idx.Row][idx.Col][c] += upstream[flat++];
                    }
                }
            }

            return Tensor.FromJagged3D(dInput,
                new int[] { inputHeight, inputWidth, channels });
        }

        /// <inheritdoc/>
        public override Tensor FeedForward(Tensor input)
        {
            // Same guard as in Conv: the shape must match the built one (it also says when the
            // layer is not built at all). Previously a smaller input overwrote only a corner of the
            // buffer and the rest kept the previous sample — MaxPool then returned the previous
            // image's maximum from a window of zeros. Rank ≥ 4 used to fail with a
            // NotSupportedException() without any message.
            // The shape is local to this method; the caller's tensor stays untouched — see Conv.FeedForward.
            int[] shape = SpatialShape.RequireBuiltShape(input.Shape, InputShape ?? Array.Empty<int>(), Name);

            double[][][] inputs = this.inputs!;
            double[][][] output = this.output!;
            PoolingIndex[,,] indexes = this.indexes!;

            // flat path instead of (double[,,])ToArray() — see Conv.FeedForward
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

            int inputHeight = inputs.Length;
            int inputWidth = inputs[0].Length;
            int channels = inputs[0][0].Length;

            int outHeight = OutputShape![0];
            int outWidth = OutputShape[1];

            // For each channel and each pooling region, find the maximum value
            for (int c = 0; c < channels; c++)
            {
                for (int i = 0; i < outHeight; i++)
                {
                    for (int j = 0; j < outWidth; j++)
                    {
                        double maxVal = double.NegativeInfinity;
                        int maxRow = -1;
                        int maxCol = -1;
                        // Iterate over the pooling window
                        for (int m = 0; m < poolSize; m++)
                        {
                            for (int n = 0; n < poolSize; n++)
                            {
                                int curRow = i * poolSize + m;
                                int curCol = j * poolSize + n;
                                if (curRow < inputHeight && curCol < inputWidth)
                                {
                                    double val = inputs[curRow][curCol][c];
                                    if (val > maxVal)
                                    {
                                        maxVal = val;
                                        maxRow = curRow;
                                        maxCol = curCol;
                                    }


                                }
                            }
                        }
                        output[i][j][c] = maxVal;
                        indexes[i,j,c] = new PoolingIndex(maxRow, maxCol);
                    }
                }
            }
            return Tensor.FromJagged3D(output, OutputShape);
        }

        // Replica for parallel training — see Layer.CreateReplica. There are no parameters, so
        // own buffers are enough.
        internal override Layer CreateReplica()
        {
            ThrowIfNotBuiltForReplica();

            MaxPool replica = (MaxPool)MemberwiseClone();
            replica.AllocateBuffers(InputShape!, OutputShape!);
            return replica;
        }
    }
}
