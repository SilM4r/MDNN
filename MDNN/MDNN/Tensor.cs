namespace MDNN
{
    /// <summary>
    /// An n-dimensional array of doubles: flat data in row-major order plus a shape.
    /// </summary>
    public class Tensor
    {
        /// <summary>Size of each dimension.</summary>
        public int[] Shape { get; private set; }
        /// <summary>The elements as a flat array in row-major order.</summary>
        public double[] Data
        {
            get 
            { 
                if (_data == null)
                {
                    _data = FlattenArray(OriginalInput);
                    return _data;
                }
                else
                {
                    return _data;
                }
            }
        }

        private Array OriginalInput
        {
            get
            {
                if (_originalInput == null)
                {
                    _originalInput = ReshapeToArray(Data, Shape); 
                    return _originalInput;
                }
                else
                {
                    return _originalInput;
                }
            }
            set { _originalInput = value; }
        }

        private Array? _originalInput = null;
        private double[]? _data = null;

        /// <summary>
        /// Creates a tensor from any regular double/int/float array — 1D, multidimensional or jagged.
        /// The shape is taken from the array.
        /// </summary>
        public Tensor(Array input)
        {
            Type scalarType = ScalarType(input, out bool isJagged);

            // double[], double[,,] … — the shape comes from Rank, the data is copied lazily on first .Data
            if (!isJagged && scalarType == typeof(double))
            {
                OriginalInput = input;
                Shape = Enumerable.Range(0, input.Rank)
                                  .Select(input.GetLength)
                                  .ToArray();
                return;
            }

            // A jagged array always has Rank 1, so the shape CANNOT be read from Rank. The
            // constructor used to do exactly that: double[4][4][1] became Shape [4] over 16
            // elements, Conv silently returned nonsense ([1,1,2] instead of [3,3,2]) and Dense
            // threw IndexOutOfRange. GetJaggedShape also rejects irregular arrays.
            //
            // Uses the flat path, not ConvertJaggedToMulti (Array.CreateInstance + per-element
            // SetValue = reflection and boxing on every double). Measured on 5000×784:
            //   double[][]  360.0 ms / 748.0 MB      <- reflection path
            //   int[][]      12.9 ms /  29.9 MB      <- flat path
            //
            // int/float are converted to double right away, so OriginalInput is double like everywhere else.
            Shape = isJagged
                ? GetJaggedShape(input)
                : Enumerable.Range(0, input.Rank).Select(input.GetLength).ToArray();
            _data = FlattenArray(input);
        }

        /// <summary>
        /// Wraps flat row-major <paramref name="data"/> with the given <paramref name="shape"/>
        /// without copying it — the tensor and the caller share the array.
        /// </summary>
        public Tensor(double[] data, int[] shape)
        {
            ArgumentNullException.ThrowIfNull(data);
            ArgumentNullException.ThrowIfNull(shape);

            int size = shape.Aggregate(1, (a, b) => a * b);
            if (size != data.Length)
                throw new ArgumentException(
                    $"The shape [{string.Join(", ", shape)}] has {size} elements, but the data has {data.Length}.");

            Shape = shape;
            _data = data;
        }
        /// <summary>
        /// Changes the shape in place; the number of elements must stay the same.
        /// </summary>
        public void Reshape(int[] newShape)
        {
            int newSize = newShape.Aggregate(1, (a, b) => a * b);

            if (newSize != Data.Length)
                throw new ArgumentException("The new shape must have the same number of elements as the original.");

            Shape = newShape;
            _originalInput = null;
        }

        // Previously this used Array.CreateInstance + a recursive FillSubTensor copying element
        // by element via GetValue/SetValue — reflection and boxing on every double. Measured:
        // 48 KB allocated per ONE sample slice. `TrainLoop` calls this for every sample of every
        // epoch: for MNIST (49k samples, 20 epochs) that is ~1M calls, tens of GB just for slicing.
        //
        // The data is flat in row-major order, so the slice is a contiguous block — compute the
        // offset and length and copy once.
        /// <summary>
        /// Returns the sub-tensor selected by fixing the first <c>indices.Length</c> dimensions
        /// (e.g. <c>GetSubTensor(i)</c> = the i-th sample). The result is a copy.
        /// </summary>
        public Tensor GetSubTensor(params int[] indices)
        {
            if (indices.Length >= Shape.Length)
                throw new ArgumentException("The number of indices must be less than the number of tensor dimensions.");

            // block length = product of the remaining dimensions
            int blockSize = 1;
            for (int d = indices.Length; d < Shape.Length; d++)
            {
                blockSize *= Shape[d];
            }

            // offset = sum of index[d] * (product of the dimensions to the right of d)
            int offset = 0;
            int stride = blockSize;
            for (int d = indices.Length - 1; d >= 0; d--)
            {
                if (indices[d] < 0 || indices[d] >= Shape[d])
                {
                    throw new ArgumentOutOfRangeException(nameof(indices),
                        $"Index {indices[d]} is out of range for dimension {d} (size {Shape[d]}).");
                }

                offset += indices[d] * stride;
                stride *= Shape[d];
            }

            double[] block = new double[blockSize];
            Array.Copy(Data, offset, block, 0, blockSize);

            int[] blockShape = new int[Shape.Length - indices.Length];
            Array.Copy(Shape, indices.Length, blockShape, 0, blockShape.Length);

            return new Tensor(block, blockShape);
        }

        /// <summary>
        /// The data as a regular .NET array of the tensor's shape (<c>double[]</c>, <c>double[,]</c>, …).
        /// </summary>
        public Array ToArray()
        {
            return OriginalInput;
        }

        private static Array ReshapeToArray(double[] data, int[] shape)
        {
            return BuildMultiDimArray(data, shape, 0);
        }

        /// <summary>
        /// An empty tensor (0 elements). Returned by <c>Backward(computeInputGradient: false)</c> —
        /// clearer than null, and a mistake fails on the length rather than producing nonsense numbers.
        /// </summary>
        public static Tensor Empty => new Tensor(Array.Empty<double>());

        /// <summary>
        /// Returns a copy of <paramref name="length"/> consecutive items along the first dimension,
        /// starting at <paramref name="start"/>.
        /// </summary>
        public Tensor Slice(int start, int length)
        {
            if (start < 0 || length <= 0 || start + length > Shape[0])
                throw new ArgumentException("Invalid slice range!");


            int sampleSize = Data.Length / Shape[0]; // size of one sample (along the first dimension)
            double[] slicedData = new double[length * sampleSize];

            Array.Copy(Data, start * sampleSize, slicedData, 0, length * sampleSize);

            int[] newShape = (int[])Shape.Clone();
            newShape[0] = length; // the first dimension is now shortened

            return new Tensor(slicedData, newShape);
        }

        // Layers return their output through this. They used to do `new Tensor(output)`, but
        // `output` is a working array the layer overwrites on every pass, and Tensor copies the
        // data only on the first access to `.Data`. A result kept from the first forward pass
        // therefore CHANGED RETROACTIVELY after the second one.
        //
        // It costs nothing extra: `new Tensor(array)` computes Shape via LINQ and makes the copy
        // in `FlattenArray` on first read anyway. Here `_data` is set directly — the same one copy,
        // minus the LINQ. Measured: 200k passes over a 128-neuron layer 21 ms / 243 MB → 13 ms / 214 MB.
        /// <summary>
        /// A snapshot of the vector as a 1D tensor — a COPY, not a window into the source array.
        /// </summary>
        public static Tensor FromVector(double[] source)
        {
            return new Tensor((double[])source.Clone(), new int[] { source.Length });
        }

        // Flattens an [h][w][c] jagged array directly into a Tensor of the given shape.
        //
        // Conv and MaxPool used to return `new Tensor(ConvertJaggedToMulti(output))`, which on
        // EVERY forward pass built a copy of the whole output via reflection (Array.CreateInstance
        // + per-element SetValue, i.e. boxing on every double). Measured: 8 MB allocated per CNN
        // sample. Here it is a single pass writing into a flat array.
        internal static Tensor FromJagged3D(double[][][] source, int[] shape)
        {
            double[] flat = new double[shape[0] * shape[1] * shape[2]];

            int index = 0;
            for (int i = 0; i < shape[0]; i++)
            {
                for (int j = 0; j < shape[1]; j++)
                {
                    double[] row = source[i][j];
                    for (int k = 0; k < shape[2]; k++)
                    {
                        flat[index++] = row[k];
                    }
                }
            }

            return new Tensor(flat, shape);
        }

        internal static Array ConvertJaggedToMulti(Array jaggedArray)
        {
            int[] shape = GetJaggedShape(jaggedArray);

            // Create an empty multidimensional array of the matching shape
            Array multiArray = Array.CreateInstance(typeof(double), shape);

            // Recursively copy the values into the multidimensional array
            CopyJaggedToMulti(jaggedArray, multiArray, new int[0]);

            return multiArray;
        }

        // Scalar type of the array. Unwraps ALL levels of a jagged array, not just one. Previously
        // only one level was peeled, so `double[][]` passed but `double[][][]` failed with
        // "invalid type on input array" — exactly the shape of sequence data
        // ([sequence][step][feature]), the very case sequence training exists for.
        private static Type ScalarType(Array array, out bool isJagged)
        {
            Type elementType = array.GetType().GetElementType()!;
            isJagged = false;

            while (elementType.IsArray)
            {
                elementType = elementType.GetElementType()!;
                isJagged = true;
            }

            // Compare types, not names: .NET names are "Int32"/"Single"/"Double", so the former
            // check for "Int"/"Float" never matched and the library rejected int and float inputs —
            // even though its own error message promised them.
            if (elementType != typeof(double) && elementType != typeof(int) && elementType != typeof(float))
            {
                throw new ArgumentException(
                    $"A tensor can only be created from a double, int or float array, not {elementType.Name}.");
            }

            return elementType;
        }

        // Shape of a jagged array — and a check that it really is REGULAR.
        //
        // The shape is taken from the first sub-array at each level. Without the check the rest
        // was silently assumed to match, so an irregular array passed and the data got scrambled:
        //   int[][] { {1,2}, {3,4,5}, {6} }  ->  shape [3,2] over six elements
        // Six elements fit into 3×2, so not even the length gave it away. For double[][] it
        // instead threw IndexOutOfRange deep in the copying — three types, three different
        // behaviours, none of them correct.
        private static int[] GetJaggedShape(Array array)
        {
            List<int> shape = new List<int>();
            Array? level = array;

            while (level != null && level.Length > 0)
            {
                shape.Add(level.Length);
                level = level.GetValue(0) as Array;
            }

            int[] result = shape.ToArray();
            EnsureRectangular(array, result, 0);
            return result;
        }

        // All sub-arrays at the same level must have the same length, otherwise the array has no shape.
        private static void EnsureRectangular(Array array, int[] shape, int depth)
        {
            if (depth >= shape.Length)
            {
                return;
            }

            if (array.Length != shape[depth])
            {
                throw new ArgumentException(
                    $"The array is not regular: at level {depth} there is a sub-array of length {array.Length}, " +
                    $"but the first has {shape[depth]}. All sub-arrays at the same level must have " +
                    "the same length — otherwise the array has no shape and cannot become a tensor.");
            }

            if (depth + 1 >= shape.Length)
            {
                return;   // the last level holds scalars, not sub-arrays
            }

            for (int i = 0; i < array.Length; i++)
            {
                if (array.GetValue(i) is not Array sub)
                {
                    throw new ArgumentException(
                        $"The array is not regular: at level {depth} element {i} is a scalar, " +
                        "while the first element is a sub-array. All sub-arrays at the same level " +
                        "must have the same length.");
                }

                EnsureRectangular(sub, shape, depth + 1);
            }
        }

        private static void CopyJaggedToMulti(Array jagged, Array multi, int[] indices)
        {
            int dim = indices.Length;

            for (int i = 0; i < jagged.Length; i++)
            {
                int[] newIndices = indices.Concat(new int[] { i }).ToArray();

                if (jagged.GetValue(i) is Array subArray)
                {
                    CopyJaggedToMulti((Array)subArray, multi, newIndices);
                }
                else
                {
                    multi.SetValue(Convert.ToDouble(jagged.GetValue(i)), newIndices);
                }
            }
        }

        // Flattens any array (1D, multidimensional or jagged) into a double[] in row-major order.
        //
        // Previously this was `foreach (var item in array)` over `System.Array` plus a `List<double>`.
        // Both hurt: foreach over an untyped Array yields `object`, so EVERY double was boxed, and
        // the List reallocated as it doubled. A 6 KB payload (784 doubles) cost ~40 KB of
        // allocations, 6.5× overhead — and `Tensor.Data` is called practically all the time.
        //
        // Now: compute the exact length first, allocate one array, and copy along typed paths.
        // For double of any rank, Buffer.BlockCopy is enough.
        private static double[] FlattenArray(Array array)
        {
            double[] result = new double[CountElements(array)];
            int written = 0;
            FlattenInto(array, result, ref written);
            return result;
        }

        // How many scalars the array contains. For jagged arrays the sub-arrays must be summed
        // (they may differ in length); for 1D and multidimensional arrays Length is enough.
        private static int CountElements(Array array)
        {
            if (array.GetType().GetElementType()?.IsArray != true)
            {
                return array.Length;
            }

            int total = 0;
            for (int i = 0; i < array.Length; i++)
            {
                if (array.GetValue(i) is Array sub)
                {
                    total += CountElements(sub);
                }
            }
            return total;
        }

        private static void FlattenInto(Array array, double[] target, ref int written)
        {
            Type? elementType = array.GetType().GetElementType();

            // jagged → recurse into sub-arrays (GetValue returns a reference, no boxing)
            if (elementType?.IsArray == true)
            {
                for (int i = 0; i < array.Length; i++)
                {
                    if (array.GetValue(i) is Array sub)
                    {
                        FlattenInto(sub, target, ref written);
                    }
                }
                return;
            }

            // double of any rank → one block copy, no boxing.
            // Buffer.BlockCopy works on multidimensional primitive arrays too, and the row-major
            // order matches what the rest of the library expects from Data.
            if (elementType == typeof(double))
            {
                Buffer.BlockCopy(array, 0, target, written * sizeof(double), array.Length * sizeof(double));
                written += array.Length;
                return;
            }

            // int/float 1D — typed loop without boxing
            if (array is int[] ints)
            {
                for (int i = 0; i < ints.Length; i++) target[written++] = ints[i];
                return;
            }

            if (array is float[] floats)
            {
                for (int i = 0; i < floats.Length; i++) target[written++] = floats[i];
                return;
            }

            // Everything else (e.g. int[,]) — rare; boxing cannot be avoided here.
            foreach (object? item in array)
            {
                target[written++] = Convert.ToDouble(item);
            }
        }
        private static Array BuildMultiDimArray(double[] data, int[] shape, int offset = 0)
        {
            if (shape.Length == 0)
                throw new ArgumentException("Shape must not be empty.");

            if (shape.Length == 1)
            {
                double[] arr1D = new double[shape[0]];
                Array.Copy(data, offset, arr1D, 0, shape[0]);
                return arr1D;
            }

            Array arr = Array.CreateInstance(typeof(double), shape);

            int subArraySize = 1;
            for (int i = 1; i < shape.Length; i++)
                subArraySize *= shape[i];


            for (int i = 0; i < shape[0]; i++)
            {

                for (int j = 0; j < subArraySize; j++)
                {

                    int[] subIndices = GetMultiDimIndices(j, shape.Skip(1).ToArray());

                    int[] fullIndices = new int[shape.Length];
                    fullIndices[0] = i;
                    for (int k = 0; k < subIndices.Length; k++)
                        fullIndices[k + 1] = subIndices[k];

                    double value = data[offset + i * subArraySize + j];
                    arr.SetValue(value, fullIndices);
                }
            }

            return arr;
        }
        private static int[] GetMultiDimIndices(int linearIndex, int[] shape)
        {
            int rank = shape.Length;
            int[] indices = new int[rank];
            for (int i = rank - 1; i >= 0; i--)
            {
                indices[i] = linearIndex % shape[i];
                linearIndex /= shape[i];
            }
            return indices;
        }

    }
}