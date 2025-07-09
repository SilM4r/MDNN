namespace My_DNN
{
    public class Tensor
    {
        public int[] Shape { get; private set; }
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

        public Array OriginalInput
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
            private set { _originalInput = value; }
        }

        private Array? _originalInput = null;
        private double[]? _data = null;

        public Tensor(Array input)
        {
            OriginalInput = input;
            Shape = Enumerable.Range(0, input.Rank)
                              .Select(input.GetLength)
                              .ToArray();
        }

        public Tensor(double[] Data, int[] Shape)
        {
            this.Shape = Shape;
            _data = Data;
        }
        public void Reshape(int[] newShape)
        {
            int newSize = newShape.Aggregate(1, (a, b) => a * b);

            if (newSize != Data.Length)
                throw new ArgumentException("Nový tvar musí mít stejný počet prvků jako původní!");

            Shape = newShape;
            _originalInput = null;
        }

        public object GetValue(int[] indices)
        {
            if (indices.Length >= Shape.Length)
                throw new ArgumentException("Počet indexů nesmí být větší než dimenze tenzoru.");

            return GetSubTensor(OriginalInput, indices);
        }
        public Tensor GetTensorValue(int[] indices)
        {
            if (indices.Length >= Shape.Length)
                throw new ArgumentException("Počet indexů nesmí být větší než dimenze tenzoru.");

            return new Tensor(GetSubTensor(OriginalInput, indices));
        }

        public Array GetOriginalData()
        {
            return OriginalInput;
        }

        public Array ReshapeToArray(double[] data, int[] shape)
        {
            return BuildMultiDimArray(data, shape, 0);
        }

        public Tensor Slice(int start, int length)
        {
            if (start < 0 || length <= 0 || start + length > Shape[0])
                throw new ArgumentException("Invalid slice range!");


            int sampleSize = Data.Length / Shape[0]; // Velikost jednoho vzorku (dle první dimenze)
            double[] slicedData = new double[length * sampleSize];

            Array.Copy(Data, start * sampleSize, slicedData, 0, length * sampleSize);

            int[] newShape = (int[])Shape.Clone();
            newShape[0] = length; // První dimenze je teď zkrácená

            return new Tensor(slicedData, newShape);
        }

        public static Array ConvertJaggedToMulti(Array jaggedArray)
        {
            int[] shape = GetJaggedShape(jaggedArray);

            // Vytvoříme prázdné multidimenzionální pole odpovídajícího tvaru
            Array multiArray = Array.CreateInstance(typeof(double), shape);

            // Rekurzivně zkopírujeme hodnoty do multidimenzionálního pole
            CopyJaggedToMulti(jaggedArray, multiArray, new int[0]);

            return multiArray;
        }

        public static Tensor ConvertArrayToTensor(Array array)
        {
            Type type = array.GetType();
            Type? elementType = type.GetElementType();

            bool isjagged = false;

            if (elementType == null)
            {
                throw new Exception("input array is null");
            }

            if (elementType.IsArray)
            {
                elementType = elementType.GetElementType();
                isjagged = true;
            }

            if (elementType?.Name == "Int" || elementType?.Name == "Float" || elementType?.Name == "Double")
            {
                if (isjagged)
                {
                    return new Tensor(ConvertJaggedToMulti(array));
                }
                else
                {
                    return new Tensor(array);
                }
                
            }
            else
            {
                throw new Exception("invalid type on input array, it can be only `int`,`double`,`float`");
            }
        }

        private static int[] GetJaggedShape(Array array)
        {
            List<int> shape = new List<int>();

            while (array is Array firstDim && firstDim.Length > 0)
            {
                shape.Add(firstDim.Length);
                array = firstDim.GetValue(0) as Array;
            }

            return shape.ToArray();
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

        private double[] FlattenArray(Array array)
        {
            List<double> flatList = new List<double>();
            foreach (var item in array)
            {
                if (item is Array subArray)
                    flatList.AddRange(FlattenArray(subArray)); 
                else
                    flatList.Add(Convert.ToDouble(item)); 
            }
            return flatList.ToArray();
        }
        public static Array GetSubTensor(Array source, int[] fixedIndices)
        {

            if (fixedIndices.Length == source.Rank)
            {
                object value = source.GetValue(fixedIndices);
                Type elementType = source.GetType().GetElementType();
                Array result = Array.CreateInstance(elementType, 1);
                result.SetValue(value, 0);
                return result;
            }
            else
            {
                int fixedCount = fixedIndices.Length;
                int remainingRank = source.Rank - fixedCount;
                int[] newShape = new int[remainingRank];
                for (int i = 0; i < remainingRank; i++)
                {
                    newShape[i] = source.GetLength(fixedCount + i);
                }
                Type elemType = source.GetType().GetElementType();
                Array result = Array.CreateInstance(elemType, newShape);
                FillSubTensor(source, fixedIndices, result, new int[remainingRank], 0);
                return result;
            }
        }
        private static void FillSubTensor(Array source, int[] fixedIndices, Array target, int[] targetIndices, int dim)
        {
            if (dim == targetIndices.Length)
            {

                int totalLength = fixedIndices.Length + targetIndices.Length;
                int[] fullIndices = new int[totalLength];
                for (int i = 0; i < fixedIndices.Length; i++)
                    fullIndices[i] = fixedIndices[i];
                for (int i = 0; i < targetIndices.Length; i++)
                    fullIndices[fixedIndices.Length + i] = targetIndices[i];

                target.SetValue(source.GetValue(fullIndices), targetIndices);
            }
            else
            {
                for (int i = 0; i < target.GetLength(dim); i++)
                {
                    targetIndices[dim] = i;
                    FillSubTensor(source, fixedIndices, target, targetIndices, dim + 1);
                }
            }
        }

        public static Array BuildMultiDimArray(double[] data, int[] shape, int offset = 0)
        {
            if (shape.Length == 0)
                throw new ArgumentException("Shape nesmí být prázdné.");

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