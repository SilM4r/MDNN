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
            // Multidim podobu si vyžádá jen ten, kdo o ni stojí (Train v sekvenčním režimu
            // ji přetypovává na double[]). Data se vyříznou plochou cestou jako u
            // GetTensorValue, teprve pak se z nich případně poskládá pole.
            return GetTensorValue(indices).OriginalInput;
        }

        // Výřez podtenzoru podle prvních `indices.Length` dimenzí.
        //
        // Dřív se to počítalo přes GetSubTensor: Array.CreateInstance + rekurzivní
        // FillSubTensor, který kopíroval prvek po prvku pomocí GetValue/SetValue — tedy
        // reflexe a boxing na každý double. Změřeno: 48 KB alokací na JEDEN výřez vzorku.
        // A `TrainLoop` tohle volá na každý vzorek každé epochy: u MNIST (49k vzorků,
        // 20 epoch) to je ~1M volání, tedy desítky GB jen za krájení.
        //
        // Data jsou plochá v row-major pořadí, takže výřez je souvislý blok — stačí spočítat
        // offset a délku a jednou zkopírovat.
        public Tensor GetTensorValue(int[] indices)
        {
            if (indices.Length >= Shape.Length)
                throw new ArgumentException("Počet indexů nesmí být větší než dimenze tenzoru.");

            // délka bloku = součin zbývajících dimenzí
            int blockSize = 1;
            for (int d = indices.Length; d < Shape.Length; d++)
            {
                blockSize *= Shape[d];
            }

            // offset = suma index[d] * (součin dimenzí napravo od d)
            int offset = 0;
            int stride = blockSize;
            for (int d = indices.Length - 1; d >= 0; d--)
            {
                if (indices[d] < 0 || indices[d] >= Shape[d])
                {
                    throw new ArgumentOutOfRangeException(nameof(indices),
                        $"Index {indices[d]} je mimo rozsah dimenze {d} (velikost {Shape[d]}).");
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

        // Zploští [h][w][c] jagged pole rovnou do Tensoru s daným tvarem.
        //
        // Conv i MaxPool dřív vracely `new Tensor(ConvertJaggedToMulti(output))`, což na
        // KAŽDÝ forward postavilo přes reflexi (Array.CreateInstance + SetValue po prvcích,
        // tedy boxing na každý double) kopii celého výstupu. Změřeno: 8 MB alokací na jeden
        // vzorek CNN. Tady se jen jednou projde a zapíše do plochého pole.
        public static Tensor FromJagged3D(double[][][] source, int[] shape)
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

            // Rozbalit VŠECHNY úrovně jagged pole, ne jen jednu. Dřív se odloupla jedna,
            // takže `double[][]` prošlo, ale `double[][][]` skončilo na hlášce
            // „invalid type on input array" — a to je přesně tvar sekvenčních dat
            // ([sekvence][krok][rys]), tedy případ, kvůli kterému sekvenční trénink existuje.
            while (elementType != null && elementType.IsArray)
            {
                elementType = elementType.GetElementType();
                isjagged = true;
            }

            // Porovnání na typ, ne na název: .NET jména jsou "Int32"/"Single"/"Double",
            // takže dřívější test na "Int"/"Float" nikdy neprošel a int i float vstupy
            // knihovna odmítala — přestože je vlastní chybová hláška slibuje.
            if (elementType == typeof(double))
            {
                return isjagged ? new Tensor(ConvertJaggedToMulti(array)) : new Tensor(array);
            }

            if (elementType == typeof(int) || elementType == typeof(float))
            {
                // int/float převedeme na double: tvar + plochá data (vnitřně počítáme v double,
                // a Conv/MaxPool si přetypovávají GetOriginalData() na double[,,]).
                int[] shape = isjagged
                    ? GetJaggedShape(array)
                    : Enumerable.Range(0, array.Rank).Select(array.GetLength).ToArray();

                return new Tensor(FlattenArray(array), shape);
            }

            throw new Exception("invalid type on input array, it can be only `int`,`double`,`float`");
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

        // Zploští libovolné pole (1D, multidim i jagged) do double[] v row-major pořadí.
        //
        // Dřív to bylo `foreach (var item in array)` nad `System.Array` plus `List<double>`.
        // Obojí bolelo: foreach nad netypovaným Array vrací `object`, takže se KAŽDÝ double
        // zaboxoval, a List se zdvojnásobováním realokoval. Payload 6 KB (784 doublů) stál
        // ~40 KB alokací, tedy 6,5x navíc — a `Tensor.Data` se volá prakticky pořád.
        //
        // Nově: napřed se spočítá přesná délka, alokuje se jedno pole, a kopíruje se
        // typovanými cestami. Pro double libovolného ranku stačí Buffer.BlockCopy.
        private static double[] FlattenArray(Array array)
        {
            double[] result = new double[CountElements(array)];
            int written = 0;
            FlattenInto(array, result, ref written);
            return result;
        }

        // Kolik skalárů pole obsahuje. U jagged se musí sečíst podpole (můžou být různě
        // dlouhá), u 1D i multidim stačí Length.
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

            // jagged → rekurze přes podpole (GetValue vrací referenci, neboxuje se)
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

            // double libovolného ranku → jedno blokové kopírování, žádný boxing.
            // Buffer.BlockCopy funguje i na multidim polích primitiv a row-major pořadí
            // odpovídá tomu, co od Data očekává zbytek knihovny.
            if (elementType == typeof(double))
            {
                Buffer.BlockCopy(array, 0, target, written * sizeof(double), array.Length * sizeof(double));
                written += array.Length;
                return;
            }

            // int/float 1D — typovaná smyčka bez boxingu
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

            // Zbytek (např. int[,]) — vzácné, tady se boxingu nevyhneme.
            foreach (object? item in array)
            {
                target[written++] = Convert.ToDouble(item);
            }
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