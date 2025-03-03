using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace mdnn
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

            // 2) Zavolej slice (GetSubTensor) nebo co potřebuješ
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
            // Získáme tvar jagged pole (např. {2, 3} pro double[][])
            int[] shape = GetJaggedShape(jaggedArray);

            // Vytvoříme prázdné multidimenzionální pole odpovídajícího tvaru
            Array multiArray = Array.CreateInstance(typeof(double), shape);

            // Rekurzivně zkopírujeme hodnoty do multidimenzionálního pole
            CopyJaggedToMulti(jaggedArray, multiArray, new int[0]);

            return multiArray;
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
                    flatList.AddRange(FlattenArray(subArray)); // Rekurzivně rozbalíme
                else
                    flatList.Add(Convert.ToDouble(item)); // Převod na double
            }
            return flatList.ToArray();
        }

        /// <summary>
        /// Vrátí podtensor (slice) z původního multidimenzionálního pole, kde jsou fixovány první fixedIndices.Length dimenze.
        /// Pokud jsou fixovány všechny dimenze (tj. fixedIndices.Length == source.Rank),
        /// vrátí se jednoprvkové pole obsahující skalární hodnotu.
        /// </summary>
        /// <param name="source">Původní multidimenzionální pole (např. double[,,]).</param>
        /// <param name="fixedIndices">Pole fixních indexů pro počáteční dimenze (např. {0,1} nebo {0,0,0}).</param>
        /// <returns>Nové multidimenzionální pole se zbývajícími dimenzemi, nebo jednoprvkové pole při fixaci všech dimenzí.</returns>
        public static Array GetSubTensor(Array source, int[] fixedIndices)
        {
            // Pokud jsou fixovány všechny dimenze, vrátíme skalární hodnotu zabalenou do jednoprvkového pole.
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

        /// <summary>
        /// Rekurzivně naplní cílové pole hodnotami ze zdrojového pole podle kombinace fixních indexů a indexů cílového pole.
        /// </summary>
        /// <param name="source">Původní pole.</param>
        /// <param name="fixedIndices">Fixní indexy.</param>
        /// <param name="target">Cílové pole, do kterého se budou zapisovat hodnoty.</param>
        /// <param name="targetIndices">Pracovní pole indexů pro cílové pole.</param>
        /// <param name="dim">Aktuální dimenze, kterou procházíme.</param>
        private static void FillSubTensor(Array source, int[] fixedIndices, Array target, int[] targetIndices, int dim)
        {
            if (dim == targetIndices.Length)
            {
                // Sestavíme celkový index pro zdrojové pole
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
        /// <summary>
        /// Z flat pole 'data' a daného tvaru 'shape' vyrobí reálné vícerozměrné pole typu double[...,].
        /// </summary>
        public static Array BuildMultiDimArray(double[] data, int[] shape, int offset = 0)
        {
            // 1) Pokud je shape prázdné, nemá smysl
            if (shape.Length == 0)
                throw new ArgumentException("Shape nesmí být prázdné.");

            // 2) Pokud je to 1D
            if (shape.Length == 1)
            {
                double[] arr1D = new double[shape[0]];
                Array.Copy(data, offset, arr1D, 0, shape[0]);
                return arr1D;
            }

            // 3) Jinak vytvoříme reálné nD pole
            Array arr = Array.CreateInstance(typeof(double), shape);

            // subArraySize = součin zbývajících dimenzí
            int subArraySize = 1;
            for (int i = 1; i < shape.Length; i++)
                subArraySize *= shape[i];

            // Pro každý index v 1. dimenzi (shape[0]) vyplníme data
            for (int i = 0; i < shape[0]; i++)
            {
                // Tady naplníme subpole (dimenzí shape[1..end]) element-po-elementu
                for (int j = 0; j < subArraySize; j++)
                {
                    // Převedeme lineární index j na vícerozměrný index subIndices
                    int[] subIndices = GetMultiDimIndices(j, shape.Skip(1).ToArray());

                    // Složíme full index pro arr
                    int[] fullIndices = new int[shape.Length];
                    fullIndices[0] = i;
                    for (int k = 0; k < subIndices.Length; k++)
                        fullIndices[k + 1] = subIndices[k];

                    // Přečteme jednu hodnotu z flat data
                    double value = data[offset + i * subArraySize + j];
                    arr.SetValue(value, fullIndices);
                }
            }

            return arr;
        }

        /// <summary>
        /// Pomocná metoda: lineární index → vícerozměrné indexy pro zadaný shape.
        /// Např. shape = [3,4], linearIndex=5 → (1,1).
        /// </summary>
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