using My_DNN.Layers.classes;
using My_DNN.Save_neural_network;
namespace My_DNN.Layers
{
    public class MaxPool : LayerWithUntrainedParameters
    {
        public override string Name => "MaxPool";

        public override int[] Input_size_and_shape => inputsShape;

        public override int[] Output_size_and_shape => outputShape;

        public override Tensor? Layer_output
        {
            get
            {
                if (output == null)
                {
                    return null;
                }
                else
                {
                    return new Tensor(output);
                }
            }
        }
        public override Tensor? Layer_raw_output
        {
            get
            {
                if (output == null)
                {
                    return null;
                }
                else
                {
                    return new Tensor(output);
                }
            }
        }

        public int PoolSize
        {
            get { return poolSize; }
        }

        private double[][][]? output;
        private double[][][]? dOutput;
        private double[][][]? inputs;

        private PoolingIndex[,,] indexes;

        private int[] inputsShape;
        private int[] outputShape;
        private int poolSize;

        public MaxPool(int poolSize)
        {
            this.poolSize = poolSize;

            if (LayerManager.number_of_penultimate_output_in_Layer[0] == -1)
            {
                inputsShape = new int[] { 0 };
                outputShape = new int[] { 0 };
                indexes = new PoolingIndex[0,0,0];
            }
            else
            {
                inputsShape = LayerManager.number_of_penultimate_output_in_Layer;

                if (inputsShape.Length == 1)
                {
                    inputsShape = new int[] {0,0,0 };
                }

                int outHeight = (inputsShape[0] - poolSize) / poolSize + 1;
                int outWidth = (inputsShape[1] - poolSize) / poolSize + 1;

                outputShape = new int[] { outHeight, outWidth, inputsShape[2] };
                indexes = new PoolingIndex[outputShape[0], outputShape[1], outputShape[2]];
            }

            
        }
        public MaxPool(ExportMaxPoolLayer layer)
        {
            inputsShape = layer.InputsShape ;
            poolSize = (int)layer.PoolSize;

            int outHeight = (inputsShape[0] - poolSize) / poolSize + 1;
            int outWidth = (inputsShape[1] - poolSize) / poolSize + 1;

            outputShape = new int[] { outHeight, outWidth, inputsShape[2] };
            indexes = new PoolingIndex[outputShape[0], outputShape[1], outputShape[2]];

            inputs = new double[inputsShape[0]][][];
            for (int i = 0; i < inputsShape[0]; i++)
            {
                inputs[i] = new double[inputsShape[1]][];
                for (int j = 0; j < inputsShape[1]; j++)
                {
                    inputs[i][j] = new double[inputsShape[2]];
                }
            }

            indexes = new PoolingIndex[outputShape[0], outputShape[1], outputShape[2]];

            output = new double[outputShape[0]][][];
            dOutput = new double[outputShape[0]][][];
            for (int i = 0; i < outputShape[0]; i++)
            {
                output[i] = new double[outputShape[1]][];
                dOutput[i] = new double[outputShape[1]][];
                for (int j = 0; j < outputShape[1]; j++)
                {
                    output[i][j] = new double[outputShape[2]];
                    dOutput[i][j] = new double[outputShape[2]];
                }
            }
        }

        public override Tensor CalculateLayerGradients(Tensor TensordOutput, Layer next_layer)
        {


            if (TensordOutput.Shape.Length == 1)
            {
                double[] next_layer_e = TensordOutput.Data;
                double de = 0;


                LayerBasedOnNeurons? nextlayer = next_layer as LayerBasedOnNeurons;

                double[] e = new double[output.Count() * output[0].Count() * output[0][0].Count()];

                if (nextlayer != null)
                {

                    for (int j = 0; j < (output.Count() * output[0].Count() * output[0][0].Count()); j++)
                    {
                        for (int k = 0; k < nextlayer.Neurons.Count(); k++)
                        {
                            de += next_layer_e[k] * nextlayer.Neurons[k].Weights[j];
                        }

                        e[j] = de;
                        de = 0;
                    }
                    for (int i = 0; i < output.Count(); i++)
                    {
                        for (int j = 0; j < output[0].Count(); j++)
                        {
                            for (int k = 0; k < output[0][0].Count(); k++)
                            {
                                dOutput[i][j][k] = e[i * (output[0][0].Count() * output[0].Count()) + j * output[0][0].Count() + k];
                            }

                        }
                    }
                }
                else
                {
                    throw new NotImplementedException();
                }
            }
            else
            {
                double[,,] tensorDOutput = (double[,,])TensordOutput.GetOriginalData();

                for (int i = 0; i < tensorDOutput.GetLength(0); i++)
                {
                    for (int j = 0; j < tensorDOutput.GetLength(1); j++)
                    {
                        for (int k = 0; k < tensorDOutput.GetLength(2); k++)
                        {
                            dOutput[i][j][k] = tensorDOutput[i, j, k];
                        }
                    }
                }
            }


            int inputHeight = inputs.Length;
            int inputWidth = inputs[0].Length;
            int channels = inputs[0][0].Length;


            // Inicializace dInput nulami
            double[][][] dInput = new double[inputHeight][][];
            for (int i = 0; i < inputHeight; i++)
            {
                dInput[i] = new double[inputWidth][];
                for (int j = 0; j < inputWidth; j++)
                {
                    dInput[i][j] = new double[channels];
                }
            }

            int outHeight = dOutput.Length;
            int outWidth = dOutput[0].Length;

            // Pro každý pooling region:
            for (int i = 0; i < outHeight; i++)
            {
                for (int j = 0; j < outWidth; j++)
                {
                    for (int c = 0; c < channels; c++)
                    {
                        // Načteme uložený index maximální hodnoty
                        PoolingIndex idx = indexes[i, j, c];
                        // Propagujeme gradient – pokud se pooling okna nepřekrývají, můžeme jednoduše přiřadit.
                        dInput[idx.Row][idx.Col][c] += dOutput[i][j][c];
                    }
                }
            }

            return new Tensor(Tensor.ConvertJaggedToMulti(dInput));
        }

        public override Tensor FeedForward(Tensor TensorValues)
        {
            if (TensorValues.Shape.Length == 1)
            {
                int rows, cols;
                ConvertTo2D(TensorValues.Data, out rows, out cols);
                TensorValues.Reshape(new int[] { rows, cols, 1 });
            }
            else if (TensorValues.Shape.Length == 2)
            {
                TensorValues.Reshape(new int[] { TensorValues.Shape[0], TensorValues.Shape[0], 1 });
            }
            else if (TensorValues.Shape.Length >= 4)
            {
                throw new NotSupportedException();
            }

            double[,,] inputs_val = (double[,,])TensorValues.GetOriginalData();

            for (int i = 0; i < TensorValues.Shape[0]; i++)
            {
                for (int j = 0; j < TensorValues.Shape[1]; j++)
                {
                    for (int k = 0; k < TensorValues.Shape[2]; k++)
                    {
                        inputs[i][j][k] = inputs_val[i, j, k];
                    }
                }
            }

            int inputHeight = inputs.Length;
            int inputWidth = inputs[0].Length;
            int channels = inputs[0][0].Length;

            // Výpočet rozměrů výstupu – pro valid pooling (bez paddingu)
            int outHeight = (inputHeight - poolSize) / poolSize + 1;
            int outWidth = (inputWidth - poolSize) / poolSize + 1;

            // Pro každý kanál, každý pooling region, najdeme maximální hodnotu
            for (int c = 0; c < channels; c++)
            {
                for (int i = 0; i < outHeight; i++)
                {
                    for (int j = 0; j < outWidth; j++)
                    {
                        double maxVal = double.NegativeInfinity;
                        int maxRow = -1;
                        int maxCol = -1;
                        // Iterace přes pooling okno
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
            return new Tensor(Tensor.ConvertJaggedToMulti(output));
        }

        public override void LayerAdjustment(int? number_of_elements = null, int[]? input_Shape = null)
        {
            if (number_of_elements != null)
            {
                poolSize = (int)number_of_elements;

                int outHeight = (inputsShape[0] - poolSize) / poolSize + 1;
                int outWidth = (inputsShape[1] - poolSize) / poolSize + 1;

                outputShape = new int[] { outHeight, outWidth, inputsShape[2] };
            }

            if (input_Shape != null)
            {
                if (input_Shape.Length == 1)
                {
                    double[] n = new double[input_Shape[0]];

                    for (int i = 0; i < input_Shape[0]; i++)
                    {
                        n[i] = 1;
                    }

                    int rows, cols;
                    ConvertTo2D(n, out rows, out cols);
                    inputsShape = new int[] { rows, cols, 1 };
                }
                else if (input_Shape.Length == 3)
                {
                    inputsShape = input_Shape;
                }
                else
                {
                    throw new Exception();
                }


                int outHeight = (inputsShape[0] - poolSize) / poolSize + 1;
                int outWidth = (inputsShape[1] - poolSize) / poolSize + 1;

                outputShape = new int[] { outHeight, outWidth, inputsShape[2] };

                inputs = new double[inputsShape[0]][][];
                for (int i = 0; i < inputsShape[0]; i++)
                {
                    inputs[i] = new double[inputsShape[1]][];
                    for (int j = 0; j < inputsShape[1]; j++)
                    {
                        inputs[i][j] = new double[inputsShape[2]];
                    }
                }
            }

            indexes = new PoolingIndex[outputShape[0], outputShape[1], outputShape[2]];

            output = new double[outputShape[0]][][];
            dOutput = new double[outputShape[0]][][];
            for (int i = 0; i < outputShape[0]; i++)
            {
                output[i] = new double[outputShape[1]][];
                dOutput[i] = new double[outputShape[1]][];
                for (int j = 0; j < outputShape[1]; j++)
                {
                    output[i][j] = new double[outputShape[2]];
                    dOutput[i][j] = new double[outputShape[2]];
                }
            }
        }

        private void ConvertTo2D(double[] array, out int rows, out int cols)
        {
            int n = array.Length;

            rows = (int)Math.Ceiling(Math.Sqrt(n));

            while (n % rows != 0)
            {
                rows++;
            }

            cols = n / rows;
        }
    }
}
