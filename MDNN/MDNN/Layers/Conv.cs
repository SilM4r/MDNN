using mdnn.Activation_functions.classes;
using mdnn.Layers.classes;
using mdnn.Optimizers;
using mdnn.Save_neural_network;

namespace mdnn.Layers
{
    public class Conv : Layer
    {
        public override string Name => "CNN";

        public override int[] Input_size_and_shape => inputsShape;

        public override int[] Output_size_and_shape => outputShape;

        public override Tensor Layer_output => new Tensor(output);

        public override Tensor Layer_raw_output => new Tensor(raw_output);

        public override Activation_func Activation_Func
        {
            get { return activation_func; }
            set 
            {
                activation_func = value;
                CheckIsActivationFuncIsNotApplyToLayer();
            }
        }

        public double[][][][] Kernel  // [numFilters][kernelRows][kernelCols][inputChannels]
        {
            get { return kernels; }
        }
        public double[] Biases
        {
            get { return biases; }
        }
        public string Padding
        {
            get { return padding; }
        }

        private double[][][][] kernels;
        private double[][][][] dKernels;
        private double[] biases;
        private double[] dBiases;
        private double[][][] dOutput;
        private string padding;
        private double[][][] output;
        private double[][][] raw_output;
        private double[][][] inputs;

        private int[] inputsShape;
        private int[] outputShape;

        private int mini_batch_size;
        private Optimizer optimizer;

        private Activation_func activation_func;

        public Conv(ExportCNNLayer layer)
        {
            kernels = layer.Kernel;
            biases = layer.Biases;
            activation_func = Activation_func.inicialization_activation_func(layer.Name_of_activation_function);
            CheckIsActivationFuncIsNotApplyToLayer();
            padding = layer.Padding;

            optimizer = Optimizer.Clone_optimizer(GeneralNeuralNetworkSettings.optimizer);

            inputsShape = layer.InputShape;

            if (padding.ToLower() == "same")
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

        public Conv(int number_of_kernels,int kernel_size, Activation_func? activation_func = null, string padding = "valid")
        {
            if (activation_func == null)
            {
                if (LayerManager.number_of_penultimate_output_in_Layer[0] == 0)
                {
                    activation_func = GeneralNeuralNetworkSettings.output_activation_func;
                }
                else
                {
                    activation_func = GeneralNeuralNetworkSettings.hidden_layers_activation_func;
                }
            }

            inputsShape = new int[] { 0 };
            outputShape = new int[] { 0 };

            this.activation_func = activation_func;
            CheckIsActivationFuncIsNotApplyToLayer();
            int inputChannels;

            if (LayerManager.number_of_penultimate_output_in_Layer[0] == 0)
            {
                inputChannels = 0;
            }
            else if (LayerManager.number_of_penultimate_output_in_Layer.Length == 1 || LayerManager.number_of_penultimate_output_in_Layer.Length == 2)
            {
                inputChannels = 1;
            }
            else
            {
                inputChannels = LayerManager.number_of_penultimate_output_in_Layer[2];
            }

            this.padding = padding;

            inicializationK_B_dK_dB(number_of_kernels,new int[] { kernel_size , kernel_size , inputChannels },true);

            optimizer = Optimizer.Clone_optimizer(GeneralNeuralNetworkSettings.optimizer);
            mini_batch_size = 0;

        }

        public override void LayerAdjustment(int? number_of_kernels = null, int[]? input_Shape = null)
        {
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
                    inputsShape = new int[] {rows,cols,1 };
                }
                else if (input_Shape.Length == 3)
                {
                    inputsShape = input_Shape;
                }
                else
                {
                    throw new Exception();
                }

                if (padding.ToLower() == "same")
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


            if (number_of_kernels == null)
            {
                number_of_kernels = Kernel.Length;
            }
            inicializationK_B_dK_dB(number_of_kernels, new int[] { Kernel[0].Count(), Kernel[0][0].Count(), inputsShape[2] }, true);
        }

        public override Tensor FeedForward(Tensor TensorValues)
        {
            if (TensorValues.Shape.Length == 2)
            {
                TensorValues.Reshape(new int[] { TensorValues.Shape[0], TensorValues.Shape[0], 1 });
            }

            else if (TensorValues.Shape.Length == 1)
            {
                int rows, cols;
                ConvertTo2D(TensorValues.Data, out rows, out cols);
                TensorValues.Reshape(new int[] { rows, cols, 1 });
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

            int inputRows = inputs.Length;
            int inputCols = inputs[0].Length;
            int inputChannels = inputs[0][0].Length;
            int numFilters = kernels.Length;
            int kernelRows = kernels[0].Length;
            int kernelCols = kernels[0][0].Length;
            int kernelChannels = kernels[0][0][0].Length;

            if (kernelChannels != inputChannels)
            {
                throw new ArgumentException("Počet kanálů vstupu a kernelu se musí shodovat.");
            }

            double[][][] paddedInput;
            if (padding.ToLower() == "same")
            {
                int padRows = kernelRows / 2;
                int padCols = kernelCols / 2;
                paddedInput = Pad3D(inputs, padRows, padCols);
            }
            else
            {
                paddedInput = inputs;
            }
            

            // Pro každý filtr (výstupní kanál)
            for (int f = 0; f < numFilters; f++)
            {
                for (int i = 0; i < outputShape[0]; i++)
                {
                    for (int j = 0; j < outputShape[1]; j++)
                    {
                        double sum = 0;
                        for (int ki = 0; ki < kernelRows; ki++)
                        {
                            for (int kj = 0; kj < kernelCols; kj++)
                            {
                                for (int c = 0; c < inputChannels; c++)
                                {
                                    sum += paddedInput[i + ki][j + kj][c] * kernels[f][ki][kj][c];
                                }
                            }
                        }
                        // Přičteme bias pro filtr f
                        double result = sum + biases[f];
                        output[i][j][f] = activation_func.Apply(result);
                        raw_output[i][j][f] = result;
                    }
                }
            }

            return new Tensor(Tensor.ConvertJaggedToMulti(output));
        }
        public override void BackPropagation(Tensor TensordOutput)
        {
            mini_batch_size++;

            int inChannels = inputs[0][0].Length;

            // Rozměry kernelů
            int numFilters = kernels.Length;
            int kH = kernels[0].Length;
            int kW = kernels[0][0].Length;

            // Padding – pokud je režim "same", padujeme vstup stejně jako ve forward propagaci
            double[][][] paddedInput;
            if (padding.ToLower() == "same")
            {
                paddedInput = Pad3D(inputs,kH/2,kW/2);
            }
            else
            {
                paddedInput = inputs;
            }

            // 1. Výpočet dBiases: pro každý filtr se sečtou všechny hodnoty z dOutput
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

            // 2. Výpočet dKernels: pro každý filtr a každý prvek kernelu se vynásobí odpovídající hodnota z paddedInput a dOutput
            for (int f = 0; f < numFilters; f++)
            {
                for (int ki = 0; ki < kH; ki++)
                {
                    for (int kj = 0; kj < kW; kj++)
                    {
                        for (int c = 0; c < inChannels; c++)
                        {
                            double sum = 0;
                            for (int i = 0; i < outputShape[0]; i++)
                            {
                                for (int j = 0; j < outputShape[1]; j++)
                                {
                                    sum += paddedInput[i + ki][j + kj][c] * dOutput[i][j][f];
                                }
                            }
                            dKernels[f][ki][kj][c] += sum;
                        }
                    }
                }
            }
        }

        public override void UpdateParams()
        {
            int numFilters = kernels.Length;
            for (int f = 0; f < numFilters; f++)
            {
                int kernelHeight = kernels[f].Length;
                int kernelWidth = kernels[f][0].Length;
                int inChannels = kernels[f][0][0].Length;
                // Aktualizace vah pro daný filtr
                for (int i = 0; i < kernelHeight; i++)
                {
                    for (int j = 0; j < kernelWidth; j++)
                    {
                        for (int c = 0; c < inChannels; c++)
                        {
                            kernels[f][i][j][c] = optimizer.Update(kernels[f][i][j][c], dKernels[f][i][j][c] / mini_batch_size);
                        }
                    }
                }
                // Aktualizace biasu pro daný filtr
                biases[f] = optimizer.Update(biases[f], dBiases[f] / mini_batch_size);
            }
            mini_batch_size = 0;
            inicializationK_B_dK_dB();
        }

        public override Tensor CalculateLayerGradients(Tensor TensordOutput, Layer next_layer)
        {
            double[][][] dInput;

            if (TensordOutput.Shape.Length == 1)
            {
                double[] next_layer_e = TensordOutput.Data;
                double de = 0;
                

                LayerBasedOnNeurons? nextlayer = next_layer as LayerBasedOnNeurons;

                double[] e = new double[output.Count()* output[0].Count()* output[0][0].Count()];

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
                            for (int k = 0;k < output[0][0].Count(); k++)
                            {
                                dOutput[i][j][k] = e[i * (output[0][0].Count() * output[0].Count()) + j * output[0][0].Count() + k] * activation_func.Derivative(raw_output[i][j][k]);
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

                for (int i = 0;i < tensorDOutput.GetLength(0);i++)
                {
                    for (int j = 0; j < tensorDOutput.GetLength(1); j++)
                    {
                        for (int k = 0; k < tensorDOutput.GetLength(2); k++)
                        {
                            dOutput[i][j][k] = tensorDOutput[i, j, k] * activation_func.Derivative(raw_output[i][j][k]);
                        }
                    }
                }
            }

            // Rozměry vstupu
            int H = inputs.Length;
            int W = inputs[0].Length;
            int inChannels = inputs[0][0].Length;

            // Rozměry kernelů
            int numFilters = kernels.Length;
            int kH = kernels[0].Length;
            int kW = kernels[0][0].Length;

            // Padding – pokud je režim "same", padujeme vstup stejně jako ve forward propagaci
            int padRows = 0, padCols = 0;
            double[][][] paddedInput;
            if (padding.ToLower() == "same")
            {
                padRows = kH / 2;
                padCols = kW / 2;
                paddedInput = Pad3D(inputs, padRows, padCols);
            }
            else
            {
                paddedInput = inputs;
            }

            int paddedH = paddedInput.Length;
            int paddedW = paddedInput[0].Length;

            // Očekávané rozměry výstupu (dOutput)
            int outH = dOutput.Length;
            int outW = dOutput[0].Length;


            // Inicializace gradientů:
            // dInput_padded má stejný tvar jako paddedInput: [paddedH][paddedW][inChannels]
            double[][][] dInputPadded = new double[paddedH][][];
            for (int i = 0; i < paddedH; i++)
            {
                dInputPadded[i] = new double[paddedW][];
                for (int j = 0; j < paddedW; j++)
                {
                    dInputPadded[i][j] = new double[inChannels];
                }
            }

            // 3. Výpočet dInput (vypočítáme gradient vůči padded vstupu)
            // Pro každý výstupní bod se gradient "rozprostře" zpět na vstupní pixely pomocí hodnot kernelu
            for (int f = 0; f < numFilters; f++)
            {
                for (int i = 0; i < outH; i++)
                {
                    for (int j = 0; j < outW; j++)
                    {
                        for (int ki = 0; ki < kH; ki++)
                        {
                            for (int kj = 0; kj < kW; kj++)
                            {
                                for (int c = 0; c < inChannels; c++)
                                {
                                    dInputPadded[i + ki][j + kj][c] += kernels[f][ki][kj][c] * dOutput[i][j][f];
                                }
                            }
                        }
                    }
                }
            }

            // Pokud jsme používali "same", odstraníme padding, abychom získali dInput se stejnými rozměry jako původní inputs
            if (padding.ToLower() == "same")
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
                            dInput[i][j][c] = dInputPadded[i + padRows][j + padCols][c];
                        }
                    }
                }
            }
            else
            {
                // V režimu "valid" má dInputPadded rozměry odpovídající inputs
                dInput = dInputPadded;
            }

            return new Tensor(Tensor.ConvertJaggedToMulti(dInput));
        }

        private void CheckIsActivationFuncIsNotApplyToLayer()
        {
            if (activation_func.Apply_to_layer)
            {
                throw new ArgumentException("unfortunately it is not possible to apply an activation function to a convulsion layer that is applied to the whole layer like softmax() instead use for example ReLu() or Tanh().");
            }
        }
        private double[][][] Pad3D(double[][][] input, int padRows, int padCols)
        {
            int originalRows = input.Length;
            int originalCols = input[0].Length;
            int channels = input[0][0].Length;
            int newRows = originalRows + 2 * padRows;
            int newCols = originalCols + 2 * padCols;

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
                        padded[i + padRows][j + padCols][c] = input[i][j][c];
                    }
                }
            }

            return padded;
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
                                kernels[l][i][j][k] = GeneralNeuralNetworkSettings.rnd.NextDouble() / (k + 1) ;
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
