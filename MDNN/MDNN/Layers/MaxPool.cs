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

            // vstupní/výstupní tvar se dopočítá v LayerAdjustment (při připojení k modelu)
            inputsShape = new int[] { 0 };
            outputShape = new int[] { 0 };
            indexes = new PoolingIndex[0, 0, 0];
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

        // ---------------------------------------------------------------- konvence B

        public override Tensor Backward(Tensor gradOutput, bool preActivation = false,
                                        bool computeInputGradient = true)
        {
            if (preActivation)
                throw new NotSupportedException(
                    "MaxPool nemá aktivaci, takže preActivation nedává smysl. " +
                    "Fúze softmax+CE se týká jen výstupní Dense vrstvy.");

            if (output == null || indexes == null)
                throw new InvalidOperationException(
                    "MaxPool: backward před forwardem — pozice maxim ještě neexistují.");

            // Gradient přichází rovnou po prvcích, žádné sahání do vah souseda. Právě tím
            // odpadá celá větev „další vrstva je Dense" ze staré CalculateLayerGradients:
            // Dense si teď svůj ∂L/∂input spočítá sám a pošle ho sem hotový.
            double[] upstream = gradOutput.Data;

            int outHeight = output.Length;
            int outWidth = output[0].Length;
            int channels = output[0][0].Length;

            if (upstream.Length != outHeight * outWidth * channels)
                throw new ArgumentException(
                    $"MaxPool má výstup {outHeight}×{outWidth}×{channels} " +
                    $"({outHeight * outWidth * channels} hodnot), ale gradient shora " +
                    $"jich má {upstream.Length}.");

            if (!computeInputGradient) return Tensor.Empty;

            int inputHeight = inputs!.Length;
            int inputWidth = inputs[0].Length;

            double[][][] dInput = new double[inputHeight][][];
            for (int i = 0; i < inputHeight; i++)
            {
                dInput[i] = new double[inputWidth][];
                for (int j = 0; j < inputWidth; j++) dInput[i][j] = new double[channels];
            }

            // Gradient teče JEN do pozice, která v okně vyhrála; ostatní na výstup
            // nepřispěly, takže dostávají nulu.
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

        public override Tensor FeedForward(Tensor TensorValues)
        {
            // Stejná stráž jako u Conv: tvar musí sedět na ten zapojený. Dřív menší vstup
            // přepsal jen roh bufferu a zbytek zůstal z minulého vzorku — MaxPool pak
            // z okna samých nul vrátil maximum z předchozího obrázku. Rank ≥ 4 tu dřív
            // padal na NotSupportedException() bez jediného slova v hlášce.
            // Tvar jen pro tuhle metodu, tenzor volajícího zůstane nedotčený — viz Conv.FeedForward.
            int[] shape = SpatialShape.RequireWiredShape(TensorValues.Shape, inputsShape, Name);

            // plochá cesta místo (double[,,])GetOriginalData() — viz Conv.FeedForward
            double[] inputValues = TensorValues.Data;

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
            return Tensor.FromJagged3D(output, outputShape);
        }

        public override void WireShapes(int? number_of_elements = null, int[]? input_Shape = null)
        {
            // Stejná pojistka jako u Conv: tvar vstupu ještě není známý (placeholder z Insert/Add).
            // Bez ní se počítalo (0 - poolSize)/poolSize + 1 → záporný rozměr → pád při alokaci.
            if (input_Shape != null && input_Shape.Any(d => d <= 0))
            {
                inputsShape = new int[] { 0 };
                outputShape = new int[] { 0 };
                return;
            }

            if (number_of_elements != null)
            {
                poolSize = (int)number_of_elements;

                int outHeight = (inputsShape[0] - poolSize) / poolSize + 1;
                int outWidth = (inputsShape[1] - poolSize) / poolSize + 1;

                outputShape = new int[] { outHeight, outWidth, inputsShape[2] };
            }

            if (input_Shape != null)
            {
                // tentýž normalizátor jako ve FeedForward — viz SpatialShape
                inputsShape = SpatialShape.Normalize(input_Shape, Name);

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

        // MaxPool nemá trénovatelné parametry — není co materializovat ani co ztratit.
        public override void MaterializeParameters() { }

        public override bool IsMaterialized => true;
    }
}
