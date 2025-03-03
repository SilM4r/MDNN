
using My_DNN.Layers.classes;
using My_DNN.Loss_functions;
namespace My_DNN
{
    public class Train
    {
        private MDNN model;

        private uint epoch;
        private uint totalEpoch;
        private uint size_of_mini_batch;
        private double lowestLoss;
        private Random rnd = new Random();

        private List<int> listOfepoch = new List<int>();
        private List<double> listOfValidLoss = new List<double>();
        private List<double> listOfTrainLoss = new List<double>();

        public uint Number_of_skip_frist_Epoch_in_plotter = 0;
        public uint Number_Of_Show_Epoch_In_Console = 100;

        private Tensor trainData_inputs;
        private Tensor testData_inputs;
        private Tensor validData_inputs;

        private Tensor trainData_current_output;
        private Tensor testData_current_output;
        private Tensor validData_current_output;
        public Tensor TestData_inputs
        { 
            get => testData_inputs; 
            set => testData_inputs = value; 
        }
        public Tensor ValidData_inputs
        {
            get => validData_inputs;
            set => validData_inputs = value;
        }
        public Tensor TestData_current_output
        {
            get => testData_current_output;
            set => testData_current_output = value;
        }
        public Tensor ValidData_current_output
        {
            get => validData_current_output;
            set => validData_current_output = value;
        }

        public uint Current_epoch
        {
            get { return epoch; }
        }

        public uint Total_epoch
        {
            get { return totalEpoch; }
            set { totalEpoch = value; }
        }

        public uint Mini_batch
        {
            get { return size_of_mini_batch; }
            set { size_of_mini_batch = value; }
        }

        public Train(MDNN model) 
        {
            this.model = model;
            epoch = 0;
            totalEpoch = 0;
            size_of_mini_batch = 1;
        }

        public Train(MDNN model, uint epoch,uint totalEpoch, uint size_of_mini_batch)
        {
            this.model = model;
            this.epoch = epoch;
            this.totalEpoch = totalEpoch;
            this.size_of_mini_batch=size_of_mini_batch;
        }

        public Tensor Fit(Tensor inputs_values, Tensor target_values)
        {
            CheckLayersAreNotEmpty();

            double[] output_target_values = target_values.Data;

            Tensor output = model.FeedForward(inputs_values);

            List<Tensor> de = Gradient.GetGradients(output_target_values, model);

            for (int i = 0; i < model.Layers.Layers.Count(); i++)
            {
                model.Layers.Layers[i].BackPropagation(de[i]);
            }

            return output;
        }

        public async Task<Tensor> FitAsync(Tensor inputs_values, Tensor target_values)
        {
            CheckLayersAreNotEmpty();

            double[] output_target_values = target_values.Data;

            Tensor output = await model.FeedForwardAsync(inputs_values);

            List<Tensor> de = await Gradient.GetGradientsAsync(output_target_values, model);

            Task[] tasks = new Task[model.Layers.Layers.Count()];

            for (int i = 0; i < model.Layers.Layers.Count(); i++)
            {
                int index = i;
                tasks[index] = Task.Run(async() =>
                {
                    await model.Layers.Layers[index].BackPropagationAsync(de[index]);
                });
            }
            await Task.WhenAll(tasks);

            return output;
        }

        public void BackPropagation(double[] target_values)
        {
            CheckLayersAreNotEmpty();

            List<Tensor> de = Gradient.GetGradients(target_values, model);

            for (int i = 0; i < model.Layers.Layers.Count(); i++)
            {
                model.Layers.Layers[i].BackPropagation(de[i]);
            }

            UpdateParams();
        }

        public void UpdateParams()
        {
            CheckLayersAreNotEmpty();

            epoch++;
            foreach (Layer layer in model.Layers.Layers)
            {
                layer.UpdateParams();
            }
        }

        public async Task UpdateParamsAsync()
        {
            CheckLayersAreNotEmpty();

            epoch++;
            int index = -1;
            Task[] tasks = new Task[model.Layers.Layers.Count()];
            foreach (Layer layer in model.Layers.Layers)
            {
                index++;
                tasks[index] = Task.Run(async () =>
                {
                    await layer.UpdateParamsAsync();
                });
            }

            await Task.WhenAll(tasks);
        }

        public void TestNeuralNetwork(Tensor inputs_values, Tensor current_output_values)
        {
            if (model.Layers.Layers[0].Input_size_and_shape[0] == 0)
            {
                if (GeneralNeuralNetworkSettings.SequenceTrain)
                {
                    GeneralNeuralNetworkSettings.modelInputSizeAndShape = inputs_values.GetTensorValue(new int[] { 0,0 }).Shape;
                }

                else
                {
                    GeneralNeuralNetworkSettings.modelInputSizeAndShape = inputs_values.GetTensorValue(new int[] { 0 }).Shape;
                }

                model.Layers.SetInputSizeForFirstLayer();
            }
            CheckLayersAreNotEmpty();

            int score = 0;
            int maxScore = 0;

            if (GeneralNeuralNetworkSettings.SequenceTrain)
            {
                for (int i = 0; i < inputs_values.Shape[0]; i++)
                {
                    model.ResetSequence();
                    for (int j = 0; j < inputs_values.Shape[1]; j++)
                    {
                        Tensor outputTensor = model.FeedForward(inputs_values.GetTensorValue(new int[] { i, j }));

                        double[] output = outputTensor.Data;

                        maxScore += output.Length;

                        bool zeroError = true;

                        // upravit na softMax
                        if (model.Layers.Layers[model.Layers.Layers.Count() - 1].Activation_Func.Apply_to_layer)
                        {
                            double maxOutputValue = output.Max();
                            int maxOutputIndex = output.ToList().IndexOf(maxOutputValue);

                            double maxCurrentOutputValue = ((double[])current_output_values.GetValue(new int[] { i,j })).Max();
                            int maxCurrentOutputIndex = ((double[])current_output_values.GetValue(new int[] { i,j })).ToList().IndexOf(maxCurrentOutputValue);

                            if (maxCurrentOutputIndex != maxOutputIndex)
                            {
                                zeroError = false;
                            }

                        }
                        else
                        {
                            for (int k = 0; k < output.Length; k++)
                            {
                                if (Math.Round(output[k]) != Math.Round(((double[])current_output_values.GetValue(new int[] { i,j }))[k]))
                                {
                                    zeroError = false;
                                    break;
                                }
                            }
                        }


                        if (zeroError)
                        {
                            score++;
                        }
                    }
                }
                model.ResetSequence();
            }

            else
            {
                for (int i = 0; i < inputs_values.Shape[0]; i++)
                {
                    Tensor outputTensor = model.FeedForward(inputs_values.GetTensorValue(new int[] { i }));

                    double[] output = outputTensor.Data;
                    double[] current_output = current_output_values.GetTensorValue(new int[] { i }).Data;

                    bool zeroError = true;
                    maxScore = inputs_values.Shape[0];

                    // upravit na softMax
                    if (model.Layers.Layers[model.Layers.Layers.Count() - 1].Activation_Func.Apply_to_layer)
                    {
                        double maxOutputValue = output.Max();
                        int maxOutputIndex = output.ToList().IndexOf(maxOutputValue);

                        double maxCurrentOutputValue = current_output.Max();
                        int maxCurrentOutputIndex = current_output.ToList().IndexOf(maxCurrentOutputValue);

                        if (maxCurrentOutputIndex != maxOutputIndex)
                        {
                            zeroError = false;
                        }

                    }
                    else
                    {
                        for (int j = 0; j < output.Length; j++)
                        {
                            if (Math.Round(output[j]) != Math.Round(current_output[j]))
                            {
                                zeroError = false;
                                break;
                            }
                        }
                    }


                    if (zeroError)
                    {
                        score++;
                    }
                }
            }
            ConsoleControler.ShowScoreOfmodel(score, maxScore);
        }

        public void TrainLoop(Tensor inputs_values, Tensor current_output_values, uint number_of_epoch, uint size_of_mini_batch = 1, bool isSequence = false)
        {

            PreparationForTrainLoop(inputs_values, current_output_values, number_of_epoch, size_of_mini_batch, isSequence);

            model.info();

            for (uint epoch = this.epoch; epoch < totalEpoch; epoch++)
            {
                GeneralNeuralNetworkSettings.loss_func.ResetAverageLossPerIteration();

                for (uint miniBatch = 0; miniBatch < size_of_mini_batch; miniBatch++)
                {
                    int num = rnd.Next(trainData_inputs.Shape[0]);
                    if (!GeneralNeuralNetworkSettings.SequenceTrain)
                    {
                        Fit(trainData_inputs.GetTensorValue(new int[] { num }), trainData_current_output.GetTensorValue(new int[] { num }));
                    }

                    else
                    {
                        model.ResetSequence();
                        for (int i = 0; i < trainData_inputs.Shape[1]; i++)
                        {
                            Fit(trainData_inputs.GetTensorValue(new int[] { num,i }), trainData_current_output.GetTensorValue(new int[] { num, i }));
                        }
                    }
                }

                if (GeneralNeuralNetworkSettings.SequenceTrain)
                {
                    model.ResetSequence();
                }

                UpdateParams();

                if (epoch % (totalEpoch/ Number_Of_Show_Epoch_In_Console) == 0)
                {
                    TrainLoopControlFunc();
                }
            }

            TestNeuralNetwork(testData_inputs, testData_current_output);

            GraphPlotter.ShowLossGraph(listOfepoch.ToArray(),listOfTrainLoss.ToArray(), listOfValidLoss.ToArray());

        }

        public async Task TrainLoopAsync(Tensor inputs_values, Tensor current_output_values, uint number_of_epoch, uint size_of_mini_batch = 1, bool isSequence = false)
        {
            PreparationForTrainLoop(inputs_values, current_output_values, number_of_epoch, size_of_mini_batch, isSequence);

            model.info();

            for (uint epoch = this.epoch; epoch < totalEpoch; epoch++)
            {
                GeneralNeuralNetworkSettings.loss_func.ResetAverageLossPerIteration();
                for (uint miniBatch = 0; miniBatch < size_of_mini_batch; miniBatch++)
                {
                    int num = rnd.Next(trainData_inputs.Shape[0]);
                    if (!GeneralNeuralNetworkSettings.SequenceTrain)
                    {
                        await FitAsync(trainData_inputs.GetTensorValue(new int[] { num }), trainData_current_output.GetTensorValue(new int[] { num }));
                    }

                    else
                    {
                        model.ResetSequence();
                        for (int i = 0; i < trainData_inputs.Shape[1]; i++)
                        {
                            await FitAsync(trainData_inputs.GetTensorValue(new int[] { num, i }), trainData_current_output.GetTensorValue(new int[] { num, i }));
                        }
                    }
                }

                if (GeneralNeuralNetworkSettings.SequenceTrain)
                {
                    model.ResetSequence();
                }

                await UpdateParamsAsync();

                if (epoch % (totalEpoch / Number_Of_Show_Epoch_In_Console) == 0)
                {
                    await TrainLoopControlFuncAsync();
                }
            }

            TestNeuralNetwork(testData_inputs, testData_current_output);

            GraphPlotter.ShowLossGraph(listOfepoch.ToArray(), listOfTrainLoss.ToArray(), listOfValidLoss.ToArray());

        }

        public void SimpleTrainLoop(double[][] inputs_values, double[][] current_output_values, uint number_of_epoch, uint size_of_mini_batch = 1)
        {
            

            uint epoch = this.epoch;
            double minLoss = 100;

            this.size_of_mini_batch = size_of_mini_batch;
            totalEpoch = number_of_epoch;


            GeneralNeuralNetworkSettings.modelInputSizeAndShape = new int[] { inputs_values[0].Length };
            model.Layers.SetInputSizeForFirstLayer();

            CheckLayersAreNotEmpty();

            for (; epoch < totalEpoch; epoch++)
            {
                for (uint miniBatch = 0; miniBatch < size_of_mini_batch; miniBatch++)
                {
                    int num = rnd.Next(inputs_values.Count());

                    Fit(new Tensor(inputs_values[num]), new Tensor(current_output_values[num]));
                }

                UpdateParams();

                if (epoch % (totalEpoch / Number_Of_Show_Epoch_In_Console) == 0)
                {
                    double loss = GeneralNeuralNetworkSettings.loss_func.GetAverageLossPerIteration();
                    if (loss is not double.NaN)
                    {
                        if (loss < minLoss)
                        {
                            minLoss = loss;
                            model.Note = minLoss.ToString();
                            if (GeneralNeuralNetworkSettings.AutoSaveInTrainLoop)
                            {
                                model.SaveAsJson(GeneralNeuralNetworkSettings.AutoSaveInTrainLoopFileName);
                            }
                            
                        }
                    }
                    else
                    {
                        Console.WriteLine("Error: Nan number");
                        return;
                    }

                    ConsoleControler.ShowEpochInfo(model);
                }
            }
        }

        private void PreparationForTrainLoop(Tensor inputs_values, Tensor current_output_values, uint number_of_epoch, uint size_of_mini_batch = 1, bool isSequence = false)
        {
            CheckTensorShapes(inputs_values, current_output_values);

            ShuffleTensor(inputs_values, current_output_values, out inputs_values, out current_output_values);

            if (isSequence)
            {
                GeneralNeuralNetworkSettings.SequenceTrain = true;
            }

            if (GeneralNeuralNetworkSettings.SequenceTrain)
            {
                if (inputs_values.Shape.Length == 1)
                {
                    throw new Exception("for sequential training of the model, the inputs must be at least in a two-dimensional array");
                }

                if (inputs_values.Shape.Length == 2)
                {
                    inputs_values.Reshape(new int[] { inputs_values.Shape[0], inputs_values.Shape[1], 1 });
                    current_output_values.Reshape(new int[] { inputs_values.Shape[0], inputs_values.Shape[1], 1 });
                }

                else if (inputs_values.Shape.Length >= 5)
                {
                    throw new Exception("for sequential training, the maximum input is a four-dimensional array.");
                }

                GeneralNeuralNetworkSettings.modelInputSizeAndShape = inputs_values.GetTensorValue(new int[] { 0, 0 }).Shape;
            }

            else
            {

                if (inputs_values.Shape.Length == 1)
                {
                    inputs_values.Reshape(new int[] { inputs_values.Shape[0], 1 });
                    current_output_values.Reshape(new int[] { inputs_values.Shape[0], 1 });
                }

                else if (inputs_values.Shape.Length >= 4)
                {
                    throw new Exception("for non-sequential training, the maximum input is a three-dimensional array.");
                }

                GeneralNeuralNetworkSettings.modelInputSizeAndShape = inputs_values.GetTensorValue(new int[] { 0 }).Shape;
            }

            

            if (model.Layers.Layers[0].Input_size_and_shape[0] == 0)
            {
                model.Layers.SetInputSizeForFirstLayer();
            }

            CheckLayersAreNotEmpty();

            DividingDataIntoDatasets(inputs_values, current_output_values);

            listOfepoch = new List<int>();
            listOfValidLoss = new List<double>();
            listOfTrainLoss = new List<double>();

            this.size_of_mini_batch = size_of_mini_batch;
            totalEpoch = number_of_epoch;

            lowestLoss = double.MaxValue;

            if (Number_Of_Show_Epoch_In_Console > totalEpoch)
            {
                Number_Of_Show_Epoch_In_Console = totalEpoch;
            }
        }

        private void CheckLayersAreNotEmpty()
        {
            if (model.Layers.Layers.Count() == 0)
            {
                throw new Exception("Only model with layers can be trained, please add at least one Layer using Layer.add() function.");
            }

            if (model.Layers.Layers[0].Input_size_and_shape[0] <= 0)
            {
                throw new Exception("the input layer must always have at least one input (use the SetInputSizeForFirstLayer() method in the layer class to set the input, for example model.layer.SetInputSizeForFirstLayer(new unit[] {1}))");
            }
        }

        private void CheckTensorShapes(Tensor A, Tensor B)
        {
            if(A == null || B == null)
            {
                throw new Exception("it is not possible to train with empty inputs (Tensor), inputs_values or current_output_values is null");
            }

            
            /*
            if(B.Shape.Length != A.Shape.Length)
            {
                throw new Exception("both inputs_values and current_output_values must have the same number of dimension array");
            }
            */

            //int size = A.Shape.Length;

            /*
            for(int i = 0; i < (size-1); i++)
            {
                if (B.Shape[i] != A.Shape[i])
                {
                    throw new Exception("both inputs_values and current_output_values must have the same array dimensions");
                }
            }
            */

            if (B.Shape[0] != A.Shape[0])
            {
                throw new Exception("both inputs_values and current_output_values must have the same first array dimensions");
            }


        }

        public void DividingDataIntoDatasets(Tensor inputs_values, Tensor current_output_values)
        {
            int TrainData_size;
            int ValidData_size;
            int TestData_size;

            int index = -1;
            int index2 = -1;

            int totalSize = inputs_values.Shape[0];

            if (validData_inputs == null && testData_inputs == null)
            {
                TrainData_size = (int)(totalSize * 0.7);
                ValidData_size = (int)(totalSize * 0.15);
                TestData_size = totalSize - TrainData_size - ValidData_size;

                validData_inputs = inputs_values.Slice(TrainData_size, ValidData_size);
                testData_inputs = inputs_values.Slice(TrainData_size + ValidData_size, TestData_size);

                validData_current_output = current_output_values.Slice(TrainData_size, ValidData_size);
                testData_current_output = current_output_values.Slice(TrainData_size + ValidData_size, TestData_size);

                trainData_inputs = inputs_values.Slice(0, TrainData_size);
                trainData_current_output = current_output_values.Slice(0, TrainData_size);
            }
            else if (validData_inputs != null && testData_inputs == null)
            {
                int oldValidSize = validData_inputs.Shape[0];

                ValidData_size = (int)(oldValidSize * 0.8);
                TestData_size = oldValidSize - ValidData_size;

                trainData_inputs = inputs_values;
                trainData_current_output = current_output_values;

                validData_inputs = validData_inputs.Slice(0, ValidData_size);
                validData_current_output = validData_current_output.Slice(0, ValidData_size);

                testData_inputs = validData_inputs.Slice(ValidData_size, TestData_size);
                testData_current_output = validData_current_output.Slice(ValidData_size, TestData_size);
            }
            else if (validData_inputs == null && testData_inputs != null)
            {
                TrainData_size = (int)(totalSize * 0.8);
                ValidData_size = totalSize - TrainData_size;

                trainData_inputs = inputs_values.Slice(0, TrainData_size);
                trainData_current_output = current_output_values.Slice(0, TrainData_size);

                validData_inputs = inputs_values.Slice(TrainData_size, ValidData_size);
                validData_current_output = current_output_values.Slice(TrainData_size, ValidData_size);
            }
            else
            {
                trainData_inputs = inputs_values;
                trainData_current_output = current_output_values;
            }
        }
      

        public void ShuffleTensor(Tensor tensorA, Tensor tensorB, out Tensor shuffledA, out Tensor shuffledB)
        {
            if (tensorA.Shape[0] != tensorB.Shape[0])
                throw new ArgumentException("Oba tensory musí mít stejný počet vzorků");

            int batchSize = tensorA.Shape[0]; // První dimenze určuje počet vzorků
            int[] indices = Enumerable.Range(0, batchSize).OrderBy(_ => Random.Shared.Next()).ToArray();

            // Vytvoříme nové pole pro zamíchaná data
            double[] shuffledDataA = new double[tensorA.Data.Length];
            double[] shuffledDataB = new double[tensorB.Data.Length];

            int sampleSizeA = tensorA.Data.Length / batchSize; // Velikost jednoho vzorku
            int sampleSizeB = tensorB.Data.Length / batchSize;

            for (int i = 0; i < batchSize; i++)
            {
                int srcIndex = indices[i];

                Array.Copy(tensorA.Data, srcIndex * sampleSizeA, shuffledDataA, i * sampleSizeA, sampleSizeA);
                Array.Copy(tensorB.Data, srcIndex * sampleSizeB, shuffledDataB, i * sampleSizeB, sampleSizeB);
            }

            // Vytvoříme nové tensory se stejným tvarem
            shuffledA = new Tensor(shuffledDataA, tensorA.Shape);
            shuffledB = new Tensor(shuffledDataB, tensorB.Shape);
        }
        private void TrainLoopControlFunc()
        {
            Loss lossFunc = GeneralNeuralNetworkSettings.loss_func;
            double loss = lossFunc.GetResetAverageLossPerIteration();
            if (GeneralNeuralNetworkSettings.SequenceTrain)
            {
                for (int i = 0; i < ValidData_inputs.Shape[0]; i++)
                {
                    model.ResetSequence();
                    for (int j = 0; j < ValidData_inputs.Shape[1]; j++)
                    {
                        model.FeedForward(ValidData_inputs.GetTensorValue(new int[] { i, j }));
                        lossFunc.CalculateLoss(model.Layers.Layers[model.Layers.Layers.Count() - 1].Layer_output.Data, ValidData_current_output.GetTensorValue(new int[] { i, j }).Data);
                    }
                }
            }

            else
            {
                for (int i = 0; i < ValidData_inputs.Shape[0]; i++)
                {
                    model.FeedForward(ValidData_inputs.GetTensorValue(new int[] { i }));
                    lossFunc.CalculateLoss(model.Layers.Layers[model.Layers.Layers.Count() - 1].Layer_output.Data, ValidData_current_output.GetTensorValue(new int[] { i}).Data);
                }
            }

            ExtraControlFunc(loss);
            ConsoleControler.ShowEpochInfo(model, loss);
        }

        private async Task TrainLoopControlFuncAsync()
        {
            Loss lossFunc = GeneralNeuralNetworkSettings.loss_func;
            List<double> layerOutput = new List<double>();

            double loss = lossFunc.GetResetAverageLossPerIteration();
            if (GeneralNeuralNetworkSettings.SequenceTrain)
            {
                for (int i = 0; i < ValidData_inputs.Shape[0]; i++)
                {
                    model.ResetSequence();
                    for (int j = 0; j < ValidData_inputs.Shape[1]; j++)
                    {
                        Tensor output = await model.FeedForwardAsync(ValidData_inputs.GetTensorValue(new int[] { i, j }));
                        lossFunc.CalculateLoss(output.Data, ValidData_current_output.GetTensorValue(new int[] { i, j }).Data);
                    }
                }
            }

            else
            {
                Task[] tasks = new Task[ValidData_inputs.Shape[0]];
                
                for (int i = 0; i < ValidData_inputs.Shape[0]; i++)
                {
                    int index = i;
                    tasks[index] = Task.Run(async () =>
                    {
                        Tensor output = await model.FeedForwardAsync(ValidData_inputs.GetTensorValue(new int[] { index }));
                        double lossOutput = lossFunc.CalculateAndGetLoss(output.Data, ValidData_current_output.GetTensorValue(new int[] { index }).Data);

                        if (lossOutput is not double.NaN)
                        {
                            layerOutput.Add(lossOutput);
                        }
                    });

                }
                await Task.WhenAll(tasks);

                lossFunc.ResetAverageLossPerIteration();
            }

            

            ExtraControlFunc(loss);
            if (GeneralNeuralNetworkSettings.SequenceTrain)
            {
                ConsoleControler.ShowEpochInfo(model, loss);
            }
            else
            {
                ConsoleControler.ShowEpochInfo(model, loss, layerOutput.Sum() / layerOutput.Count());
            }
                
        }

        private void ExtraControlFunc(double loss)
        {
            Loss lossFunc = GeneralNeuralNetworkSettings.loss_func;

            if (loss is double.NaN || lossFunc.GetAverageLossPerIteration() is double.NaN)
            {
                GraphPlotter.ShowLossGraph(listOfepoch.ToArray(), listOfTrainLoss.ToArray(), listOfValidLoss.ToArray());
                ConsoleControler.ErrorHandler("NaN value in output", "The output from the neural network is either too small or too large, hence the value of nan. Please try other values ​​in the training parameters (for example: learning rate or hyperammetry )", true);
                return;
            }

            if (epoch >= ((totalEpoch / Number_Of_Show_Epoch_In_Console) * Number_of_skip_frist_Epoch_in_plotter))
            {
                listOfepoch.Add((int)epoch);
                listOfTrainLoss.Add(loss);
                listOfValidLoss.Add(lossFunc.GetAverageLossPerIteration());
            }

            double validLoss = lossFunc.GetAverageLossPerIteration();

            if (GeneralNeuralNetworkSettings.AutoSaveInTrainLoop && (validLoss < lowestLoss))
            {
                lowestLoss = validLoss;
                model.SaveAsJson(GeneralNeuralNetworkSettings.AutoSaveInTrainLoopFileName);
            }
        }

    }
}
