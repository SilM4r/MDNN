using My_DNN.Layers;
using My_DNN.Layers.classes;


namespace My_DNN
{
    public static class ConsoleControler
    {
        public static void ShowModelInfo(MDNN model)
        {
            if (model.Layers.Layers.Count != 0)
            {
                Console.WriteLine($"Number of input : {string.Join(", ", model.Layers.Layers[0].Input_size_and_shape) }");
                Console.WriteLine($"Number of output: {string.Join(", ", model.Layers.Layers[model.Layers.Layers.Count() - 1].Output_size_and_shape)}");
                Console.WriteLine($"Number of hidden layers: {model.Layers.Layers.Count() - 1}");
                Console.WriteLine($"Schema layers: {model.Schema}");
            }

            else
            {
                Console.WriteLine($"Number of input : 0");
                Console.WriteLine($"Number of output: 0");
                Console.WriteLine($"Number of hidden layers: 0");
                Console.WriteLine($"Schema layers: []");
            }

            
            
            Console.WriteLine($"Loss function: {model.Loss.Name}");
            Console.WriteLine($"Optimizer function: {model.Optimizer.Name}");

            Console.WriteLine("--------------------------------------------------------------------------------------------------------------------");
            Console.WriteLine("|                                          | Schema layers |                                                       |");
            Console.WriteLine("--------------------------------------------------------------------------------------------------------------------");

            int numberOfLayer = 0;
            bool isEmptyInput = false;

            foreach (Layer layer in model.Layers.Layers)
            {

                string shapeString;
                numberOfLayer++;
                switch (layer)
                {
                    case Dense dense:
                        if (dense.Input_size_and_shape[0] == 0)
                        {
                            shapeString = "unknow";
                            isEmptyInput = true;
                        }
                        else
                        {
                            shapeString = dense.Input_size_and_shape[0].ToString();
                        }
                        Console.WriteLine("--------------------------------------------------------------------------------------------------------------------");
                        Console.WriteLine($"| {numberOfLayer}. Name: {dense.Name} | Number of neurons: {dense.Neurons.Count} | Number of inputs: {shapeString} | Activation func: {dense.Activation_Func.Name} |");
                        Console.WriteLine("--------------------------------------------------------------------------------------------------------------------");
                        break;
                    case RNN rnn:
                        if (rnn.Input_size_and_shape[0] == 0)
                        {
                            shapeString = "unknow";
                            isEmptyInput = true;
                        }
                        else
                        {
                            shapeString = rnn.Input_size_and_shape[0].ToString();
                        }
                        Console.WriteLine("--------------------------------------------------------------------------------------------------------------------");
                        Console.WriteLine($"| {numberOfLayer}. Name: {rnn.Name} | Number of neurons: {rnn.Neurons.Count} | Number of inputs: {rnn.Input_size_and_shape[0]} (+ {rnn.Input_size_and_shape[0]} Recurrents inputs) | Activation func: {rnn.Activation_Func.Name} |");
                        Console.WriteLine("--------------------------------------------------------------------------------------------------------------------");
                        break;
                    case Conv conv:
                        if (conv.Output_size_and_shape.Length == 1)
                        {
                            shapeString = "unknow";
                            isEmptyInput = true;
                        }
                        else
                        {
                            shapeString = $"[{conv.Output_size_and_shape[0]},{conv.Output_size_and_shape[1]},{conv.Output_size_and_shape[2]}]";
                        }

                        Console.WriteLine("--------------------------------------------------------------------------------------------------------------------");
                        Console.WriteLine($"| {numberOfLayer}. Name: {conv.Name} | Number of kernels: {conv.Kernel.Count()} | Kernel size:{conv.Kernel[0].Count()}*{conv.Kernel[0][0].Count()} | Output shape: {shapeString} | Activation func: {conv.Activation_Func.Name} |");
                        Console.WriteLine("--------------------------------------------------------------------------------------------------------------------");
                        break;
                    case MaxPool pool:


                        if (pool.Output_size_and_shape.Length == 1)
                        {
                            shapeString = "unknow";
                            isEmptyInput = true;
                        }
                        else
                        {
                            shapeString = $"[{pool.Output_size_and_shape[0]},{pool.Output_size_and_shape[1]},{pool.Output_size_and_shape[2]}]";
                        }

                        Console.WriteLine("--------------------------------------------------------------------------------------------------------------------");
                        Console.WriteLine($"| {numberOfLayer}. Name: {pool.Name} | PoolSize: {pool.PoolSize} * {pool.PoolSize} | stride: {pool.PoolSize} | Output shape: {shapeString}  ");
                        Console.WriteLine("--------------------------------------------------------------------------------------------------------------------");
                        break;
                    default:
                        Console.WriteLine("--------------------------------------------------------------------------------------------------------------------");
                        Console.WriteLine("| Tato vrstva není definována |");
                        Console.WriteLine("--------------------------------------------------------------------------------------------------------------------");
                        break;
                }
            }
            if (isEmptyInput)
            {
                Console.WriteLine($"! Warning !");
                Console.WriteLine("These are indicative model data only!");
                Console.WriteLine("This information is before the input values ​​are set, so some data may be missing from the model. ");
                Console.WriteLine("As soon as you start training the model, the missing data will appear by itself, if not manually set the input values ​​using GeneralNeuralNetworkSettings.modelInputSizeAndShape");
            }   
            else
            {
                Console.WriteLine($"Total trainable params: {CountTrainableParams(model)}");
                Console.WriteLine($"Current epoch: {model.Train.CurrentEpoch}");
                Console.WriteLine($"Target epoch: {model.Train.TotalEpoch}");
                Console.WriteLine($"Size of mini batch: {model.Train.MiniBatch}");
            }

            Console.WriteLine();
        }

        // Počet trénovatelných parametrů modelu.
        // Conv = numFilters × (kH × kW × inChannels) + biasy; dřív chybělo × numFilters
        // (Kernel.Length) → conv se drasticky podpočítával. MaxPool a spol. = 0 parametrů.
        public static int CountTrainableParams(MDNN model)
        {
            int total = 0;
            foreach (Layer layer in model.Layers.Layers)
            {
                switch (layer)
                {
                    case Dense dense:
                        total += dense.Neurons.Count * dense.Input_size_and_shape[0] + dense.Neurons.Count;
                        break;
                    case RNN rnn:
                        total += rnn.Neurons.Count * (rnn.Input_size_and_shape[0] + 1) + rnn.Neurons.Count;
                        break;
                    case Conv conv:
                        total += conv.Kernel.Length
                                 * (conv.Kernel[0].Count() * conv.Kernel[0][0].Count() * conv.Kernel[0][0][0].Count())
                                 + conv.Biases.Length;
                        break;
                }
            }
            return total;
        }

        public static void ShowEpochInfo(MDNN model,double? trainLoss = null, double? validLoss = null, double? validAccuracy = null)
        {
            DateTime timeNow = DateTime.Now;

            TimeSpan subTime;
            TimeSpan estimatedCompletionTime;

            // Stav je per-Train instance (dřív statiky) → druhý trénink v procesu už nepočítá
            // z časů toho prvního. pastEpochs == 0 se navíc explicitně ošetří: dělení nulou
            // dávalo TimeSpan z Infinity ticků → OverflowException.
            uint pastEpochs = model.Train.CurrentEpoch >= model.Train.LastReportEpoch
                ? model.Train.CurrentEpoch - model.Train.LastReportEpoch
                : 0;

            model.Train.LastReportEpoch = model.Train.CurrentEpoch;

            uint remainingEpochs = model.Train.TotalEpoch > model.Train.CurrentEpoch
                ? model.Train.TotalEpoch - model.Train.CurrentEpoch
                : 0;

            if (model.Train.LastReportTime == DateTime.MinValue || pastEpochs == 0)
            {
                subTime = TimeSpan.Zero;
                estimatedCompletionTime = TimeSpan.Zero;
            }
            else
            {
                subTime = timeNow - model.Train.LastReportTime;
                estimatedCompletionTime = (subTime * remainingEpochs) / pastEpochs;
            }
            Console.WriteLine("########################################################################");
            Console.WriteLine($"Epoch: {model.Train.CurrentEpoch} / {model.Train.TotalEpoch} {(model.Train.CurrentEpoch / (float)model.Train.TotalEpoch) * 100} %");
            Console.WriteLine();
            if(trainLoss == null )
            {
                Console.WriteLine($"Loss:{model.Loss.GetResetAverageLossPerIteration()}");
            }
            else if (validLoss == null)
            {
                Console.WriteLine($"Train Loss:{trainLoss}");
                Console.WriteLine($"Valid Loss:{model.Loss.GetResetAverageLossPerIteration()}");
            }
            else
            {
                Console.WriteLine($"Train Loss:{trainLoss}");
                Console.WriteLine($"Valid Loss:{validLoss}");
                
            }

            if (validAccuracy != null)
            {
                Console.WriteLine($"Valid accuracy: {validAccuracy:0.00} %");
            }

            // nejlepší valid loss zatím + epocha, kde padl (vynech, dokud žádný report neproběhl)
            if (model.Train.LowestValidLoss != double.MaxValue)
            {
                Console.WriteLine($"Best valid loss: {model.Train.LowestValidLoss} @ epoch {model.Train.BestEpoch}");
            }

            Console.WriteLine();
            Console.WriteLine($"Time: {subTime}");
            Console.WriteLine($"Estimate of completion time: {estimatedCompletionTime}");
            Console.WriteLine($"Estimated end of learning at: {DateTime.Now + estimatedCompletionTime}");
            Console.WriteLine();


            model.Train.LastReportTime = DateTime.Now;
        }

        public static void ShowScoreOfmodel(int score,int maxScore)
        {
            Console.WriteLine();
            Console.WriteLine("------------- Model Score ------------- ");
            Console.WriteLine($"Precision: {(float)((float)(score) / (float)(maxScore)) * 100}%");
            Console.WriteLine($"Total number of samples tested: {maxScore}");
            Console.WriteLine($"Number of correctly guessed tested samples: {score}");
            Console.WriteLine($"Number of mistyped samples tested: {maxScore - score}");
            Console.WriteLine();
        }
        
        public static void ShowComplexScoreOfmodel((int,int) test, (int,int) train,(int,int) valid)
        {
            Console.WriteLine();
            Console.WriteLine("------------- Model Score ------------- ");
            Console.WriteLine($"Precision on test dataset: {(float)((float)(test.Item1) / (float)(test.Item2)) * 100}%");
            Console.WriteLine($"Total number of samples tested on test dataset: {test.Item2}");
            Console.WriteLine($"Number of correctly guessed tested samples on test dataset: {test.Item1}");
            Console.WriteLine($"Number of mistyped samples tested on test dataset: {test.Item2 - test.Item1}");
            Console.WriteLine();
            Console.WriteLine($"Precision on train dataset: {(float)((float)(train.Item1) / (float)(train.Item2)) * 100}%");
            Console.WriteLine($"Precision on valid dataset: {(float)((float)(valid.Item1) / (float)(valid.Item2)) * 100}%");
            Console.WriteLine();
        }

        public static void ShowEarlyStopping(uint epoch, uint patience, double bestLoss, uint bestEpoch)
        {
            Console.WriteLine();
            Console.WriteLine("------------- Early stopping ------------- ");
            Console.WriteLine($"Training stopped at epoch {epoch}: {patience} valid reports without improvement.");
            Console.WriteLine($"Best valid loss: {bestLoss} @ epoch {bestEpoch}");
            Console.WriteLine();
        }


        public static void ErrorHandler(string name, string description, bool fatalError = false)
        {
            Console.WriteLine();
            Console.WriteLine(fatalError
                ? "----------------------  !Fatal Error!  ---------------------- "
                : "----------------------   Error  ---------------------- ");
            Console.WriteLine($"Name: {name}");
            Console.WriteLine($"Description: {description} ");

            if (fatalError)
            {
                Environment.Exit(0);
            }
        }

        private static string WriteSpace(int x, int y)
        {
            string res = "";
            for (int j = x; j < y; j++)
            {
                res += " ";
            }

            return res;
        }
    }
}
