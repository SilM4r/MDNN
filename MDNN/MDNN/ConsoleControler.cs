using My_DNN.Layers;
using My_DNN.Layers.classes;
using My_DNN.Loss_functions;
using ScottPlot.Finance;
using static System.Formats.Asn1.AsnWriter;
using static SkiaSharp.HarfBuzz.SKShaper;


namespace My_DNN
{
    public static class ConsoleControler
    {
        private static DateTime time;

        private static uint lastEpochInfo = 0;

        public static void ShowModelInfo(MDNN model)
        {
            int totalParams = 0;

            if (model.Layers.Layers.Count() > 0)
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

            int number_of_layer = 0;
            bool isEmptyInput = false;

            foreach (Layer layer in model.Layers.Layers)
            {

                string shapeString;
                number_of_layer++;
                switch (layer)
                {
                    case Dense:
                        totalParams += ((Dense)layer).Neurons.Count * ((Dense)layer).Input_size_and_shape[0] + ((Dense)layer).Neurons.Count;
                        if (((Dense)layer).Input_size_and_shape[0] == 0)
                        {
                            shapeString = "unknow";
                            isEmptyInput = true;
                        }
                        else
                        {
                            shapeString = ((Dense)layer).Input_size_and_shape[0].ToString();
                        }
                        Console.WriteLine("--------------------------------------------------------------------------------------------------------------------");
                        Console.WriteLine($"| {number_of_layer}. Name: {layer.Name} | Number of neurons: {((Dense)layer).Neurons.Count} | Number of inputs: {shapeString} | Activation func: {((Dense)layer).Activation_Func.Name} |");
                        Console.WriteLine("--------------------------------------------------------------------------------------------------------------------");
                        break;
                    case RNN:
                        totalParams += ((RNN)layer).Neurons.Count * (((RNN)layer).Input_size_and_shape[0] + 1) + ((RNN)layer).Neurons.Count;
                        if (((RNN)layer).Input_size_and_shape[0] == 0)
                        {
                            shapeString = "unknow";
                            isEmptyInput = true;
                        }
                        else
                        {
                            shapeString = ((RNN)layer).Input_size_and_shape[0].ToString();
                        }
                        Console.WriteLine("--------------------------------------------------------------------------------------------------------------------");
                        Console.WriteLine($"| {number_of_layer}. Name: {layer.Name} | Number of neurons: {((RNN)layer).Neurons.Count} | Number of inputs: {((RNN)layer).Input_size_and_shape[0]} (+ {((RNN)layer).Input_size_and_shape[0]} Recurrents inputs) | Activation func: {((RNN)layer).Activation_Func.Name} |");
                        Console.WriteLine("--------------------------------------------------------------------------------------------------------------------");
                        break;
                    case Conv:
                        totalParams += (((Conv)layer).Kernel[0].Count() * ((Conv)layer).Kernel[0][0].Count() * ((Conv)layer).Kernel[0][0][0].Count()) + ((Conv)layer).Biases.Length;

                        if (((Conv)layer).Output_size_and_shape.Length == 1)
                        {
                            shapeString = "unknow";
                            isEmptyInput = true;
                        }
                        else
                        {
                            shapeString = $"[{((Conv)layer).Output_size_and_shape[0]},{((Conv)layer).Output_size_and_shape[1]},{((Conv)layer).Output_size_and_shape[2]}]";
                        }

                        Console.WriteLine("--------------------------------------------------------------------------------------------------------------------");
                        Console.WriteLine($"| {number_of_layer}. Name: {layer.Name} | Number of kernels: {((Conv)layer).Kernel.Count()} | Kernel size:{((Conv)layer).Kernel[0].Count()}*{((Conv)layer).Kernel[0][0].Count()} | Output shape: {shapeString} | Activation func: {((Conv)layer).Activation_Func.Name} |");
                        Console.WriteLine("--------------------------------------------------------------------------------------------------------------------");
                        break;
                    case MaxPool:


                        if (((MaxPool)layer).Output_size_and_shape.Length == 1)
                        {
                            shapeString = "unknow";
                            isEmptyInput = true;
                        }
                        else
                        {
                            shapeString = $"[{((MaxPool)layer).Output_size_and_shape[0]},{((MaxPool)layer).Output_size_and_shape[1]},{((MaxPool)layer).Output_size_and_shape[2]}]";
                        }

                        Console.WriteLine("--------------------------------------------------------------------------------------------------------------------");
                        Console.WriteLine($"| {number_of_layer}. Name: {layer.Name} | PoolSize: {((MaxPool)layer).PoolSize} * {((MaxPool)layer).PoolSize} | stride: {((MaxPool)layer).PoolSize} | Output shape: {shapeString}  ");
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
                Console.WriteLine($"Total trainable params: {totalParams}");
                Console.WriteLine($"Current epoch: {model.Train.Current_epoch}");
                Console.WriteLine($"Target epoch: {model.Train.Total_epoch}");
                Console.WriteLine($"Size of mini batch: {model.Train.Mini_batch}");
            }

            Console.WriteLine();
        }


        public static void ShowEpochInfo(MDNN model,double? TrainLoss = null, double? ValidLoss = null)
        {
            DateTime timeNow = DateTime.Now;

            TimeSpan subTime; 
            TimeSpan estimatedCompletionTime;

            uint pastEpochs = model.Train.Current_epoch - lastEpochInfo;

            lastEpochInfo = model.Train.Current_epoch;

            if (time == DateTime.MinValue) 
            {
                subTime = TimeSpan.Zero;
                estimatedCompletionTime = TimeSpan.Zero;
            }
            else
            {
                subTime = timeNow - time;
                estimatedCompletionTime = (subTime * (model.Train.Total_epoch - model.Train.Current_epoch)) / pastEpochs;
            }
            Console.WriteLine("########################################################################");
            Console.WriteLine($"Epoch: {model.Train.Current_epoch} / {model.Train.Total_epoch} {(model.Train.Current_epoch / (float)model.Train.Total_epoch) * 100} %");
            Console.WriteLine();
            if(TrainLoss == null )
            {
                Console.WriteLine($"Loss:{model.Loss.GetResetAverageLossPerIteration()}");
            }
            else if (ValidLoss == null)
            {
                Console.WriteLine($"Valid Loss:{model.Loss.GetResetAverageLossPerIteration()}");
                Console.WriteLine($"Train Loss:{TrainLoss}");
            }
            else
            {
                Console.WriteLine($"Valid Loss:{ValidLoss}");
                Console.WriteLine($"Train Loss:{TrainLoss}");
            }

            Console.WriteLine();
            Console.WriteLine($"Time: {subTime}");
            Console.WriteLine($"Estimate of completion time: {estimatedCompletionTime}");
            Console.WriteLine($"Estimated end of learning at: {DateTime.Now + estimatedCompletionTime}");
            Console.WriteLine();
            

            time = DateTime.Now;
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


        public static void ErrorHandler(string name, string description, bool fatalError = false)
        {
            Console.WriteLine();
            if (fatalError)
            {
                Console.WriteLine("----------------------  !Fatal Error!  ---------------------- ");
            }
            else
            {
                Console.WriteLine("----------------------   Error  ---------------------- ");
            }
            Console.WriteLine($"Name: {name}");
            Console.WriteLine($"Description: {description} ");

            if (fatalError)
            {
                Environment.Exit(0);
            }
        }

        private static string writeSpace(int x, int y)
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
