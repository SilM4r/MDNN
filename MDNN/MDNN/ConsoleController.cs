using MDNN.Layers;


namespace MDNN
{
    /// <summary>
    /// Console output of the library: model summary, training progress, scores and errors.
    /// </summary>
    public static class ConsoleController
    {
        /// <summary>Prints the model summary: inputs, outputs, loss, optimizer and a table of layers.</summary>
        public static void ShowModelInfo(NeuralNetwork model)
        {
            if (model.Layers.Layers.Count != 0)
            {
                Console.WriteLine($"Number of input : {string.Join(", ", model.Layers.Layers[0].InputShape) }");
                Console.WriteLine($"Number of output: {string.Join(", ", model.Layers.Layers[model.Layers.Layers.Count() - 1].OutputShape)}");
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
                        if (dense.InputShape[0] == 0)
                        {
                            shapeString = "unknown";
                            isEmptyInput = true;
                        }
                        else
                        {
                            shapeString = dense.InputShape[0].ToString();
                        }
                        Console.WriteLine("--------------------------------------------------------------------------------------------------------------------");
                        Console.WriteLine($"| {numberOfLayer}. Name: {dense.Name} | Number of neurons: {dense.Neurons.Count} | Number of inputs: {shapeString} | Activation func: {dense.Activation.Name} |");
                        Console.WriteLine("--------------------------------------------------------------------------------------------------------------------");
                        break;
                    case RNN rnn:
                        if (rnn.InputShape[0] == 0)
                        {
                            shapeString = "unknown";
                            isEmptyInput = true;
                        }
                        else
                        {
                            shapeString = rnn.InputShape[0].ToString();
                        }
                        Console.WriteLine("--------------------------------------------------------------------------------------------------------------------");
                        // The recurrence is diagonal: each neuron gets ONE extra input, its own
                        // previous output — not the whole layer (see README, RTRL).
                        Console.WriteLine($"| {numberOfLayer}. Name: {rnn.Name} | Number of neurons: {rnn.Neurons.Count} | Number of inputs: {shapeString} (+ 1 recurrent input per neuron) | Activation func: {rnn.Activation.Name} |");
                        Console.WriteLine("--------------------------------------------------------------------------------------------------------------------");
                        break;
                    case Conv conv:
                        if (conv.OutputShape.Length == 1)
                        {
                            shapeString = "unknown";
                            isEmptyInput = true;
                        }
                        else
                        {
                            shapeString = $"[{conv.OutputShape[0]},{conv.OutputShape[1]},{conv.OutputShape[2]}]";
                        }

                        Console.WriteLine("--------------------------------------------------------------------------------------------------------------------");
                        Console.WriteLine($"| {numberOfLayer}. Name: {conv.Name} | Number of kernels: {conv.Kernels.Count()} | Kernel size:{conv.Kernels[0].Count()}*{conv.Kernels[0][0].Count()} | Output shape: {shapeString} | Activation func: {conv.Activation.Name} |");
                        Console.WriteLine("--------------------------------------------------------------------------------------------------------------------");
                        break;
                    case MaxPool pool:


                        if (pool.OutputShape.Length == 1)
                        {
                            shapeString = "unknown";
                            isEmptyInput = true;
                        }
                        else
                        {
                            shapeString = $"[{pool.OutputShape[0]},{pool.OutputShape[1]},{pool.OutputShape[2]}]";
                        }

                        Console.WriteLine("--------------------------------------------------------------------------------------------------------------------");
                        Console.WriteLine($"| {numberOfLayer}. Name: {pool.Name} | PoolSize: {pool.PoolSize} * {pool.PoolSize} | stride: {pool.PoolSize} | Output shape: {shapeString}  ");
                        Console.WriteLine("--------------------------------------------------------------------------------------------------------------------");
                        break;
                    default:
                        Console.WriteLine("--------------------------------------------------------------------------------------------------------------------");
                        Console.WriteLine("| This layer is not defined |");
                        Console.WriteLine("--------------------------------------------------------------------------------------------------------------------");
                        break;
                }
            }
            if (isEmptyInput)
            {
                Console.WriteLine($"! Warning !");
                Console.WriteLine("These are indicative model data only!");
                Console.WriteLine("This information is before the input values are set, so some data may be missing from the model. ");
                Console.WriteLine("It fills in on the first pass through the model; to set it earlier, call model.Layers.SetInputShape(new int[] { … }).");
            }   
            else
            {
                Console.WriteLine($"Total trainable params: {CountTrainableParams(model)}");
                Console.WriteLine($"Current epoch: {model.Train.CurrentEpoch}");
                Console.WriteLine($"Target epoch: {model.Train.TotalEpoch}");
                Console.WriteLine($"Size of mini batch: {model.Train.MiniBatchSize}");
            }

            Console.WriteLine();
        }

        // Conv = numFilters × (kH × kW × inChannels) + biases; previously the × numFilters
        // (Kernels.Length) factor was missing, so conv layers were heavily undercounted.
        /// <summary>
        /// Returns the number of trainable parameters of the model. MaxPool and similar layers
        /// contribute 0.
        /// </summary>
        public static int CountTrainableParams(NeuralNetwork model)
        {
            int total = 0;
            foreach (Layer layer in model.Layers.Layers)
            {
                switch (layer)
                {
                    case Dense dense:
                        total += dense.Neurons.Count * dense.InputShape[0] + dense.Neurons.Count;
                        break;
                    case RNN rnn:
                        total += rnn.Neurons.Count * (rnn.InputShape[0] + 1) + rnn.Neurons.Count;
                        break;
                    case Conv conv:
                        total += conv.Kernels.Length
                                 * (conv.Kernels[0].Count() * conv.Kernels[0][0].Count() * conv.Kernels[0][0][0].Count())
                                 + conv.Biases.Length;
                        break;
                }
            }
            return total;
        }

        /// <summary>
        /// Prints a training progress report: epoch, losses, validation accuracy, best valid loss
        /// and the estimated time to completion.
        /// </summary>
        /// <param name="model">The model being trained.</param>
        /// <param name="trainLoss">Training loss; when null, the model's current average loss is printed as "Loss".</param>
        /// <param name="validLoss">Validation loss; when null (and <paramref name="trainLoss"/> is set), it is read from the model's loss function.</param>
        /// <param name="validAccuracy">Validation accuracy in percent; omitted from the output when null.</param>
        public static void ShowEpochInfo(NeuralNetwork model, double? trainLoss = null, double? validLoss = null, double? validAccuracy = null)
        {
            DateTime timeNow = DateTime.Now;

            TimeSpan subTime;
            TimeSpan estimatedCompletionTime;

            // The state lives on the Train instance (previously statics), so a second training run
            // in the process no longer uses the first run's times. pastEpochs == 0 is handled
            // explicitly: dividing by zero produced a TimeSpan of Infinity ticks → OverflowException.
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
            // Same guard as in Percent(): TotalEpoch == 0 is legitimate (manual training via
            // Fit/UpdateParams has no epochs), and dividing by zero printed "NaN %".
            string progress = model.Train.TotalEpoch > 0
                ? $"{(double)model.Train.CurrentEpoch / model.Train.TotalEpoch * 100:0.00} %"
                : "—";

            Console.WriteLine("########################################################################");
            Console.WriteLine($"Epoch: {model.Train.CurrentEpoch} / {model.Train.TotalEpoch} {progress}");
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

            // best valid loss so far + the epoch it was reached (skipped until the first report)
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

        // Accuracy in percent. An empty set is a legitimate state (the user turned off the final
        // test, or the set came out empty) — without this guard 0/0 gave NaN and the output
        // read "Accuracy: NaN%".
        //
        // Computed in `double`. Previously a triple cast to `float` (just for printing) lost
        // digits and produced values like "60.000004 %".
        private static string Percent(int part, int whole)
            => whole > 0
                ? $"{(double)part / whole * 100:0.00} %"
                : "n/a (no samples)";

        /// <summary>
        /// Prints the result of an evaluation: accuracy and the numbers of correct and wrong samples.
        /// </summary>
        /// <param name="score">Number of correctly predicted samples.</param>
        /// <param name="maxScore">Total number of evaluated samples.</param>
        public static void ShowScore(int score, int maxScore)
        {
            Console.WriteLine();
            Console.WriteLine("------------- Model Score ------------- ");
            Console.WriteLine($"Accuracy: {Percent(score, maxScore)}");
            Console.WriteLine($"Total number of samples tested: {maxScore}");
            Console.WriteLine($"Number of correctly guessed tested samples: {score}");
            Console.WriteLine($"Number of mistyped samples tested: {maxScore - score}");
            Console.WriteLine();
        }
        
        // Making up a test set from the validation set would be misleading, so a missing one is
        // simply not shown.
        /// <summary>
        /// Prints accuracy on the test, train and validation datasets.
        /// </summary>
        /// <param name="test">(correct, total) on the test set; null when no test set was supplied via SetDatasets().</param>
        /// <param name="train">(correct, total) on the training set.</param>
        /// <param name="valid">(correct, total) on the validation set.</param>
        public static void ShowDatasetScores((int Correct, int Total)? test, (int Correct, int Total) train, (int Correct, int Total) valid)
        {
            Console.WriteLine();
            Console.WriteLine("------------- Model Score ------------- ");

            if (test != null)
            {
                (int correct, int total) = test.Value;
                Console.WriteLine($"Accuracy on test dataset: {Percent(correct, total)}");
                Console.WriteLine($"Total number of samples tested on test dataset: {total}");
                Console.WriteLine($"Number of correctly guessed tested samples on test dataset: {correct}");
                Console.WriteLine($"Number of mistyped samples tested on test dataset: {total - correct}");
            }
            else
            {
                Console.WriteLine("Test dataset: none (not supplied via SetDatasets)");
            }

            Console.WriteLine();
            Console.WriteLine($"Accuracy on train dataset: {Percent(train.Item1, train.Item2)}");
            Console.WriteLine($"Accuracy on valid dataset: {Percent(valid.Item1, valid.Item2)}");
            Console.WriteLine();
        }

        /// <summary>Prints a notice that training was stopped early, with the best valid loss and its epoch.</summary>
        public static void ShowEarlyStopping(uint epoch, uint patience, double bestLoss, uint bestEpoch)
        {
            Console.WriteLine();
            Console.WriteLine("------------- Early stopping ------------- ");
            Console.WriteLine($"Training stopped at epoch {epoch}: {patience} valid reports without improvement.");
            Console.WriteLine($"Best valid loss: {bestLoss} @ epoch {bestEpoch}");
            Console.WriteLine();
        }


        // Previously `fatalError: true` called `Environment.Exit(0)` — the library killed the host
        // process, with exit code 0 ("success") at that. Control flow now belongs to the caller.
        /// <summary>
        /// ONLY prints an error; it does not throw or exit. The caller that reports the error is
        /// responsible for throwing (see <see cref="TrainingDivergedException"/>).
        /// </summary>
        /// <param name="name">Short name of the error.</param>
        /// <param name="description">Description of the error.</param>
        /// <param name="fatalError">Only changes the header of the output.</param>
        public static void ShowError(string name, string description, bool fatalError = false)
        {
            Console.WriteLine();
            Console.WriteLine(fatalError
                ? "----------------------  !Fatal Error!  ---------------------- "
                : "----------------------   Error  ---------------------- ");
            Console.WriteLine($"Name: {name}");
            Console.WriteLine($"Description: {description} ");
        }
    }
}
