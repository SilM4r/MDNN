using ScottPlot;

namespace MDNN
{
    // Lives in the separate MDNN.Plotting package, not in the core: the core only offers a hook
    // (Train.LossChart) and knows nothing about rendering.
    /// <summary>
    /// Plots the training and validation loss with ScottPlot.
    /// </summary>
    public static class GraphPlotter
    {
        /// <summary>
        /// Enables the loss chart at the end of TrainLoop (and on divergence): <c>model.Train.UseLossChart();</c>
        /// </summary>
        /// <returns>The same <see cref="Train"/> instance, for chaining.</returns>
        public static Train UseLossChart(this Train train)
        {
            train.LossChart = SaveLossChart;
            return train;
        }

        /// <summary>
        /// Saves a train/validation loss chart to <c>loss.png</c> (800×600) in the current directory.
        /// </summary>
        /// <param name="epochs">Epoch numbers (x axis).</param>
        /// <param name="trainLoss">Training loss per epoch.</param>
        /// <param name="validLoss">Validation loss per epoch.</param>
        public static void SaveLossChart(int[] epochs, double[] trainLoss, double[] validLoss)
        {
            if (trainLoss.Length != epochs.Length || validLoss.Length != epochs.Length)
            {
                throw new ArgumentException(
                    $"Epoch, train loss and valid loss must have the same length ({epochs.Length}, {trainLoss.Length}, {validLoss.Length}).");
            }


            Plot myPlot = new();


            var curve = myPlot.Add.Scatter(epochs, trainLoss);

            curve.LegendText = "Train Loss";
            var curve1 = myPlot.Add.Scatter(epochs, validLoss);
            curve1.LegendText = "Valid Loss";

            myPlot.XLabel("Epochs");
            myPlot.YLabel("Loss");
            myPlot.Title("Loss graph");
            myPlot.Legend.Alignment = Alignment.UpperRight;
            myPlot.SavePng("loss.png", 800, 600);
        }

        
    }
}
