using ScottPlot;

namespace My_DNN
{
    // Graf trénovací a validační loss přes ScottPlot. Je v samostatném balíčku MDNN.Plotting,
    // ne v jádru: jádro jen nabídne háček (Train.LossChart) a o vykreslení neví nic.
    public static class GraphPlotter
    {
        // Zapne graf na konci TrainLoop (a při divergenci): model.Train.UseLossChart();
        public static Train UseLossChart(this Train train)
        {
            train.LossChart = ShowLossGraph;
            return train;
        }

        public static void ShowLossGraph(int[] epoch,double[] TrainDataLoss, double[] ValidDataLoss)
        {
            if (TrainDataLoss.Length != epoch.Length || ValidDataLoss.Length != epoch.Length)
            {
                throw new ArgumentException(
                    $"Epoch, train loss and valid loss must have the same length ({epoch.Length}, {TrainDataLoss.Length}, {ValidDataLoss.Length}).");
            }


            Plot myPlot = new();


            var curve = myPlot.Add.Scatter(epoch, TrainDataLoss);

            curve.LegendText = "Train Loss";
            var curve1 = myPlot.Add.Scatter(epoch, ValidDataLoss);
            curve1.LegendText = "Valid Loss";

            myPlot.XLabel("Epochs");
            myPlot.YLabel("Loss");
            myPlot.Title("Loss graph");
            myPlot.Legend.Alignment = Alignment.UpperRight;
            myPlot.SavePng("loss.png", 800, 600);
        }

        
    }
}
