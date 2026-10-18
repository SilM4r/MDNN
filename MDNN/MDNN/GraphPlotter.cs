using ScottPlot;

namespace My_DNN
{
    public static class GraphPlotter
    {
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
