# MDNN.Plotting

Loss charts for [MDNN](https://www.nuget.org/packages/MDNN) training runs, drawn with
[ScottPlot](https://scottplot.net). Kept in a separate package so the core library does not
depend on a plotting stack.

```
dotnet add package MDNN.Plotting
```

```csharp
using MDNN;

model.Train.UseLossChart();   // TrainLoop then saves loss.png at the end
model.Train.TrainLoop(inputs, targets, epochs: 50, miniBatchSize: 32);
```

The chart of training and validation loss over epochs is saved as `loss.png` in the working
directory at the end of `TrainLoop()`, and also when training diverges.
`model.Train.ShowLossChart = false` turns it off again.

To plot with something else, skip this package and assign your own callback to
`model.Train.LossChart` — it receives the epochs, training loss and validation loss.

Full documentation: [github.com/SilM4r/MDNN](https://github.com/SilM4r/MDNN)
