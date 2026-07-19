# MDNN (My Deep Neural Network)

MDNN is a library for designing and training neural networks in C#. It allows straightforward creation and configuration of neural network models, their training, and their subsequent integration into applications.

## Contents

- [Key features](#key-features)
- [Installation](#installation)
- [Quick start](#quick-start)
- [Model configuration](#model-configuration)
- [Adding layers](#adding-layers)
- [Training the model](#training-the-model)
- [GPU acceleration and asynchronous execution](#gpu-acceleration-and-asynchronous-execution)
- [Saving and loading models](#saving-and-loading-models)
- [Supporting utilities](#supporting-utilities)
- [Tests](#tests)

## Key features

Supported layer types:

- Dense (fully connected)
- Convolutional (Conv)
- Max pooling (MaxPool)
- Recurrent (RNN)

Additional capabilities:

- A range of activation functions, optimizers, and loss functions
- Straightforward integration into C# projects
- Optional GPU computation
- Asynchronous training and inference
- Saving and loading models in JSON format
- Pre-built training loops

## Installation

MDNN is distributed as a dynamic library, `MDNN.dll`. To use it:

1. Add a reference to `MDNN.dll` in your project.
2. Include the relevant namespaces in your source code.

Alternatively, download the repository and build it; the build produces a fresh `MDNN.dll`.

## Quick start

A minimal example that creates and trains a network:

```csharp
using My_DNN;
using My_DNN.Layers;
using My_DNN.Layers.classes;
using My_DNN.Optimizers;
using My_DNN.Loss_functions;
using My_DNN.Activation_functions;

double[][] inputsDataset  = { /* input samples */ };
double[][] outputDataset  = { /* corresponding targets */ };

Layer outputLayer = new Dense(1, new Linear()); // output layer
Optimizer optimizer = new SGD(0.01);            // optimization algorithm
Loss loss = new MSE();                          // loss function

uint epochs = 1000;

MDNN model = new MDNN(outputLayer, optimizer, loss);
model.Layers.Add(new Dense(8, new ReLu()));     // hidden layer

model.Train.TrainLoop(inputsDataset, outputDataset, epochs, 1);

model.SaveAsJson("save");
```

## Model configuration

The model is created through the `MDNN` class, which is the central object for working with the network:

```csharp
MDNN model = new MDNN(outputLayer, optimizer, loss);
```

Constructor parameters:

- `outputLayer` (required) — a `Layer` representing the output layer.
- `optimizer` (optional) — an `Optimizer`. Defaults to `SGD(0.0001)`.
- `loss` (optional) — a loss function. Defaults to `MSE()`.

Supported optimizers: `SGD`, `Adam`, `Momentum`.
Supported loss functions: `MSE`, `CrossEntropy`.

Each model owns its own configuration, so multiple independent models can coexist in the same process.

You can also create custom optimizers or loss functions by inheriting the corresponding base class (for example `Optimizer` or `Loss`) and implementing its abstract members.

Note on `CrossEntropy`: it is intended to be paired with a `Softmax` output layer. It computes the fused softmax + categorical cross-entropy gradient (`output − target`), which is the numerically stable form. Using `CrossEntropy` without a `Softmax` output layer raises an exception.

## Adding layers

Layers are added through the `Layers` property:

```csharp
model.Layers.Add(new Dense(64, new ReLu()));    // hidden layer, 64 neurons, ReLU
model.Layers.Add(new Dense(32, new Sigmoid())); // hidden layer, 32 neurons, sigmoid
```

Besides `Add()`, the `Layers` API includes:

- `Insert()` — insert a layer at a given position
- `RemoveAt()` — remove a layer at a given position
- `OutputLayerActivationFunc()` — set a new output activation function
- `ClearAllLayersAndSetNewOutputLayer()` — remove all layers and set a new output layer

Supported layers:

- `Dense()`
- `RNN()`
- `Conv()`
- `MaxPool()`

If a layer constructor is called without an activation function, the default hidden-layer activation (ReLU) is used. Pass the activation explicitly for output layers (for example `Linear` for regression or `Softmax` for classification).

You can also define a custom layer by inheriting one of the abstract classes `Layer`, `LayerBasedOnNeurons`, or `LayerWithUntrainedParameters` and implementing their abstract members.

Available activation functions: `Linear`, `ReLu`, `Leak_ReLu`, `Sigmoid`, `Tanh`, `Softmax`.

## Training the model

Training is driven by the `Train` class. Four levels of control are available, from fully automated to fully manual.

### `TrainLoop()`

The most complete training procedure. It provides:

- Automatic checkpointing of the best-validation model (early-stopping checkpoint)
- Automatic shuffling and splitting of the dataset into training, validation, and test sets
- Progress reporting to the console
- Detection of `NaN` values
- Automatic plotting of the loss over epochs

Parameters:

- `Array inputs_values` (required) — input dataset. Each row is one training sample.
- `Array current_output_values` (required) — corresponding targets.
- `uint number_of_epoch` (required) — number of training epochs.
- `uint size_of_mini_batch` (optional, default `1`) — minibatch size.
- `bool isSequence` (optional, default `false`) — set to `true` for sequential input data (for example time series).

### `SimpleTrainLoop()`

A simplified training loop with checkpointing, console reporting, and `NaN` detection, without the dataset splitting and plotting of `TrainLoop()`.

Parameters:

- `double[][] inputs_values` (required)
- `double[][] current_output_values` (required)
- `uint number_of_epoch` (required)
- `uint size_of_mini_batch` (optional, default `1`)

### `Fit()` and `UpdateParams()`

An intermediate approach that lets you write your own training loop. `Fit()` runs the forward pass and backpropagation but accumulates gradients instead of applying them; `UpdateParams()` applies the accumulated gradients. Calling them one after another is equivalent to single-sample training; accumulating several `Fit()` calls before one `UpdateParams()` is equivalent to minibatch training.

```csharp
Random rnd = new Random();
double[][] inputsDataset = { /* input data */ };
double[][] currentOutputDataset = { /* targets */ };

MDNN model = new MDNN(new Dense(3), new Adam(0.001));

int numberOfEpochs = 5000;
int miniBatchSize = 16;

for (int i = 0; i < numberOfEpochs; i++)
{
    for (int j = 0; j < miniBatchSize; j++)
    {
        int num = rnd.Next(inputsDataset.Length);
        model.Train.Fit(new Tensor(inputsDataset[num]), new Tensor(currentOutputDataset[num]));
    }

    model.Train.UpdateParams();
}
```

### `FeedForward()` and `BackPropagation()`

The most granular approach, splitting `Fit()` into a separate forward pass (`FeedForward()`) and backpropagation (`BackPropagation()`). This gives full control over the individual training steps, which is useful for research or advanced optimization. After backpropagation, call `UpdateParams()` to apply the changes.

`BackPropagation()` has two overloads: one that takes the target values (and computes the layer gradients internally) and one that takes precomputed per-layer gradients.

## GPU acceleration and asynchronous execution

### GPU

Network computation can optionally run on an NVIDIA GPU through the accompanying `gpu.dll` library (written in C++ / CUDA). This requires the CUDA Toolkit and `gpu.dll`. GPU computation is a per-model setting:

```csharp
model.Context.CalculationViaGpu = true;
```

Support currently targets NVIDIA GPUs only.

### Asynchronous execution

Each synchronous method has an asynchronous counterpart, for example:

- `TrainLoop()` — `TrainLoopAsync()`
- `Fit()` — `FitAsync()`
- `GetResults()` — `GetResultsAsync()`

```csharp
await model.Train.TrainLoopAsync(inputsDataset, outputDataset, 1000);
```

## Saving and loading models

After training, a model can be saved to JSON and later loaded for inference. No retraining is required to use a saved model.

```csharp
model.SaveAsJson("save"); // writes save.json

MDNN loaded = MDNN.LoadModel("save.json");
Tensor input = new Tensor(Tensor.ConvertJaggedToMulti(inputsDataset));
Tensor result = loaded.GetResults(input);
```

## Supporting utilities

### Tensor

`Tensor` is the universal data type for multidimensional arrays. It stores:

- the original multidimensional array (`OriginalInput`),
- an equivalent flat array (`Data`) for faster computation,
- and the shape as a list of dimensions (`Shape`).

It supports reshaping via `Reshape(int[] newShape)`, along with convenient element access and conversions between jagged and multidimensional arrays.

### Console output

The static `ConsoleControler` class handles console output:

- `ShowModelInfo()` — prints detailed information about the model
- `ShowEpochInfo()` — prints information about the current epoch during training
- `ShowScoreOfmodel()` — prints the model's accuracy
- `ErrorHandler()` — prints error messages

### NetworkContext

Each model owns a `NetworkContext` (`model.Context`) that holds its runtime configuration: the loss function, optimizer, input shape, sequence-training flag, and the GPU flag. Because this state is per-model, two models in the same process do not interfere with each other.

`GeneralNeuralNetworkSettings` holds only process-wide defaults (the default activation functions and a shared random generator).

### Plotting

The `GraphPlotter` class visualizes training progress. Its `ShowLossGraph()` method produces a graph of training and validation loss over epochs and saves it as `loss.png` in the application's root directory. This makes it easy to spot overfitting or undertraining. Plotting uses the ScottPlot library.

## Tests

The repository includes an xUnit test project (`MDNN.Tests`) with numerical gradient checks for every layer, unit tests for the activation, loss, and optimizer implementations, and end-to-end training smoke tests. Run them with:

```
dotnet test
```

The gradient checks act as a regression guard: any change that breaks the underlying math is caught automatically.
