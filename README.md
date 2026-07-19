# MDNN (My Deep Neural Network)

MDNN is a library for designing and training neural networks in C#. It allows straightforward creation and configuration of neural network models, their training, and their subsequent integration into applications.

## Contents

- [Key features](#key-features)
- [Installation](#installation)
- [Quick start](#quick-start)
- [Examples](#examples)
- [Model configuration](#model-configuration)
- [Adding layers](#adding-layers)
- [Training the model](#training-the-model)
- [GPU acceleration and asynchronous execution](#gpu-acceleration-and-asynchronous-execution)
- [Saving and loading models](#saving-and-loading-models)
- [Supporting utilities](#supporting-utilities)
- [How it works internally](#how-it-works-internally)
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

## Examples

Full, runnable examples for each task type — classification, regression, sequential data (RNN), and image data (Conv + MaxPool) — live in a separate repository:

[github.com/SilM4r/MDNN_examples](https://github.com/SilM4r/MDNN_examples)

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

## How it works internally

This section describes what happens under the hood. It is not required reading to use the library, but it explains the design and is useful when extending it.

### Object model

```
MDNN (model)
├── NetworkContext        per-model runtime config (loss, optimizer, input shape, flags)
├── LayerManager (Layers) ordered list of layers + lazy shape inference
│   └── Layer             Dense · Conv · MaxPool · RNN
│       └── Neuron        weights + bias + its own optimizer   (Dense, RNN)
└── Train                 training loops and the forward/backward entry points
```

- `MDNN` is the top-level object. It owns a `NetworkContext`, a `LayerManager` (exposed as `model.Layers`), and a `Train` helper (`model.Train`).
- `Layer` is the abstract base for every layer type. `Dense` and `RNN` are built from `Neuron` objects (they extend `LayerBasedOnNeurons`); `Conv` holds kernels and biases directly; `MaxPool` has no trainable parameters and only records which positions were selected.
- A `Neuron` holds its weights, bias, accumulated gradients, and its **own** optimizer instance.
- `Tensor` is the data carrier passed between layers (a flat `Data` array plus a `Shape`).
- `NetworkContext` (`model.Context`) holds the per-model state. Because this state is not global, two models in the same process are fully independent.

### Lazy shape inference

Layers are constructed without knowing their input size — you only specify the number of neurons (or kernels). On the first forward pass the model records the input shape into `Context.InputShape` and calls `Layers.SetInputSizeForFirstLayer()`, which walks the layers calling each one's `LayerAdjustment()`. Every layer derives its real input size from the previous layer's output size and only then allocates its parameters (neurons, kernels). This is why a layer's weights do not exist until the model first sees data.

### Forward pass

`model.GetResults(input)` threads a `Tensor` through the layers in order, calling each layer's `FeedForward()`. A `Dense` layer computes `w · x + b` per neuron followed by the activation; layer-wide activations such as `Softmax` are applied across the whole layer at once.

### Backward pass

Backpropagation is driven by `Gradient.GetGradients(target, model)`:

1. The output-layer error is computed from the loss derivative. For an element-wise output activation it is multiplied by that activation's derivative; for the fused softmax + cross-entropy case (below) it is used directly.
2. The error is propagated backwards: each layer's `CalculateLayerGradients()` turns the next layer's error into its own error (the chain rule), applying the layer activation's derivative along the way.
3. Each layer's `BackPropagation()` then accumulates the parameter gradients (for neuron-based layers, into each `Neuron`'s `gradientsW` / `gradientsB`).

Gradients are **accumulated, not applied**. `UpdateParams()` divides the accumulated gradients by the number of samples seen and hands them to the optimizer. This is why several `Fit()` calls followed by one `UpdateParams()` are equivalent to one minibatch of that size.

### Softmax + cross-entropy fusion

When the loss is `CrossEntropy` (which reports `RequiresSoftmax`) and the output layer uses `Softmax`, the output gradient is computed directly as `output − target`. This is the fused, numerically stable form: the softmax Jacobian is skipped and the activation derivative is deliberately *not* multiplied in, which would otherwise apply it twice. Using `CrossEntropy` without a `Softmax` output layer raises an exception.

### Optimizers

Each `Neuron` (and each `Conv` layer) owns its own optimizer instance, cloned from the one in the model's `NetworkContext`. Optimizer state is therefore per-parameter and per-model — for example `Adam` keeps first/second-moment estimates with bias correction for every individual weight. The optimizer's `Update(value, gradient, index)` returns the new parameter value.

### Recurrent layers (RTRL)

`RNN` layers train with **Real-Time Recurrent Learning** rather than backpropagation-through-time. Instead of unrolling the sequence, the layer carries forward-in-time sensitivities (`∂h/∂weight` and `∂h/∂bias`) that are advanced at every timestep inside `FeedForward()`. During backpropagation the incoming error is multiplied by those stored sensitivities. Call `model.ResetSequence()` at the start of each sequence to zero the hidden state and the sensitivities.

### Weight initialization

Dense and RNN neurons are initialized with a Xavier/He-style uniform scheme — `U(−1, 1) · sqrt(6 / n_inputs)` — and biases start at zero.

### Correctness: numerical gradient checking

The entire backward pass is validated by the test suite using numerical gradient checking (central difference): every layer's analytic gradients are compared against finite-difference estimates. These checks act as a regression guard — any change that breaks the underlying math is caught automatically.

## Tests

The repository includes an xUnit test project (`MDNN.Tests`) with numerical gradient checks for every layer, unit tests for the activation, loss, and optimizer implementations, and end-to-end training smoke tests. Run them with:

```
dotnet test
```

The gradient checks act as a regression guard: any change that breaks the underlying math is caught automatically.
