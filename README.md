**MDNN (My Deep Neural Network)**
==============

MDNN (My Deep Neural Network) is a library for designing and training neural networks in C#. It allows easy creation and configuration of neural network models, their training and subsequent integration into projects.

## 📚 **Content**

- [📌 Key features](#-key-properties)
- [🛠 Installation](#-installation)
- [🚀 Quick Start](#-quick-start)
- [⚙ Model Configuration](#-configuration-model)
- [📌 Add layers to the model](#-adding-layers-to-model)
- [🎯 Model Training](#-model-training)
- [⏳ Optimization](#-optimization)
- [📦 Production Deployment](#-production-deployment)
- [👏 Support features](#-support-features)

## 📌 Key features

Support for different layer types, including:
- **Dense** (fully connected layers)
- **MaxPooling**
- **Convolutional neural networks (CNNs)**
- **Recurrent Neural Networks (RNN)**
  
Possibility to use a wide range of:
- Activation functions
- Optimization algorithms
- Loss functions
- Easy integration into C# projects
- Support for GPU computation acceleration
- Save and load models in JSON format
- Predefined training loops to simplify the training process

## 🛠 Installation

MDNN is distributed as a dynamic library **MDNN.dll**. To use it, you must:

1. Add the **MDNN.dll** file to the project. (add a new slot to the project)
2. Include the appropriate namespaces in the source code.

Alternatively, you can download the entire repository and run the build, which will automatically generate a new **MDNN.dll** file.

## 🚀 Quick Start

Below is a simple example of using MDNN to create and train a neural network.

```csharp
using My_DNN. Layers;
using My_DNN. Layers.classes;
using My_DNN. Optimizers;
using My_DNN;
using My_DNN. Activation_functions;
using My_DNN. Loss_functions;

namespace MDNN_example
{

    internal class Program
    {

        static void Main(string[] args)
        {

            double[,] inputsDataset = new double[][] {...}; // Input data
            double[,] outputDataset = new double[][] {...}; // Corresponding output data
            
            Layer outputLayer = new Dense(1, new Linear()); // Configure the output layer
            Optimizer optimizer = new SGD(0.01); // Setting up the optimization algorithm
            Loss loss = new MSE(); // Definition of a loss function
            
            uint epoch = 1000;
            
            //Initializing the model
            MDNN model = new MDNN(outputLayer, optimizer, loss);
            model. Layers.Add(new Dense(1, new ReLu())); // Add a hidden layer
            
            Tensor tensorInputDataset = new Tensor(inputsDataset);
            Tensor tensorOutputDataset = new Tensor(outputDataset);
            
            //Train a model
            model. Train.TrainLoop(tensorInputDataset, tensorOutputDataset, epoch, 1);
            
            //Save the model
            model. SaveAsJson("save");
        }
    }
}
```
## ⚙ Model configuration

The MDNN library allows easy configuration of the neural network architecture, including the output layer, optimization algorithm and loss function.
The model is initialized using the MDNN class, which serves as the central object for manipulating the neural network:

```csharp
MDNN model = new MDNN(outputLayer, optimizer, loss);
```
### Constructor parameters

- **`outputLayer`** *(required parameter)* – object of the **Layer** type, representing the output layer of the mesh.
- **`optimizer`** *(optional parameter)* – object of the **Optimizer** type, that specifies the optimization algorithm. If not specified, the default value is SGD(0.0001)**.
- **`loss`** *(optional parameter)* – object representing the loss function. The default value is **MSE()**.

supported optimizers: SGD,ADAM,Momentum
Supported loss functions: MSE, Cross Entropy

Alternatively, the library also supports the creation of your own optimizers and lossy functions – just inherit the corresponding parent class, for example **`loss`**.

## 📌 Add layers to the model

Additional layers can be added to the model using the **`Layers`** class:

```csharp
model.Layers.Add(new Dense(64, new ReLu())); // Addition of a hidden layer with 64 neurons and ReLU activation
model.Layers.Add(new Dense(32, new Sigmoid())); // Additional layer with 32 neurons and sigmoid activation
```

In addition to the Add()`** function, the **`Layers` class also contains methods for removing or adding layers at a specific position, and a number of other functions for manipulating layers.
- **`Insert()`**
- **`RemoveAt()`**
- **`OutputLayerActivationFunc()`** - set a new output activation function
- **`ClearAllLayersAndSetNewOutputLayer`** - deletes all layers and sets a new output layer

The MDNN library supports the following layers:
- `Dense()`
- `RNN()`
- `Conv()`
- `MaxPool()`
- WIP: Transformer Layer or Attention Layer

If necessary, you can also create your own specialized layer. It is enough to inherit one of the following abstract classes:
- `Layer`
- `LayerBasedOnNeurons`
- `LayerWithUntrainedParameters`

Then, it is necessary to implement all their abstract methods. Once a new layer is defined, it can be added to the model and used in training.

## 🎯 Train a model

The model is trained using the Train class, which contains all the necessary methods to control the training. The user has the option to choose between four training methods according to the desired degree of control over the model training.
- **`TrainLoop()`**
- **`SimpleTrainLoop()`**
- **`Fit()`** and **`UpdateParams()`**
- **`BackPropagation()`** and **`FeedForward()`**

### **`TrainLoop()`**
This method is the main and most advanced training procedure in the library. It includes a complete train loop and provides the following advanced features:

- Automatic saving of the model with the best validation (so-called early stopping checkpoint).
- Automatic mixing and division of the dataset into training, validation and testing parts.
- Continuous listing of information about the training progress (e.g. current epochs, metrics) to the console.
- Error detection, such as the occurrence of NaN values.
- Automatic plotting of the progress of the loss function across epochs.

Parameters:
- **`Tensor inputs_values`** – *required parameter*
Input dataset in tensor format. Each row corresponds to one training sample.

- **`Tensor current_output_values`** – *mandatory parameter*
Corresponding outputs (labels) for input data, also in tensor format.

- **`uint number_of_epoch`** – *required parameter*
Specifies the number of training epochs.

- **`uint size_of_mini_batch`** – *optional parameter*
The size of the minibatch used during training. If not specified, the default value of `1` is used.

- **`bool isSequence`** – *optional*
If set to `true`, the model will expect sequential input data (e.g. time series, video or other sequential data). The default value is `false`.

### **`SimpleTrainLoop()`**
It is a simplified version of the **`TrainLoop()`** method. It includes a complete train loop and provides the following functions:

- Automatic saving of the model with the best validation (so-called early stopping checkpoint).
- Continuous listing of information about the training progress (e.g. current epochs, metrics) to the console.
- Error detection, such as the occurrence of NaN values.

Parameters:
- **`double[][] inputs_values`** – *required parameter*
Input dataset in double[][] format. Each row corresponds to one training sample.

- **`double[][] current_output_values`** – *required parameter*
Corresponding outputs (labels) for input data, also in double[][] format.

- **`uint number_of_epoch`** – *required parameter*
Specifies the number of training epochs.

- **`uint size_of_mini_batch`** – *optional parameter*
The size of the minibatch used during training. If not specified, the default value of `1` is used.

### **`Fit()`** and **`UpdateParams()`**

It is an intermediate approach to model training that allows you to implement your own training loop. This approach does not provide pre-built functionality, but the user has a set of support classes that can be used to create additional components, such as listing training information to the console, visualizing training losses using graphs, or implementing test procedures to evaluate the model.

This method consists of two functions: **`Fit()`** and **`UpdateParams()`**.
The **`Fit()`** function performs both output calculation (*feedforward*) and backpropagation, but without immediate updating of the model parameters – instead, gradients are accumulated.

Updating of parameters based on accumulated gradients is performed only when the **`UpdateParams()`` function is called. This approach is equivalent to training with a *minibatch*, which can lead to more efficient and stable learning.

If the user does not want to use the minibatch mode, it is sufficient to call both functions immediately after each other.

```csharp
Random rnd = new Random();
double[][] inputsDataset = new double[][] {...}; // input data
double[][] currentOutputDataset = new double[][] {...}; // currentOutput data

// Initializing the model
MDNN model = new MDNN(new Dense(3), new Adam(0.001));

//Number of epochs and minibatch size
int number_of_epoch = 5000;
int size_of_miniBatch = 16;

// number of all elements in the dataset
int number_of_element_intDataset = inputsDataset.Length;

// Main training loop
for (int i = 0, i < number_of_epoch, i++)
 {

  //Secondary Loop Minibatch
  for (int j = 0, j < size_of_miniBatch, j++)
  {
      int num = rnd. Next(number_of_element_intDataset);
      
      double[] inputs = inputsDataset[num];
      double[] output = currentOutputDataset[num];
      
      // calculation on one specific element, which is randomly selected from the whole dataset
      model. Train.Fit(new Tensor(inputs),new Tensor(output));
  }
  
  //Updating training parameters
  model. Train.UpdateParams();
 }

// supporting function that tests the entire dataset on the trained dataset
model. Train.TestNeuralNetwork(new Tensor(Tensor.ConvertJaggedToMulti(inputsDataset)), new Tensor(Tensor.ConvertJaggedToMulti(currentOutputDataset)));
```

Parameters:

**`Fit()`** :
- **`Tensor inputs_values`** – *required parameter*
Input dataset in tensor format. Each row corresponds to one training sample.

- **`Tensor target_values`** – *required parameter*
Corresponding outputs (labels) for input data, also in tensor format.

**`UpdateParams()`** - has no parameters

### **`BackPropagation()`** and **`FeedForward()`**

It is the most advanced training method that provides maximum control over the individual stages of learning. Unlike the Fit()`** and **`UpdateParams()`** approach, the **`Fit()`** method is divided into two separate functions: **`FeedForward()`** and **`BackPropagation()`**. The **`FeedForward()`` function performs the forward calculation of the neural network, while **`BackPropagation()`** provides backpropagation of the error and calculation of gradients. This approach allows detailed manipulation of the individual steps of the training process, which is especially suitable for research purposes or advanced optimizations.

**Warning:** To update the model parameters itself, it is necessary to call the **`UpdateParams()`` method afterwards.

```csharp

Random rnd = new Random();
double[][] inputsDataset = new double[][] {...}; input data
double[][] currentOutputDataset = new double[][] {...}; currentOutput data

// Initializing the model
MDNN model = new MDNN(new Dense(3), new Adam(0.001));

// Number of epochs and minibatch size
int number_of_epoch = 5000;
int size_of_miniBatch = 16;

//number of all elements in the dataset
int number_of_element_intDataset = inputsDataset.Length;

//Main training loop
for (int i = 0, i < number_of_epoch, i++)
 {

    //Secondary Loop Minibatch
    for (int j = 0, j < size_of_miniBatch, j++)
    {
        int num = rnd. Next(number_of_element_intDataset);
        
        double[] inputs = inputsDataset[num];
        double[] output = currentOutputDataset[num];
        
        // Forward calculation
        model.Train.FeedForward(new Tensor(Tensor.ConvertJaggedToMulti(inputs)));
        
        // Calculation of gradients (supporting method)
        Tensor[] gradients = Gradient.GetGradients(new Tensor(Tensor.ConvertJaggedToMulti(inputs)), model);
        
        // Backpropagation of gradients
        model.Train.BackPropagation(gradients);
     }
    
    // Updating training parameters
    model.Train.UpdateParams();
 }

// A supporting function that tests the model on a trained dataset
model.Train.TestNeuralNetwork(new Tensor(Tensor.ConvertJaggedToMulti(inputsDataset)), new Tensor(Tensor.ConvertJaggedToMulti(currentOutputDataset)));
```

Method parameters:


**`FeedForward()`**

- **`Tensor inputs_values`** – *required parameter*
Input data in tensor format.

**`BackPropagation()`**

The **`BackPropagation()`` method has two overloads:

1. **First overload:**
- **`Tensor[] layer_gradients`** – *required parameter*
An array of gradients of individual layers in tensor format. It is used for manually controlled back-propagation.

2. **Second overload:**
- **`Tensor target_values`** – *required parameter*
Target output values corresponding to the input data, also in tensor format. In this overload, the method internally calculates the appropriate `layer_gradients` values.

## ⏳ Optimization

The MDNN library supports efficient optimization of neural network calculations. The main optimization techniques include:

- **GPU Utilization** – Neural network calculations can be performed on GPUs, which significantly speeds up model training, especially when working with large amounts of data. For this purpose, a custom library `gpu.dll` has been developed, which is written in **C++** using **CUDA** and allows efficient parallel computations.

- **Asynchronous computing** – The library enables fully asynchronous processing of neural network calculations, resulting in more efficient use of computational resources and reduced latency during training.

Thanks to these optimizations, MDNN can be used to effectively train deep neural networks even on large data sets.

### Requirements for computations via GPU

To enable neural network calculations on the GPU, the following components must be downloaded:
- A library **`gpu.dll`**
- **CUDA Toolkit**

Currently, the library only supports calculations on the **NVIDIA GPU**. Support for AMD and Intel graphics cards is in development.

### Activating calculations via GPU

The following code can be used to enable neural network computation via the GPU:
```csharp
GeneralNeuralNetworkSettings.calculationViaGpu = true;
```

### Using asynchronous functions

The library supports asynchronous processing of neural network training. Example of use:

```csharp
await model.Train.TrainLoopAsync(tensorInputDataset, tensorOutputDataset, 1000);
```

Each synchronous function has its equivalent asynchronous version, which allows for efficient parallel computations.

for example:
- **`TrainLoop()`** -> **`TrainLoopAsync()`**
- **`Fit()`** -> **`FitAsync()`**
- **`GetResults()`** -> **`GetResultsAsync()`**

## 📦 Production deployment

After the training process is complete and the model is saved in JSON format (e.g. using the `model. SaveAsJson("save")`) is followed by the **production deployment** phase. In this phase, the model is integrated into the target application or system, where it is used for inference – i.e. to make predictions based on new input data.

### Using the trained model

There`s no need for retraining to use the model in production. Just load it and then apply inputs to it:

```csharp
double[][] inputsDataset = new double[][] {...}; //input data

MDNN model = MDNN.LoadModel("Completed training.json");
Tensor inputTensor = new Tensor(Tensor.ConvertJaggedToMulti(inputsDataset));

model.GetResults(inputTensor);

```

## 👏 Support features

The library contains a number of supporting methods designed to facilitate work with neural networks.

For example, in the MDNN library you will find:
- functions for plotting graphs of training losses in individual epochs,
- Tools for clear display of outputs and statistics in the console,
- and many other useful tools that make model tuning and analysis more efficient.

These features significantly contribute to clarity, efficiency and comfort when working with neural networks.

### Tensor

The **`Tensor`** class serves as a universal data type for working with multidimensional arrays (*arrays*).
It allows efficient manipulation of data of any dimension and ensures its uniform representation across the entire library.

Internally, this class stores:
- the original multidimensional array `OriginalInput` (e.g. about the size `[5][5][5][5]`),
- equivalent one-dimensional array `Data` (e.g. size `[125]`) for faster calculations,
- and information about the shape of the original pattern in the form of a list of dimensions `Shape` (e.g. `[5, 5, 5, 5]`).

One of the key features of the class is the support of very easy and fast transformation of data into another dimension using the **Reshape(int[] newShape)** operation, which significantly increases the flexibility when working with different data structures.
Thanks to this structure, `Tensor` makes it easy to access elements, perform mathematical operations, and work efficiently with data in a neural network, regardless of its original dimension.

### Console outputs

The library contains a static ConsoleManager class that provides all outputs to the console:
- **`ShowModelInfo()`** – prints detailed information about the current model.
- **`ShowEpochInfo()`** – displays information about the current epoch during training.
- **`ShowScoreOfModel()`** – prints the achieved accuracy of the model.
- **`ErrorHandler()`** - handles and prints error messages, making it easier to diagnose and debug the model.

### `GeneralNeuralNetworkSettings`

It is a static support class that stores the default settings of the entire library for neural networks. For example, it contains a default trigger function, a lossy function, or an optimization algorithm.

The class also serves as a simple **dependency injection mechanism**, which allows management and passing of common dependencies between individual library components without the need to bind them tightly.

Attributes:
- **`default_output_activation_func`** (*Activation_func*)
Default trigger for the output layer (e.g. `Linear`).

- **`default_hidden_layers_activation_func`** (*Activation_func*)
Default activation function for hidden layers (e.g. `ReLU`).

- **`loss_func`** (*Loss*)
The default loss function used in training (e.g. `MSE` – mean square error).

- **`optimizer`**  (*Optimizer*)
Default optimization algorithm (e.g. `SGD` with a learning rate of `0.0001`).

- **`calculationViaGpu`** (*bool*)
Specifies whether the calculations should be performed on the GPU (`true`) or the CPU (`false`).

- **`SequenceTrain`** (*bool*)
Sequential training mode (e.g. for recurrent networks).

- **`modelInputSizeAndShape`** (*int[]*)
Defines the shape and size of the model`s input tensor.


### Creating charts

The library contains the **`GraphPlotter`** class, which is used to visualize the progress of neural network training. Its main purpose is to provide the user with a tool for monitoring the development of loss functions during the training process.
The class has a single method **`ShowLossGraph()`**, which generates a graph of training and validation loss (*TrainLoss* and *ValidLoss*) depending on the number of epochs. The resulting graph is automatically saved as an image named `loss.png` in the root directory of the application.

With this visual overview, the user can easily identify problems such as overfitting or undertraining and adjust the training parameters accordingly.

The publicly available **ScottPlot** library is used to plot the graph, which allows simple and clear generation of scientific graphs.
