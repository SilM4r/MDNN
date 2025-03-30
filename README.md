# MDNN
MDNN (my deep neural network) je knihovna pro tvorbu a trénování neuronových sítí v jazyce C#. Tato knihovna umožňuje snadné vytvoření vlastních modelů neuronových sítí a jejich trénování.

## 📌 Funkce
- Podpora různých typů vrstev, jako jsou například Dense,MaxPooling, CNN (konvoluční neuronové sítě) a RNN (rekurentní neuronové sítě)
- Možnost použití různých aktivačních funkcí,optimalizačních algoritmů,ztrátových funkcí.
- Snadná integrace s jinými projekty v C#
- Výpočet přes grafické karty (GPU)
- Možnost ukládání modelu do JSON a jeho následné načítání
- předem vytvořené trénovací smyčky.

## 🛠 Instalace
Knihovna se aplikuje do projektu pomocí souboru MDNN.dll. Tento soubor najdete ve složce projektu, nebo si můžete stáhnout celý repozitář a spustit jej – automaticky se vygeneruje nový soubor MDNN.dll.

## 🚀 Rychlé použití
Tohle je velmi jednoduchá ukázka toho jak je jednoduché vytvořit a natrénovat neuronovou sít. 

```csharp
using My_DNN.Layers;
using My_DNN.Layers.classes;
using My_DNN.Optimizers;
using My_DNN;
using My_DNN.Activation_functions;
using My_DNN.Loss_functions;

namespace MDNN_example
{
    internal class Program
    {
        static void Main(string[] args)
        {
            double[,] inputsDataset = new double[][] {...};  // input data 
            double[,] ouputDataset = new double[][] {...};   /// current output data

            Layer outputLayer = new Dense(1, new Linear()); // outputLayer setting
            Optimizer optimizer = new SGD(0.01); // optimizer setting
            Loss loss = new MSE(); //loss setting setting

            uint epoch = 1000;

            // setting the model structure
            MDNN model = new MDNN(outputLayer, optimizer, loss);

            model.Layers.Add(new Dense(1, new ReLu()));  // adding a hidden layer

            Tensor tensorInputDataset = new Tensor(inputsDataset);
            Tensor tensorOutputDataset = new Tensor(ouputDataset);

            // model train 
            model.Train.TrainLoop(tensorInputDataset, tensorOutputDataset, epoch,1);
            //save model
            model.SaveAsJson("save");
        }
    }
}
```

## ⏱️ Nejjednoduší začátek 
zde je ukázka minimálních věcích které je potřeba nastavit pro trénování, zbylé nastaveí se nastaví na defaultní hodonty které jsou co možná nejvíc univerzální aby šli použít na jakákoliv úkoly.
```csharp
using My_DNN.Layers;
using My_DNN.Layers.classes;
using My_DNN.Optimizers;
using My_DNN;
using My_DNN.Activation_functions;
using My_DNN.Loss_functions;

namespace MDNN_example
{
    internal class Program
    {
        static void Main(string[] args)
        {
            //  input data and current output data
            double[][] inputsDataset = new double[][] {...};
            double[][] ouputDataset = new double[][] {...}; 

            // outputLayer setting
            Layer outputLayer = new Dense(1, new Linear()); 

            // setting the model structure
            MDNN model = new MDNN(outputLayer);

            // model train 
            uint epoch = 1000;
            model.Train.SimpleTrainLoop(inputsDataset, ouputDataset, epoch);
        }
    }
}
```


