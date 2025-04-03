**# MDNN (My Deep Neural Network)

MDNN (My Deep Neural Network) je knihovna pro návrh a trénování neuronových sítí v jazyce C#. Umožňuje snadnou tvorbu a konfiguraci modelů neuronových sítí, jejich trénování a následnou inferenci.

## 📌 Klíčové vlastnosti

- Podpora různých typů vrstev, včetně:
  - **Dense** (plně propojené vrstvy)
  - **MaxPooling**
  - **Konvoluční neuronové sítě (CNN)**
  - **Rekurentní neuronové sítě (RNN)**
- Možnost využití široké škály:
  - Aktivačních funkcí
  - Optimalizačních algoritmů
  - Ztrátových funkcí
- Snadná integrace do projektů v C#
- Podpora akcelerace výpočtů pomocí GPU
- Ukládání a načítání modelů ve formátu JSON
- Předdefinované trénovací smyčky pro zjednodušení procesu trénování

## 🛠 Instalace

MDNN je distribuována jako dynamická knihovna **MDNN.dll**. Pro její použití je nutné:

1. Přidat soubor **MDNN.dll** do projektu.
2. Zahrnout příslušné jmenné prostory ve zdrojovém kódu.
3. Alternativně lze stáhnout celý repozitář a spustit sestavení, které automaticky vygeneruje nový soubor **MDNN.dll**.

## 🚀 Rychlý start

Níže je uveden jednoduchý příklad použití knihovny MDNN pro vytvoření a trénování neuronové sítě.

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
            double[,] inputsDataset = new double[][] {...};  // Vstupní data
            double[,] outputDataset = new double[][] {...};  // Odpovídající výstupní data

            Layer outputLayer = new Dense(1, new Linear());  // Konfigurace výstupní vrstvy
            Optimizer optimizer = new SGD(0.01);  // Nastavení optimalizačního algoritmu
            Loss loss = new MSE();  // Definice ztrátové funkce

            uint epoch = 1000;

            // Inicializace modelu
            MDNN model = new MDNN(outputLayer, optimizer, loss);
            model.Layers.Add(new Dense(1, new ReLu()));  // Přidání skryté vrstvy

            Tensor tensorInputDataset = new Tensor(inputsDataset);
            Tensor tensorOutputDataset = new Tensor(outputDataset);

            // Trénování modelu
            model.Train.TrainLoop(tensorInputDataset, tensorOutputDataset, epoch, 1);
            
            // Uložení modelu
            model.SaveAsJson("save");
        }
    }
}
```

## ⚙ Konfigurace modelu

Knihovna **MDNN** umožňuje snadné nastavení architektury neuronové sítě, včetně výstupní vrstvy, optimalizačního algoritmu a ztrátové funkce.
Model je inicializován pomocí třídy **MDNN**, která slouží jako centrální objekt pro manipulaci s neuronovou sítí:
```csharp
MDNN model = new MDNN(outputLayer, optimizer, loss);
```
## Parametry konstruktoru

- **`outputLayer`** *(povinný parametr)* – objekt typu **Layer**, představující výstupní vrstvu sítě.
- **`optimizer`** *(volitelný parametr)* – objekt typu **Optimizer**, který specifikuje optimalizační algoritmus. Pokud není zadán, výchozí hodnota je **SGD(0.0001)**.
- **`loss`** *(volitelný parametr)* – objekt reprezentující ztrátovou funkci. Výchozí hodnota je **MSE()**.

podporované optimizéry: SGD,ADAM,Momentum
podporované zrátové funkce: MSE, Cross Entropy

Případně knihovna podporuje i tvorbu vlastních optimalizátorů a ztrátových funkcí – stačí zdědit odpovídající mateřskou třídu například **`loss`**.

## 📌 Přidávání vrstev do modelu
Další vrstvy lze do modelu přidávat pomocí třídy **`Layers`**:

```csharp
model.Layers.Add(new Dense(64, new ReLu()));  // Přidání skryté vrstvy s 64 neurony a ReLU aktivací
model.Layers.Add(new Dense(32, new Sigmoid()));  // Další vrstva s 32 neurony a sigmoid aktivací
```

Kromě funkce **`Add()`** obsahuje třída **`Layers`** také metody pro odebírání nebo přidávání vrstev na konkrétní pozici a řadu dalších funkcí pro manipulaci s vrstvami.

Knihovna **MDNN** podporuje následující vrstvy:
- `Dense()`
- `RNN()`
- `Conv()`
- `MaxPool()`
- WIP: transformer vrstva respektive attention Layer

V případě potřeby lze vytvořit i vlastní specializovanou vrstvu. Stačí zdědit jednu z následujících abstraktních tříd:
- `Layer`
- `LayerBasedOnNeurons`
- `LayerWithUntrainedParameters`

Poté je nutné implementovat všechny jejich abstraktní metody. Jakmile je nová vrstva definována, lze ji přidat do modelu a použít při trénování.

## 🎯 Trénování modelu
modle se trénuje pomocí třídy Train, která obsahuje veškeré potřebné metody pro řízení trénování. Uživatel má možnost volit mezi čtyřmi metodami trénování podle požadované míry kontroly nad učením modelu. 
-	**`TrainLoop()`**
-	**`SimpleTrainLoop()`** 
-	**`Fit()`** 
-	**`BackPropagation() a FeedForward()`** 






