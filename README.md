**MDNN (My Deep Neural Network)**
==============

MDNN (My Deep Neural Network) je knihovna pro návrh a trénování neuronových sítí v jazyce C#. Umožňuje snadnou tvorbu a konfiguraci modelů neuronových sítí, jejich trénování a následnou integraci do projektů.

## 📚 **Obsah**
- [📌 Klíčové vlastnosti](#-klíčové-vlastnosti)  
- [🛠 Instalace](#-instalace)  
- [🚀 Rychlý start](#-rychlý-start)  
- [⚙ Konfigurace modelu](#-konfigurace-modelu)  
- [📌 Přidávání vrstev do modelu](#-přidávání-vrstev-do-modelu)  
- [🎯 Trénování modelu](#-trénování-modelu)  
- [⏳ Optimalizace](#-optimalizace)
- [📦 Produkční nasazení](#-produkční-nasazení)


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

1. Přidat soubor **MDNN.dll** do projektu. (přidat novou závyslot do projektu)
2. Zahrnout příslušné jmenné prostory ve zdrojovém kódu.

Alternativně lze stáhnout celý repozitář a spustit sestavení, které automaticky vygeneruje nový soubor **MDNN.dll**.

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
### Parametry konstruktoru

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
- **`Insert()`**
- **`RemoveAt()`**
- **`OutputLayerActivationFunc()`** - nastavý novou výstupní aktivační funkci
- **`ClearAllLayersAndSetNewOutputLayer`** - vymaže všechny vrstvy a nastavý novou výstupní vrstvu


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
model se trénuje pomocí třídy Train, která obsahuje veškeré potřebné metody pro řízení trénování. Uživatel má možnost volit mezi čtyřmi metodami trénování podle požadované míry kontroly nad učením modelu. 
-	**`TrainLoop()`**
-	**`SimpleTrainLoop()`** 
-	**`Fit()`** a **`UpdateParams()`** 
-	**`BackPropagation()`** a **`FeedForward()`**


### **`TrainLoop()`**
Tato metoda představuje hlavní a zároveň nejpokročilejší trénovací proceduru v knihovně. Zahrnuje kompletní trénovací smyčku (train loop) a poskytuje následující pokročilé funkce:

- Automatické ukládání modelu s nejlepší validací (tzv. early stopping checkpoint).
- Automatické míchání a rozdělení datasetu na trénovací, validační a testovací části.
- Průběžný výpis informací o průběhu trénování (např. aktuální epochy, metriky) do konzole.
- Detekce chyb, jako je výskyt hodnot typu NaN.
- Automatické vykreslení grafu průběhu ztrátové funkce (loss) napříč epochami.

Parametry:
- **`Tensor inputs_values`** – *povinný parametr*  
  Vstupní dataset ve formátu tenzoru. Každý řádek odpovídá jednomu trénovacímu vzorku.

- **`Tensor current_output_values`** – *povinný parametr*  
  Odpovídající výstupy (labely) pro vstupní data, rovněž ve formátu tenzoru.

- **`uint number_of_epoch`** – *povinný parametr*  
  Určuje počet trénovacích epoch.

- **`uint size_of_mini_batch`** – *nepovinný parametr*  
  Velikost minibatche používané během trénování. Pokud není specifikováno, použije se výchozí hodnota `1`.

- **`bool isSequence`** – *nepovinný parametr*  
  Pokud je nastaveno na `true`, model bude očekávat sekvenční vstupní data (např. časové řady, video nebo jiná sekvenční data). Výchozí hodnota je `false`.

### **`SimpleTrainLoop()`**
Jedná se o zjednodušenou verzi metody **`TrainLoop()`** Zahrnuje kompletní trénovací smyčku (train loop) a poskytuje následující funkce:

- Automatické ukládání modelu s nejlepší validací (tzv. early stopping checkpoint).
- Průběžný výpis informací o průběhu trénování (např. aktuální epochy, metriky) do konzole.
- Detekce chyb, jako je výskyt hodnot typu NaN.

Parametry:
- **`double[][] inputs_values`** – *povinný parametr*  
  Vstupní dataset ve formátu double[][]. Každý řádek odpovídá jednomu trénovacímu vzorku.

- **`double[][] current_output_values`** – *povinný parametr*  
  Odpovídající výstupy (labely) pro vstupní data, rovněž ve formátu double[][].

- **`uint number_of_epoch`** – *povinný parametr*  
  Určuje počet trénovacích epoch.

- **`uint size_of_mini_batch`** – *nepovinný parametr*  
  Velikost minibatche používané během trénování. Pokud není specifikováno, použije se výchozí hodnota `1`.

### **`Fit()`** a **`UpdateParams()`** 
Jedná se o středně pokročilý přístup k trénování modelů, který umožňuje implementaci vlastní trénovací smyčky (training loop). Tento přístup neposkytuje předpřipravené funkcionality, avšak uživatel má k dispozici sadu podpůrných tříd, které lze využít pro vytvoření doplňkových komponent, jako je například výpis informací o trénování do konzole, vizualizace trénovacích ztrát pomocí grafů, nebo implementace testovacích procedur pro vyhodnocení modelu.

Tato metoda je tvořena dvěma funkcemi: **`Fit()`** a **`UpdateParams()`**.  
Funkce **`Fit()`** provádí jak výpočet výstupu (tzv. *feedforward*), tak i zpětnou propagaci chyby (*backpropagation*), avšak bez okamžité aktualizace modelových parametrů – místo toho dochází k akumulaci gradientů.

Aktualizace parametrů na základě akumulovaných gradientů je provedena až při volání funkce **`UpdateParams()`**. Tento přístup odpovídá trénování pomocí *minibatch*, které může vést k efektivnějšímu a stabilnějšímu učení.

V případě, že uživatel nechce využívat minibatch režim, postačí volat obě funkce bezprostředně po sobě.

```csharp
 Random rnd = new Random();

 double[][] inputsDataset = new double[][] {...};  // input data 
 double[][] currentOutputDataset = new double[][] {...};  // currentOutput data 

 // Inicializace modelu
 MDNN model = new MDNN(new Dense(3), new Adam(0.001));

 // počet epoch a velikost minibatche
 int number_of_epoch = 5000;
 int size_of_miniBatch = 16;

 // počet všech prvků v datasetu 
 int number_of_element_intDataset = inputsDataset.Length;

 // hlavní trénovací smyčka
 for (int i = 0; i < number_of_epoch; i++)
 {
     // sekundární smyčka minibatch
     for (int j = 0; j < size_of_miniBatch; j++)
     {
         int num = rnd.Next(number_of_element_intDataset);

         double[] inputs = inputsDataset[num];
         double[] output = currentOutputDataset[num];

         // výpočet na jednom konkrétním prvku který je náhodně zvolen z celého datasetu
         model.Train.Fit(new Tensor(inputs),new Tensor(output));
     }

     // aktualizace trénovacích parametrů
     model.Train.UpdateParams();
 }

 // podpůrná funkce která otestuje celý dataset na natrénovaném datsetu
 model.Train.TestNeuralNetwork(new Tensor(Tensor.ConvertJaggedToMulti(inputsDataset)), new Tensor(Tensor.ConvertJaggedToMulti(currentOutputDataset)));
```
Parametry:
**`Fit()`** :
- **`Tensor inputs_values`** – *povinný parametr*  
  Vstupní dataset ve formátu tenzoru. Každý řádek odpovídá jednomu trénovacímu vzorku.

- **`Tensor target_values`** – *povinný parametr*  
  Odpovídající výstupy (labely) pro vstupní data, rovněž ve formátu tenzoru.

**`UpdateParams()`** - nemá žádné paramtery

### **`BackPropagation()`** a **`FeedForward()`**
Jedná se o nejpokročilejší metodu trénování, která poskytuje maximální míru kontroly nad jednotlivými fázemi učení. Na rozdíl od přístupu využívajícího funkce **`Fit()`** a **`UpdateParams()`** je zde metoda **`Fit()`** rozdělena do dvou samostatných funkcí: **`FeedForward()`** a **`BackPropagation()`**. Funkce **`FeedForward()`** provádí dopředný výpočet neuronové sítě, zatímco **`BackPropagation()`** zajišťuje zpětnou propagaci chyby a výpočet gradientů. Tento přístup umožňuje detailní manipulaci s jednotlivými kroky trénovacího procesu, což je vhodné zejména pro výzkumné účely nebo pokročilé optimalizace.

**Upozornění:** Pro samotnou aktualizaci modelových parametrů je i v tomto případě nutné následně zavolat metodu **`UpdateParams()`**.

```csharp
 Random rnd = new Random();

 double[][] inputsDataset = new double[][] {...};  // input data 
 double[][] currentOutputDataset = new double[][] {...};  // currentOutput data 

 // Inicializace modelu
 MDNN model = new MDNN(new Dense(3), new Adam(0.001));

 // počet epoch a velikost minibatche
 int number_of_epoch = 5000;
 int size_of_miniBatch = 16;

 // počet všech prvků v datasetu 
 int number_of_element_intDataset = inputsDataset.Length;

 // hlavní trénovací smyčka
 for (int i = 0; i < number_of_epoch; i++)
 {
     // sekundární smyčka minibatch
     for (int j = 0; j < size_of_miniBatch; j++)
     {
         int num = rnd.Next(number_of_element_intDataset);

         double[] inputs = inputsDataset[num];
         double[] output = currentOutputDataset[num];

         // dopředný výpočet 
         model.Train.FeedForward(new Tensor(Tensor.ConvertJaggedToMulti(inputs)));

         // výpočet gradientů (podpůrná meotda)
         Tensor[] gradients = Gradient.GetGradients(new Tensor(Tensor.ConvertJaggedToMulti(inputs)), model);

         // zpětná propagace gradientů
         model.Train.BackPropagation(gradients);
     }

     // aktualizace trénovacích parametrů
     model.Train.UpdateParams();
 }

 // podpůrná funkce která otestuje model na natrénovaném datsetu
 model.Train.TestNeuralNetwork(new Tensor(Tensor.ConvertJaggedToMulti(inputsDataset)), new Tensor(Tensor.ConvertJaggedToMulti(currentOutputDataset)));
```
Parametry metod:

**`FeedForward()`**
- **`Tensor inputs_values`** – *povinný parametr*  
  Vstupní data ve formátu tenzor.

**`BackPropagation()`**

Metoda **`BackPropagation()`** má dvě přetížení:

1. **První přetížení:**
   - **`Tensor[] layer_gradients`** – *povinný parametr*  
     Pole gradientů jednotlivých vrstev ve formátu tenzoru. Slouží k ručně řízené zpětné propagaci.

2. **Druhé přetížení:**
   - **`Tensor target_values`** – *povinný parametr*  
     Cílové výstupní hodnoty odpovídající vstupním datům, rovněž ve formátu tenzoru. V tomto přetížení metoda interně spočítá příslušné hodnoty `layer_gradients`.

## ⏳ Optimalizace

Knihovna **MDNN** podporuje efektivní optimalizaci výpočtů neuronové sítě. Mezi hlavní optimalizační techniky patří:
- **Využití GPU** – Výpočty neuronové sítě lze provádět na grafických procesorech, což výrazně urychluje trénování modelů, zejména při práci s velkým množstvím dat. Pro tento účel byla vyvinuta vlastní knihovna `gpu.dll`, která je napsaná v **C++** s využitím **CUDA** a umožňuje efektivní paralelní výpočty.
- **Asynchronní výpočty** – Knihovna umožňuje plně asynchronní zpracování výpočtů neuronové sítě, což vede k efektivnějšímu využití výpočetních zdrojů a snížení latence během trénování.

Díky těmto optimalizacím lze s MDNN efektivně trénovat hluboké neuronové sítě i na rozsáhlých datových sadách.

### Požadavky na výpočty přes GPU

Pro umožnění výpočtů neuronové sítě na GPU je nutné stáhnout následující komponenty:
- Knihovnu **`gpu.dll`**
- **CUDA Toolkit**

Aktuálně knihovna podporuje výpočty pouze na **NVIDIA GPU**. Podpora pro **AMD** a **Intel** grafické karty je ve vývoji.

### Aktivace výpočtů přes GPU

Pro zapnutí výpočtu neuronové sítě přes GPU lze použít následující kód:
```csharp
GeneralNeuralNetworkSettings.calculationViaGpu = true;
```

### Použití asynchronních funkcí

Knihovna podporuje asynchronní zpracování trénování neuronové sítě. Ukázka použití:
```csharp
await model.Train.TrainLoopAsync(tensorInputDataset, tensorOutputDataset, 1000);
```
Každá synchronní funkce má svou ekvivalentní asynchronní verzi, což umožňuje efektivní paralelní výpočty.
například:

- **`TrainLoop()`**  -> **`TrainLoopAsync()`** 
- **`Fit()`**  -> **`FitAsync()`** 
- **`GetResults()`**  -> **`GetResultsAsync()`**

## 📦 Produkční nasazení

Po dokončení procesu trénování a uložení modelu ve formátu JSON (např. pomocí metody `model.SaveAsJson("save")`) následuje fáze **produkčního nasazení**. V této fázi je model integrován do cílové aplikace nebo systému, kde slouží k inference – tedy k provádění predikcí na základě nových vstupních dat.

### Využití natrénovaného modelu

Pro použití modelu v produkčním prostředí není třeba opětovné trénování. Stačí ho načíst a následně na něj aplikovat vstupy:

```csharp
double[][] inputsDataset = new double[][] {...};  // input data 

MDNN model = MDNN.LoadModel("Completed training.json");

Tensor inputTensor = new Tensor(Tensor.ConvertJaggedToMulti(inputsDataset));

model.GetResults(inputTensor);
```






