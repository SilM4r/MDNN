# MDNN (My Deep Neural Network)

MDNN je knihovna pro návrh a trénování neuronových sítí v jazyce C#. Umožňuje snadnou tvorbu a konfiguraci modelů neuronových sítí, jejich trénování a následnou integraci do aplikací.

## Obsah

- [Klíčové vlastnosti](#klíčové-vlastnosti)
- [Instalace](#instalace)
- [Rychlý start](#rychlý-start)
- [Konfigurace modelu](#konfigurace-modelu)
- [Přidávání vrstev](#přidávání-vrstev)
- [Trénování modelu](#trénování-modelu)
- [GPU akcelerace a asynchronní výpočty](#gpu-akcelerace-a-asynchronní-výpočty)
- [Ukládání a načítání modelů](#ukládání-a-načítání-modelů)
- [Podpůrné nástroje](#podpůrné-nástroje)
- [Testy](#testy)

## Klíčové vlastnosti

Podporované typy vrstev:

- Dense (plně propojené)
- Konvoluční (Conv)
- Max pooling (MaxPool)
- Rekurentní (RNN)

Další možnosti:

- Škála aktivačních funkcí, optimizérů a ztrátových funkcí
- Snadná integrace do projektů v C#
- Volitelný výpočet na GPU
- Asynchronní trénování a inference
- Ukládání a načítání modelů ve formátu JSON
- Předdefinované trénovací smyčky

## Instalace

MDNN je distribuována jako dynamická knihovna `MDNN.dll`. Pro použití:

1. Přidejte do projektu referenci na `MDNN.dll`.
2. Zahrňte příslušné jmenné prostory ve zdrojovém kódu.

Alternativně stáhněte repozitář a sestavte ho; sestavení vygeneruje nový soubor `MDNN.dll`.

## Rychlý start

Minimální příklad, který vytvoří a natrénuje síť:

```csharp
using My_DNN;
using My_DNN.Layers;
using My_DNN.Layers.classes;
using My_DNN.Optimizers;
using My_DNN.Loss_functions;
using My_DNN.Activation_functions;

double[][] inputsDataset = { /* vstupní vzorky */ };
double[][] outputDataset = { /* odpovídající cíle */ };

Layer outputLayer = new Dense(1, new Linear()); // výstupní vrstva
Optimizer optimizer = new SGD(0.01);            // optimalizační algoritmus
Loss loss = new MSE();                          // ztrátová funkce

uint epochs = 1000;

MDNN model = new MDNN(outputLayer, optimizer, loss);
model.Layers.Add(new Dense(8, new ReLu()));     // skrytá vrstva

model.Train.TrainLoop(inputsDataset, outputDataset, epochs, 1);

model.SaveAsJson("save");
```

## Konfigurace modelu

Model se vytváří přes třídu `MDNN`, která je centrálním objektem pro práci se sítí:

```csharp
MDNN model = new MDNN(outputLayer, optimizer, loss);
```

Parametry konstruktoru:

- `outputLayer` (povinný) – objekt `Layer` představující výstupní vrstvu.
- `optimizer` (volitelný) – objekt `Optimizer`. Výchozí hodnota je `SGD(0.0001)`.
- `loss` (volitelný) – ztrátová funkce. Výchozí hodnota je `MSE()`.

Podporované optimizéry: `SGD`, `Adam`, `Momentum`.
Podporované ztrátové funkce: `MSE`, `CrossEntropy`.

Každý model vlastní svou konfiguraci, takže v jednom procesu může existovat více nezávislých modelů.

Vlastní optimizér nebo ztrátovou funkci lze vytvořit zděděním odpovídající bázové třídy (například `Optimizer` nebo `Loss`) a implementací jejích abstraktních členů.

Poznámka k `CrossEntropy`: je určená pro použití se softmax výstupní vrstvou. Počítá fúzovaný gradient softmax + kategorická cross-entropy (`výstup − cíl`), což je numericky stabilní forma. Použití `CrossEntropy` bez softmax výstupní vrstvy vyhodí výjimku.

## Přidávání vrstev

Vrstvy se přidávají přes vlastnost `Layers`:

```csharp
model.Layers.Add(new Dense(64, new ReLu()));    // skrytá vrstva, 64 neuronů, ReLU
model.Layers.Add(new Dense(32, new Sigmoid())); // skrytá vrstva, 32 neuronů, sigmoid
```

Kromě `Add()` nabízí `Layers` také:

- `Insert()` – vloží vrstvu na danou pozici
- `RemoveAt()` – odebere vrstvu na dané pozici
- `OutputLayerActivationFunc()` – nastaví novou výstupní aktivační funkci
- `ClearAllLayersAndSetNewOutputLayer()` – odebere všechny vrstvy a nastaví novou výstupní vrstvu

Podporované vrstvy:

- `Dense()`
- `RNN()`
- `Conv()`
- `MaxPool()`

Pokud se konstruktoru vrstvy nezadá aktivační funkce, použije se výchozí aktivace pro skryté vrstvy (ReLU). U výstupních vrstev zadejte aktivaci explicitně (například `Linear` pro regresi nebo `Softmax` pro klasifikaci).

Vlastní vrstvu lze vytvořit zděděním jedné z abstraktních tříd `Layer`, `LayerBasedOnNeurons` nebo `LayerWithUntrainedParameters` a implementací jejích abstraktních členů.

Dostupné aktivační funkce: `Linear`, `ReLu`, `Leak_ReLu`, `Sigmoid`, `Tanh`, `Softmax`.

## Trénování modelu

Trénování řídí třída `Train`. K dispozici jsou čtyři úrovně kontroly, od plně automatické po plně ruční.

### `TrainLoop()`

Nejúplnější trénovací procedura. Poskytuje:

- Automatické ukládání modelu s nejlepší validací (early-stopping checkpoint)
- Automatické zamíchání a rozdělení datasetu na trénovací, validační a testovací část
- Průběžný výpis do konzole
- Detekci hodnot `NaN`
- Automatické vykreslení grafu ztráty napříč epochami

Parametry:

- `Array inputs_values` (povinný) – vstupní dataset. Každý řádek je jeden trénovací vzorek.
- `Array current_output_values` (povinný) – odpovídající cíle.
- `uint number_of_epoch` (povinný) – počet trénovacích epoch.
- `uint size_of_mini_batch` (volitelný, výchozí `1`) – velikost minibatche.
- `bool isSequence` (volitelný, výchozí `false`) – nastavte na `true` pro sekvenční data (například časové řady).

### `SimpleTrainLoop()`

Zjednodušená trénovací smyčka s checkpointingem, výpisem do konzole a detekcí `NaN`, bez rozdělení datasetu a vykreslování grafu z `TrainLoop()`.

Parametry:

- `double[][] inputs_values` (povinný)
- `double[][] current_output_values` (povinný)
- `uint number_of_epoch` (povinný)
- `uint size_of_mini_batch` (volitelný, výchozí `1`)

### `Fit()` a `UpdateParams()`

Středně pokročilý přístup, který umožňuje napsat vlastní trénovací smyčku. `Fit()` provede dopředný výpočet i zpětnou propagaci, ale gradienty pouze akumuluje; `UpdateParams()` akumulované gradienty aplikuje. Volání obou funkcí bezprostředně po sobě odpovídá trénování po jednom vzorku; akumulace více `Fit()` před jedním `UpdateParams()` odpovídá trénování s minibatchem.

```csharp
Random rnd = new Random();
double[][] inputsDataset = { /* vstupní data */ };
double[][] currentOutputDataset = { /* cíle */ };

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

### `FeedForward()` a `BackPropagation()`

Nejgranulárnější přístup, který dělí `Fit()` na samostatný dopředný výpočet (`FeedForward()`) a zpětnou propagaci (`BackPropagation()`). Dává plnou kontrolu nad jednotlivými kroky trénování, což je vhodné pro výzkum nebo pokročilé optimalizace. Po zpětné propagaci je nutné zavolat `UpdateParams()`.

`BackPropagation()` má dvě přetížení: jedno bere cílové hodnoty (a gradienty vrstev spočítá interně) a druhé bere předpočítané gradienty jednotlivých vrstev.

## GPU akcelerace a asynchronní výpočty

### GPU

Výpočty sítě mohou volitelně běžet na NVIDIA GPU přes doprovodnou knihovnu `gpu.dll` (napsanou v C++ / CUDA). Vyžaduje to CUDA Toolkit a `gpu.dll`. GPU je nastavení per-model:

```csharp
model.Context.CalculationViaGpu = true;
```

Podpora aktuálně cílí pouze na NVIDIA GPU.

### Asynchronní výpočty

Každá synchronní metoda má asynchronní protějšek, například:

- `TrainLoop()` – `TrainLoopAsync()`
- `Fit()` – `FitAsync()`
- `GetResults()` – `GetResultsAsync()`

```csharp
await model.Train.TrainLoopAsync(inputsDataset, outputDataset, 1000);
```

## Ukládání a načítání modelů

Po natrénování lze model uložit do JSON a později načíst pro inference. K použití uloženého modelu není třeba opětovné trénování.

```csharp
model.SaveAsJson("save"); // zapíše save.json

MDNN loaded = MDNN.LoadModel("save.json");
Tensor input = new Tensor(Tensor.ConvertJaggedToMulti(inputsDataset));
Tensor result = loaded.GetResults(input);
```

## Podpůrné nástroje

### Tensor

`Tensor` je univerzální datový typ pro vícerozměrná pole. Uchovává:

- původní vícerozměrné pole (`OriginalInput`),
- ekvivalentní jednorozměrné pole (`Data`) pro rychlejší výpočty,
- a tvar jako seznam rozměrů (`Shape`).

Podporuje přetváření přes `Reshape(int[] newShape)` a pohodlný přístup k prvkům i převody mezi jagged a vícerozměrnými poli.

### Konzolové výstupy

Statická třída `ConsoleControler` zajišťuje výstupy do konzole:

- `ShowModelInfo()` – vypíše podrobné informace o modelu
- `ShowEpochInfo()` – vypíše informace o aktuální epoše během trénování
- `ShowScoreOfmodel()` – vypíše přesnost modelu
- `ErrorHandler()` – vypíše chybové hlášky

### NetworkContext

Každý model vlastní `NetworkContext` (`model.Context`), který drží jeho konfiguraci za běhu: ztrátovou funkci, optimizér, tvar vstupu, příznak sekvenčního trénování a příznak GPU. Protože je tento stav per-model, dva modely v jednom procesu se navzájem neovlivňují.

`GeneralNeuralNetworkSettings` drží už jen procesní výchozí hodnoty (výchozí aktivační funkce a sdílený generátor náhodných čísel).

### Tvorba grafů

Třída `GraphPlotter` vizualizuje průběh trénování. Její metoda `ShowLossGraph()` vygeneruje graf trénovací a validační ztráty v závislosti na počtu epoch a uloží ho jako `loss.png` do kořenového adresáře aplikace. To usnadňuje odhalení přeučení (overfitting) nebo nedostatečného trénování. K vykreslení se používá knihovna ScottPlot.

## Testy

Repozitář obsahuje xUnit testovací projekt (`MDNN.Tests`) s numerickými gradient checky pro každou vrstvu, jednotkovými testy aktivačních, ztrátových a optimalizačních tříd a end-to-end trénovacími smoke testy. Spuštění:

```
dotnet test
```

Gradient checky slouží jako regresní pojistka: jakákoli změna, která rozbije matematiku, je automaticky odhalena.
