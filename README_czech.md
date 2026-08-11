# MDNN (My Deep Neural Network)

MDNN je knihovna pro návrh a trénování neuronových sítí v jazyce C#. Umožňuje snadnou tvorbu a konfiguraci modelů neuronových sítí, jejich trénování a následnou integraci do aplikací.

## Obsah

- [Klíčové vlastnosti](#klíčové-vlastnosti)
- [Instalace](#instalace)
- [Rychlý start](#rychlý-start)
- [Příklady](#příklady)
- [Konfigurace modelu](#konfigurace-modelu)
- [Přidávání vrstev](#přidávání-vrstev)
- [Trénování modelu](#trénování-modelu)
- [GPU akcelerace a asynchronní výpočty](#gpu-akcelerace-a-asynchronní-výpočty)
- [Ukládání a načítání modelů](#ukládání-a-načítání-modelů)
- [Podpůrné nástroje](#podpůrné-nástroje)
- [Jak to funguje uvnitř](#jak-to-funguje-uvnitř)
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

## Příklady

Rychlý start výše je kompletní funkční příklad. Rozsáhlejší — MNIST s CNN v LeNet stylu,
IDX loadery, held-out testovací sadou a early stoppingem — je v projektu `mdnn_test`
vedle tohoto repozitáře.

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

**Jedna epocha = jeden plný průchod trénovacím setem.** Každou epochu se pořadí vzorků
znovu zamíchá a data se projdou po minibatchích, s jedním krokem optimizeru na dávku.
`number_of_epoch` tedy počítá průchody daty, ne kroky optimizeru — ty najdete
v `Train.OptimizerSteps`.

Parametry:

- `Array inputs_values` (povinný) – vstupní dataset. Každý řádek je jeden trénovací vzorek.
- `Array current_output_values` (povinný) – odpovídající cíle.
- `uint number_of_epoch` (povinný) – počet trénovacích epoch.
- `uint size_of_mini_batch` (volitelný, výchozí `1`) – velikost minibatche.
- `bool isSequence` (volitelný, výchozí `false`) – nastavte na `true` pro sekvenční data (například časové řady).

Tohle přetížení je zkratka: rozdělí předaná data podle poměrů a natrénuje. Když už máte
vlastní rozdělení, předejte sady napřímo a použijte bezdatové přetížení:

```csharp
model.Train.SetDatasets(
    train: new LabeledData(trainX, trainY),
    valid: new LabeledData(validX, validY),
    test:  new LabeledData(testX, testY));   // test je volitelný

model.Train.TrainLoop(numberOfEpoch: 50, sizeOfMiniBatch: 32);
```

Co nedodáte, to se ukrojí:

| Volání | Chování |
|---|---|
| `SetDatasets(train)` | valid i test se ukrojí z trainu (0,7 / 0,15 / 0,15) |
| `SetDatasets(train, valid)` | test se ukrojí z validu; train zůstane celý |
| `SetDatasets(train, null, test)` | valid se ukrojí z trainu; testovací sada se nikdy nedotkne |
| `SetDatasets(train, valid, test)` | nic se nekrájí |

Z testovací sady se nekrájí nikdy — je to finální nezaujatý odhad. Nastavením
`TestNeuralNetworkAfterTraining = false` navíc zabráníte i tomu, aby se testovací sada
ukrajovala z vaší validační; ta pak zůstane celá.

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
double[][] inputsDataset = { /* vstupní data */ };
double[][] currentOutputDataset = { /* cíle */ };

MDNN model = new MDNN(new Dense(3), new Adam(0.001), seed: 42);
// Dense(3) má tři výstupní neurony, takže každý cíl musí mít tři hodnoty — počet cílů musí odpovídat počtu neuronů výstupní vrstvy.

int numberOfEpochs = 50;
int miniBatchSize = 16;

Random rnd = new Random(42);
int[] order = Enumerable.Range(0, inputsDataset.Length).ToArray();

for (int epoch = 0; epoch < numberOfEpochs; epoch++)
{
    // jednou za epochu zamíchat a projít CELÝ set — přesně to dělá TrainLoop
    for (int i = order.Length - 1; i > 0; i--)
    {
        int j = rnd.Next(i + 1);
        (order[i], order[j]) = (order[j], order[i]);
    }

    for (int start = 0; start < order.Length; start += miniBatchSize)
    {
        int end = Math.Min(start + miniBatchSize, order.Length);

        for (int k = start; k < end; k++)
        {
            int num = order[k];
            model.Train.Fit(new Tensor(inputsDataset[num]), new Tensor(currentOutputDataset[num]));
        }

        model.Train.UpdateParams();   // jeden krok optimizeru na dávku
    }
}
```

Pozor: ruční smyčka žádné epochy nemá — `Train.CurrentEpoch` zůstane `0` a posouvá se
jen `Train.OptimizerSteps`.

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

MDNN loaded = MDNN.LoadModel("save.json");   // "save" funguje taky — přípona je volitelná
Tensor input = new Tensor(Tensor.ConvertJaggedToMulti(inputsDataset));
Tensor result = loaded.GetResults(input);
```

Soubor nese `FormatVersion`, seed, se kterým model vznikl, čas uložení a SHA-256 kontrolní
součet dat modelu:

```json
{
  "FormatVersion": 1,
  "Checksum": "sha256:7a54f121...",
  "Model": { "Seed": 42, "SavedAtUtc": "2026-08-11T11:08:52Z", ... }
}
```

Nesouhlasný součet vyhodí `ModelFileCorruptedException`. Pozor na to, co dokazuje a co ne:
detekuje **poškozený nebo omylem upravený** soubor, ne cíleně pozměněný — kdo soubor upraví,
přepočítá si i hash. Soubory ze starších verzí (bez `FormatVersion`) se načtou dál, jen bez
ověření.

`LoadModel(path, seed)` umožní nastavit seed pro další trénování; bez něj se obnoví seed
uložený v souboru.

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

## Jak to funguje uvnitř

Tato sekce popisuje, co se děje pod kapotou. Pro použití knihovny ji není nutné číst, ale vysvětluje návrh a hodí se při rozšiřování knihovny.

### Objektový model

```
MDNN (model)
├── NetworkContext        konfigurace za běhu per-model (loss, optimizér, tvar vstupu, příznaky)
├── LayerManager (Layers) seřazený seznam vrstev + odvození tvaru vstupu
│   └── Layer             Dense · Conv · MaxPool · RNN
│       └── Neuron        váhy + bias + vlastní optimizér   (Dense, RNN)
└── Train                 trénovací smyčky a vstupní body pro forward/backward
```

- `MDNN` je vrcholový objekt. Vlastní `NetworkContext`, `LayerManager` (přístupný jako `model.Layers`) a pomocníka `Train` (`model.Train`).
- `Layer` je abstraktní báze pro každý typ vrstvy. `Dense` a `RNN` jsou postavené z objektů `Neuron` (dědí z `LayerBasedOnNeurons`); `Conv` drží přímo kernely a biasy; `MaxPool` nemá žádné trénovatelné parametry a jen si pamatuje, které pozice byly vybrány.
- `Neuron` drží své váhy, bias, naakumulované gradienty a **vlastní** instanci optimizéru.
- `Tensor` je datový nosič předávaný mezi vrstvami (jednorozměrné pole `Data` plus `Shape`).
- `NetworkContext` (`model.Context`) drží stav per-model. Protože tento stav není globální, dva modely v jednom procesu jsou plně nezávislé.

### Odvození tvaru vstupu (lazy)

Vrstvy se vytvářejí, aniž by znaly velikost svého vstupu — zadáváte jen počet neuronů (nebo kernelů). Při prvním dopředném průchodu si model zaznamená tvar vstupu do `Context.InputShape` a zavolá `Layers.SetInputSizeForFirstLayer()`, které projde vrstvy a u každé zavolá `LayerAdjustment()`. Každá vrstva si odvodí skutečnou velikost vstupu z výstupní velikosti předchozí vrstvy a teprve pak alokuje své parametry (neurony, kernely). Proto váhy vrstvy neexistují, dokud model poprvé neuvidí data.

### Dopředný průchod

`model.GetResults(input)` protáhne `Tensor` vrstvami v pořadí a u každé zavolá `FeedForward()`. Vrstva `Dense` počítá pro každý neuron `w · x + b` následované aktivací; celovrstvové aktivace jako `Softmax` se aplikují na celou vrstvu najednou.

### Zpětný průchod

Zpětnou propagaci řídí `Gradient.GetGradients(target, model)`:

1. Chyba výstupní vrstvy se spočítá z derivace loss funkce. U prvkové výstupní aktivace se násobí derivací té aktivace; u fúzovaného případu softmax + cross-entropy (níže) se použije přímo.
2. Chyba se propaguje zpět: `CalculateLayerGradients()` každé vrstvy převede chybu následující vrstvy na svou vlastní chybu (řetízkové pravidlo) a cestou aplikuje derivaci aktivace vrstvy.
3. `BackPropagation()` každé vrstvy pak naakumuluje gradienty parametrů (u neuronových vrstev do `gradientsW` / `gradientsB` každého `Neuron`u).

Gradienty se **akumulují, neaplikují**. `UpdateParams()` vydělí naakumulované gradienty počtem viděných vzorků a předá je optimizéru. Proto je několik volání `Fit()` následovaných jedním `UpdateParams()` ekvivalentní jednomu minibatchi dané velikosti.

### Fúze softmax + cross-entropy

Když je loss funkcí `CrossEntropy` (která hlásí `RequiresSoftmax`) a výstupní vrstva používá `Softmax`, výstupní gradient se spočítá přímo jako `výstup − cíl`. To je fúzovaná, numericky stabilní forma: jacobián softmaxu se přeskočí a derivace aktivace se záměrně *nenásobí*, což by ji jinak aplikovalo dvakrát. Použití `CrossEntropy` bez softmax výstupní vrstvy vyhodí výjimku.

### Optimizéry

Každý `Neuron` (a každá vrstva `Conv`) vlastní svou instanci optimizéru, naklonovanou z toho v `NetworkContext`u modelu. Stav optimizéru je proto per-parametr a per-model — například `Adam` drží odhady prvního a druhého momentu s bias korekcí pro každou jednotlivou váhu. Metoda `Update(value, gradient, index)` optimizéru vrací novou hodnotu parametru.

### Rekurentní vrstvy (RTRL)

Vrstvy `RNN` se trénují pomocí **Real-Time Recurrent Learning**, ne backpropagation-through-time. Místo rozvinutí sekvence si vrstva nese citlivosti dopředu v čase (`∂h/∂váha` a `∂h/∂bias`), které se posouvají v každém časovém kroku uvnitř `FeedForward()`. Při zpětné propagaci se příchozí chyba násobí těmito uloženými citlivostmi. Na začátku každé sekvence zavolejte `model.ResetSequence()`, čímž se vynuluje skrytý stav i citlivosti.

### Inicializace vah

Neurony `Dense` a `RNN` se inicializují uniformním schématem ve stylu Xavier/He — `U(−1, 1) · sqrt(6 / n_vstupů)` — a biasy začínají na nule.

### Správnost: numerický gradient check

Celý zpětný průchod je ověřen testovací sadou pomocí numerického gradient checku (centrální diference): analytické gradienty každé vrstvy se porovnávají s odhady z konečných diferencí. Tyto kontroly fungují jako regresní pojistka — jakákoli změna, která rozbije matematiku, je automaticky odhalena.

## Testy

Repozitář obsahuje xUnit testovací projekt (`MDNN.Tests`) s numerickými gradient checky pro každou vrstvu, jednotkovými testy aktivačních, ztrátových a optimalizačních tříd a end-to-end trénovacími smoke testy. Spuštění:

```
dotnet test
```

Gradient checky slouží jako regresní pojistka: jakákoli změna, která rozbije matematiku, je automaticky odhalena.
