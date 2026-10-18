# MDNN (My Deep Neural Network)

MDNN je knihovna pro návrh a trénování neuronových sítí v jazyce C#. Umožňuje snadnou tvorbu a konfiguraci modelů neuronových sítí, jejich trénování a následnou integraci do aplikací.

## Obsah

- [Klíčové vlastnosti](#klíčové-vlastnosti)
- [Instalace](#instalace)
- [Rychlý start](#rychlý-start)
- [Příklady](#příklady)
- [Konfigurace modelu](#konfigurace-modelu)
- [Přidávání vrstev](#přidávání-vrstev)
- [Regularizace](#regularizace)
- [Reprodukovatelnost](#reprodukovatelnost)
- [Trénování modelu](#trénování-modelu)
- [GPU akcelerace](#gpu-akcelerace)
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
- Regularizace: L2 weight decay a dropout
- Reprodukovatelné běhy z jednoho seedu
- Snadná integrace do projektů v C#
- Volitelný výpočet na GPU
- Ukládání a načítání modelů ve formátu JSON
- Předdefinované trénovací smyčky, volitelně paralelní přes minibatch

## Instalace

MDNN je distribuována jako dynamická knihovna `MDNN.dll`. Pro použití:

1. Přidejte do projektu referenci na projekt `MDNN`, nebo na `MDNN.dll`, který sestaví.
2. Zahrňte příslušné jmenné prostory ve zdrojovém kódu.

Grafy loss jsou v samostatné knihovně `MDNN.Plotting` (přináší ScottPlot, který jádro
knihovny nepotřebuje) — pokud je chcete, přidejte referenci i na ni, viz
[Tvorba grafů](#tvorba-grafů).

Obě jdou zabalit i jako NuGet balíčky, `MDNN` a `MDNN.Plotting`:

```
dotnet pack MDNN/MDNN.sln -c Release
```

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

Kompletní spustitelné příklady pro každý typ úlohy — klasifikace, regrese, sekvenční data (RNN) a obrazová data (Conv + MaxPool) — najdete v samostatném repozitáři:

[github.com/SilM4r/MDNN_examples](https://github.com/SilM4r/MDNN_examples)

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

Vlastní optimizér nebo ztrátovou funkci lze vytvořit zděděním odpovídající bázové třídy (například `Optimizer` nebo `Loss`) a implementací jejích abstraktních členů. `CreateFresh()` vlastního optimizéru musí vrátit novou instanci se stejnými hyperparametry a prázdným stavem — každý neuron dostane vlastní kopii. Má-li jeho stav přežít uložení a načtení, přepište i `ExportState()` a `ImportState()`; bez nich se načte s prázdným stavem. Načítání podle jména postaví jen vestavěné optimizéry a loss — pro vlastní předejte instanci: `MDNN.LoadModel(cesta, optimizer: new MujOptimizer(…), loss: new MojeLoss())`. Vlastní aktivaci zatím načíst nejde. V obou případech to skončí výjimkou `ArgumentException`, místo aby se tiše dosadila jiná funkce.

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

## Regularizace

Obě techniky jsou ve výchozím stavu vypnuté a nastavují se na `model.Context`.

### L2 weight decay

K účelové funkci přičte `λ/2 · Σw²`, což v gradientu znamená `+ λ·w` — velké váhy se
penalizují, takže model je tlačen k jednodušším řešením.

```csharp
model.Context.L2 = 0.001;   // 0 = vypnuto (výchozí)
```

Co je dobré vědět:

- Uplatňuje se na **váhy a konvoluční kernely, nikdy na biasy**. Bias netvaruje rozhodovací
  hranici, jen ji posouvá, takže jeho zmenšování model nezjednoduší.
- Přičítá se k **už zprůměrovanému** gradientu, takže síla regularizace se nemění se
  změnou velikosti minibatche.
- Loss vypisovaná během tréninku člen `λ/2 · Σw²` **neobsahuje** — stejná konvence, jakou
  PyTorch používá pro `weight_decay`. Vidíte jen chybu predikce.
- Jde o klasické L2 (přičtené ke gradientu, takže u `Adam` projde i adaptivním škálováním).
  Oddělený weight decay ve stylu AdamW je jiná věc a implementovaný není.

### Dropout

Během tréninku se náhodně vypne část neuronů, takže se síť nemůže spolehnout na žádný
jednotlivý a musí si k odpovědi vybudovat několik nezávislých cest.

```csharp
model.Context.Dropout = 0.5;                        // výchozí pro každou skrytou vrstvu
model.Layers.Add(new Dense(64, new ReLu()));        // zdědí 0.5
model.Layers.Add(new Dense(32, new ReLu(), 0.2));   // vlastní hodnota má přednost
model.Layers.Add(new Dense(16, new ReLu(), 0.0));   // výslovně vypnuto
```

Hodnota zadaná na vrstvě vždy přebije globální, včetně `0.0` pro vypnutí. Hodnota vrstvy je
k dispozici i později jako `DropoutRate` (`null` znamená „zdědit").

- **Globální hodnota se nikdy neuplatní na výstupní vrstvu.** Zahazovat vlastní výstupy sítě
  není regularizace, je to poškozování, takže ho jeden globální přepínač nesmí tiše zavést.
  Explicitně zadaná hodnota na výstupní vrstvě funguje, pokud o to opravdu stojíte.
- Dropout je aktivní **jen uvnitř `Fit()`** (a tedy uvnitř trénovacích smyček).
  `GetResults()` běží bez něj, takže predikce je deterministická — stejný vstup dá vždy
  stejnou odpověď.
- Přeživší neurony se během tréninku škálují `1/(1-p)` („inverted dropout"), což drží
  střední hodnotu výstupu beze změny, a inference proto nepotřebuje žádnou korekci.
- Vypnuté neurony se **přeskakují, nenásobí nulou**. Jejich skalární součiny se vůbec
  nepočítají, ve forwardu ani v backwardu. Na síti 512-512 trvá trénink s `dropout: 0.5`
  asi 82 % času tréninku bez dropoutu — dropout tu trénink *zrychluje*, což knihovny
  postavené na GPU nedokážou.
- Zatím je podporovaný jen na vrstvách `Dense`.

Obě nastavení se ukládají s modelem, takže model, který se načte a trénuje dál, si zachová
regularizaci, se kterou byl trénovaný.

## Reprodukovatelnost

Předejte konstruktoru seed a dva běhy téhož programu dají bit po bitu stejný výsledek —
inicializace vah, pořadí vzorků, míchání datasetu i masky dropoutu čerpají z jednoho generátoru.

```csharp
MDNN model = new MDNN(new Dense(3, new Softmax()), new Adam(0.01), new CrossEntropy(), seed: 42);

int? used = model.Context.Seed;   // jde přečíst zpět a ukládá se s modelem
```

Seed nastavte **v konstruktoru**. Přiřazení `Context.Seed` nebo `Context.Random` poté, co se
z generátoru už čerpalo, vyhodí `InvalidOperationException`: váhy výstupní vrstvy se losují
už při skládání modelu, takže později nastavený seed by dal model reprodukovatelný jen zčásti
— horší než zjevná chyba, protože to vypadá, že to fungovalo.

Bez seedu knihovna použije sdílený globální generátor, přesně jako dřív.

## Trénování modelu

Trénování řídí třída `Train`. K dispozici jsou čtyři úrovně kontroly, od plně automatické po plně ruční.

### `TrainLoop()`

Nejúplnější trénovací procedura. Poskytuje:

- Automatické ukládání modelu s nejlepší validační chybou (checkpoint pro early stopping)
- Automatické zamíchání a rozdělení datasetu na trénovací, validační a testovací sadu
- Výpis průběhu do konzole
- Detekci hodnot `NaN`
- Graf loss v průběhu epoch (s `MDNN.Plotting`, viz [Tvorba grafů](#tvorba-grafů))

**Jedna epocha = jeden úplný průchod trénovací sadou.** Každou epochu se pořadí vzorků znovu
zamíchá a data se zpracují po minibatchích, s jedním krokem optimizéru na dávku. Parametr
`number_of_epoch` tedy počítá průchody daty, ne kroky optimizéru — ty najdete
v `Train.OptimizerSteps`.

Parametry:

- `Array inputs_values` (povinný) – vstupní dataset. Každý řádek je jeden trénovací vzorek.
  Vzorek má nejvýš tři rozměry, takže dataset je nejvýš 4D — pro obrázky
  `[vzorek, výška, šířka, kanály]` — a pro sekvence nejvýš 5D
  `[sekvence, krok, výška, šířka, kanály]`.
- `Array current_output_values` (povinný) – odpovídající cíle.
- `uint number_of_epoch` (povinný) – počet trénovacích epoch.
- `uint size_of_mini_batch` (volitelný, výchozí `1`) – velikost minibatche.
- `bool isSequence` (volitelný, výchozí `false`) – nastavte na `true` pro sekvenční vstupní data (například časové řady).

Tohle přetížení je zkratka: zadaná data rozdělí podle poměrů a trénuje. Pokud už máte vlastní
rozdělení, předejte sady explicitně a zavolejte přetížení bez dat:

```csharp
model.Train.SetDatasets(
    train: new LabeledData(trainX, trainY),
    valid: new LabeledData(validX, validY),
    test:  new LabeledData(testX, testY));   // test je volitelný

model.Train.TrainLoop(numberOfEpoch: 50, sizeOfMiniBatch: 32);
```

Co vynecháte, to se ukrojí za vás:

| Volání | Chování |
|---|---|
| `SetDatasets(train)` | valid a test se ukrojí z train (0.7 / 0.15 / 0.15) |
| `SetDatasets(train, valid)` | test se ukrojí z valid; train zůstane celý |
| `SetDatasets(train, null, test)` | valid se ukrojí z train; testovací sady se nikdo nedotkne |
| `SetDatasets(train, valid, test)` | nic se nekrojí |

Z testovací sady se nikdy nekrojí — je to konečný nestranný odhad. Nastavení
`TestNeuralNetworkAfterTraining = false` navíc zabrání tomu, aby se testovací sada ukrojila
z vaší validační, takže validační sada zůstane celá.

### `SimpleTrainLoop()`

Zjednodušená trénovací smyčka s checkpointingem, výpisem do konzole a detekcí `NaN`, bez rozdělení datasetu a vykreslování grafu z `TrainLoop()`.

Parametry:

- `double[][] inputs_values` (povinný)
- `double[][] current_output_values` (povinný)
- `uint number_of_epoch` (povinný)
- `uint size_of_mini_batch` (volitelný, výchozí `1`)

### Paralelní trénink (`Workers`)

`TrainLoop()` a `SimpleTrainLoop()` umí rozdělit každou minibatch mezi několik vláken:

```csharp
model.Train.Workers = 4;                      // výchozí 1 = sekvenčně
model.Train.TrainLoop(inputs, outputs, 20, 64);
```

Každé vlákno dostane vlastní repliku modelu, spočítá svůj díl dávky a gradienty se před krokem
optimizéru sečtou do modelu — stejná matematika jako sekvenčně, liší se jen pořadí sčítání
desetinných čísel. Repliky sdílejí váhy s modelem (během dávky je jen čtou), takže se mezi
dávkami kromě biasů nic nekopíruje.

Co je dobré vědět:

- Dělí se jen práce **uvnitř** dávky. Při výchozí velikosti dávky `1` se nic nezrychlí
  a víc vláken, než je vzorků v dávce, nic nepřidá. Změřeno na 10jádrovém Apple Silicon
  (4 výkonná jádra), jedna epocha: MLP 784-128-10 se se 4 vlákny učí 1,9× rychleji při dávce
  32 a 2,6× při dávce 128; malá CNN 3,1× při dávce 32. Větší dávky škálují lépe — gradienty
  replik se sčítají jednou za dávku a u malých dávek je ten součet znatelný díl práce.
- **Reprodukovatelnost:** stejný seed a stejné `Workers` dají bit po bitu stejný výsledek na
  jakémkoli stroji. Proti `Workers = 1` se výsledek liší o zaokrouhlení, s dropoutem i
  vylosovanými maskami — každá replika má vlastní generátor odvozený ze seedu modelu.
- Vlastní vrstvu zatím paralelně trénovat nejde, `RNN` jen v sekvenčním režimu
  (`isSequence: true`; mimo něj teče skrytý stav z jednoho vzorku do dalšího) a paralelní
  trénink nejde kombinovat s GPU. Všechny tři případy vyhodí `NotSupportedException`, místo
  aby se tiše trénovalo jinak.
- **Paralelně běží i validace a test** — validace mezi epochami, `TestNeuralNetwork()`
  i závěrečný test v `TrainLoop()`. Jde jen o dopředné průchody: výstupy se spočítají
  paralelně a loss a skóre se z nich pak sečtou v pořadí vzorků, takže vyjdou **bit po bitu
  stejně** jako při `Workers = 1` — a s nimi i výběr nejlepšího modelu a early stopping.
  Změřeno: zhruba 3× se 4 vlákny, až 4–4,5× s 10.
- Vlastní smyčka s `Fit()` a `UpdateParams()` běží vždy sekvenčně.

### `Fit()` a `UpdateParams()`

Prostřední cesta, která umožňuje napsat si vlastní trénovací smyčku. `Fit()` provede dopředný průchod a zpětnou propagaci, ale gradienty místo aplikace akumuluje; `UpdateParams()` naakumulované gradienty aplikuje. Volání jedno po druhém odpovídá tréninku po jednom vzorku; několik volání `Fit()` před jedním `UpdateParams()` odpovídá tréninku po minibatchích.

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
    // jednou za epochu zamíchat a projít celou sadu — totéž, co dělá TrainLoop
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

        model.Train.UpdateParams();   // jeden krok optimizéru na dávku
    }
}
```

Vlastní smyčka nemá žádné epochy: `Train.CurrentEpoch` zůstává `0` a posouvá se jen
`Train.OptimizerSteps`.

### `FeedForward()` a `BackPropagation()`

Nejjemnější úroveň, která rozdělí `Fit()` na samostatný dopředný průchod (`FeedForward()`)
a zpětnou propagaci (`BackPropagation(target)`). Dává plnou kontrolu nad jednotlivými kroky
tréninku, což se hodí pro výzkum nebo pokročilou optimalizaci. Po zpětné propagaci zavolejte
`UpdateParams()`, aby se změny aplikovaly.

```csharp
model.Train.FeedForward(new Tensor(sample));
model.Train.BackPropagation(new Tensor(target));   // akumuluje, neaplikuje
model.Train.UpdateParams();                        // jeden krok optimizéru
```

Tyhle dvě metody jsou přesně dvě poloviny `Fit()`, a je to záruka, ne náhoda:
`Train.FeedForward()` přepne model na dobu průchodu do trénovacího režimu, takže dropout se
chová stejně jako uvnitř `Fit()`. (Pro predikci použijte `model.GetResults()` — ta dropout
záměrně nechává vypnutý, takže stejný vstup dá vždy stejnou odpověď.)

**Odstraněno:** `BackPropagation()` mívala druhé přetížení, které bralo předpočítané gradienty
jednotlivých vrstev. Je pryč a nikdy nefungovalo tak, jak signatura slibovala — `Dense`
argument respektovala, zatímco `Conv` a `RNN` ho ignorovaly a četly vnitřní stav. Každá vrstva
má teď jediný vstupní bod do backwardu (viz [Zpětný průchod](#zpětný-průchod)), takže zvenku
už není co předávat.

## GPU akcelerace

### GPU

Výpočty sítě mohou volitelně běžet na NVIDIA GPU přes doprovodnou knihovnu `gpu.dll` (napsanou v C++ / CUDA). Vyžaduje to CUDA Toolkit a `gpu.dll`. GPU je nastavení per-model:

```csharp
model.Context.CalculationViaGpu = true;
```

Podpora aktuálně cílí jen na NVIDIA GPU a `gpu.dll` se načítá bez náhradní cesty — tahle
cesta je tedy jen pro Windows s CUDA.

**GPU cesta počítá ve `float`, CPU cesta v `double`.** Hodnoty se na vstupu převádějí a na
výstupu zase zpět, takže tentýž model nedá na obou cestách stejná čísla — počítejte se shodou
zhruba na 7 platných cifer místo 16. U GPU kernelů je to běžné, ale je dobré to vědět, než
začnete výsledky obou cest porovnávat. GPU cestu používají jen vrstvy `Dense`; všechno ostatní
běží na CPU bez ohledu na tenhle přepínač.

### Odstraněné `...Async` API

Dřívější verze měly asynchronní protějšky (`TrainLoopAsync()`, `FitAsync()`,
`GetResultsAsync()`). **Byly odstraněny** a nic se tím neztratilo — byly `async` bez jediného
`await`, takže vlákno nikdy neuvolnily. Paralelismus, o který se pokoušely, byl `Task.Run`
po neuronech, a to je *pomalejší* než obyčejná smyčka: režie naplánování na neuron je vyšší
než samotný výpočet.

Horší bylo, že šlo o druhou cestu k témuž výpočtu, která se od první potichu rozešla. Dropout
přibyl jen do synchronní cesty, takže asynchronní tiše počítala špatné gradienty, dokud to
neodhalilo právě odstranění.

Paralelismus se vrátil, ale přes **dávku**, ne přes neurony — to je osa, na které je opravdu
co dělit. Viz [Paralelní trénink](#paralelní-trénink-workers).

## Ukládání a načítání modelů

Po tréninku lze model uložit do JSON a později načíst k inferenci. K použití uloženého modelu není potřeba nové trénování.

```csharp
model.SaveAsJson("save"); // zapíše save.json

MDNN loaded = MDNN.LoadModel("save.json");   // funguje i "save" — přípona je volitelná
Tensor result = loaded.GetResults(new Tensor(inputsDataset[0]));   // jeden vzorek
```

Soubor nese `FormatVersion`, seed, se kterým model vznikl, čas uložení a kontrolní součet
SHA-256 dat modelu:

```json
{
  "FormatVersion": 1,
  "Checksum": "sha256:7a54f121...",
  "Model": { "Seed": 42, "SavedAtUtc": "2026-08-11T11:08:52Z", ... }
}
```

Nesouhlasící kontrolní součet vyhodí `ModelFileCorruptedException`. Pozor, co to dokazuje
a co ne: odhalí **poškozený nebo omylem upravený** soubor, ne záměrně změněný — kdo soubor
upraví, může hash přepočítat. Soubory zapsané staršími verzemi (bez `FormatVersion`) se
načtou dál, jen bez ověření.

`LoadModel(path, seed)` umožňuje nastavit seed pro pokračování v tréninku; bez něj se obnoví
seed uložený v souboru.

**Ukládá se i stav optimizéru**, takže trénink jde přerušit a plynule navázat: momenty a počet
kroků `Adam` i rychlost `Momentum` se ukládají u každého neuronu a každé vrstvy `Conv`.
N kroků, uložení, načtení a dalších M kroků dá bit po bitu stejný výsledek jako N + M kroků bez
přerušení. (Bez stavu by `Adam` začínal znovu s vynulovanými momenty a jeho první kroky po
načtení by byly velké jako na úplném začátku tréninku.) Stav soubor zhruba ztrojnásobí; pro
model jen na predikci:

```csharp
model.SaveAsJson("model", includeOptimizerState: false);
```

Soubory bez stavu (starší, nebo uložené takhle) se načtou s prázdným stavem optimizéru a soubory
se stavem přečtou i starší verze knihovny, které ho ignorují.

Pro pokračování s jinými hyperparametry — třeba s nižším learning rate — předejte optimizér do
`LoadModel`; musí být stejného druhu jako v souboru a uložený stav se načte do něj:

```csharp
MDNN model = MDNN.LoadModel("model", optimizer: new Adam(0.0001));
```

Nejlepší model, který `TrainLoop()` na konci obnoví, si nese stav optimizéru z epochy, ze které
pochází, takže se z něj dá rovnou trénovat dál.

## Podpůrné nástroje

### Tensor

`Tensor` je univerzální datový typ pro vícerozměrná pole. Uchovává:

- původní vícerozměrné pole (`OriginalInput`),
- ekvivalentní ploché pole (`Data`) pro rychlejší výpočty,
- a tvar jako seznam rozměrů (`Shape`).

`new Tensor(pole)` přijme jakékoli pravidelné pole typu `double`, `int` nebo `float` — jednorozměrné, vícerozměrné (`double[,,]`) i jagged (`double[][][]`); `int`/`float` převede na `double` a nepravidelné pole odmítne. Podporuje přetváření přes `Reshape(int[] newShape)` a pohodlný přístup k prvkům i převody mezi jagged a vícerozměrnými poli.

### Konzolové výstupy

Statická třída `ConsoleControler` obstarává výpisy do konzole:

- `ShowModelInfo()` – vypíše podrobné informace o modelu
- `ShowEpochInfo()` – vypíše informace o aktuální epoše během tréninku
- `ShowScoreOfmodel()` – vypíše přesnost modelu
- `ErrorHandler()` – vypisuje chybové hlášky

### NetworkContext

Každý model vlastní `NetworkContext` (`model.Context`), který drží jeho běhovou konfiguraci. Protože je tento stav per-model, dva modely v jednom procesu si navzájem nepřekážejí.

| Člen | Význam |
|---|---|
| `Loss`, `Optimizer` | ztrátová funkce a optimizér tohoto modelu. Pozdější přiřazení jiného optimizéru se propíše do všech vrstev — každá dostane čerstvou kopii s prázdným stavem |
| `Seed`, `Random` | vlastní generátor náhody modelu — viz [Reprodukovatelnost](#reprodukovatelnost) |
| `L2` | koeficient L2 weight decay, `0` = vypnuto |
| `Dropout` | výchozí dropout pro skryté vrstvy, `0` = vypnuto |
| `IsTraining` | běží trénovací průchod? Zapíná ho `Fit()` na dobu svého dopředného průchodu; právě tím se zapíná a vypíná dropout |
| `InputShape` | tvar vstupu modelu, odvozený při prvním průchodu |
| `SequenceTrain` | režim sekvenčního tréninku (časové řady) |
| `CalculationViaGpu` | přepínač GPU |

`GeneralNeuralNetworkSettings` drží už jen výchozí hodnoty platné pro celý proces (výchozí aktivační funkce a sdílený generátor náhody).

### Tvorba grafů

Grafy trénovací a validační loss v průběhu epoch jsou v samostatné knihovně `MDNN.Plotting`,
takže jádro knihovny nezávisí na ScottPlotu. Zapojí se jednou na model:

```csharp
using My_DNN;   // GraphPlotter je v MDNN.Plotting, stejný jmenný prostor

model.Train.UseLossChart();   // TrainLoop pak na konci uloží loss.png
```

Graf se uloží jako `loss.png` do pracovního adresáře, na konci `TrainLoop()` a při divergenci
tréninku. Snadno se tak odhalí přeučení nebo nedotrénování. `ShowLossChartInTrainLoop = false`
ho zase vypne. Bez `UseLossChart()` se žádný graf nekreslí — jádro knihovny nabízí jen háček
`Train.LossChart(epochs, trainLoss, validLoss)`, na který jde napojit i vlastní kód pro grafy.

## Jak to funguje uvnitř

Tahle sekce popisuje, co se děje pod kapotou. K používání knihovny ji číst není nutné, ale vysvětluje návrh a hodí se při jejím rozšiřování.

### Objektový model

```
MDNN (model)
├── NetworkContext        běhová konfigurace modelu (loss, optimizér, tvar vstupu, přepínače)
├── LayerManager (Layers) seřazený seznam vrstev + líné odvození tvaru
│   └── Layer             Dense · Conv · MaxPool · RNN
│       └── Neuron        váhy + bias + vlastní optimizér   (Dense, RNN)
└── Train                 trénovací smyčky a vstupní body forward/backward
```

- `MDNN` je objekt nejvyšší úrovně. Vlastní `NetworkContext`, `LayerManager` (dostupný jako `model.Layers`) a pomocníka `Train` (`model.Train`).
- `Layer` je abstraktní předek všech typů vrstev. `Dense` a `RNN` jsou složené z objektů `Neuron` (dědí z `LayerBasedOnNeurons`); `Conv` drží kernely a biasy přímo; `MaxPool` nemá trénovatelné parametry a jen si pamatuje, které pozice vyhrály.
- `Neuron` drží své váhy, bias, naakumulované gradienty a **vlastní** instanci optimizéru.
- `Tensor` je nosič dat předávaný mezi vrstvami (ploché pole `Data` plus `Shape`).
- `NetworkContext` (`model.Context`) drží stav modelu. Protože tento stav není globální, dva modely v jednom procesu jsou plně nezávislé.

### Odvození tvaru vstupu (lazy)

Vrstvy se vytvářejí bez znalosti velikosti vstupu — zadává se jen počet neuronů (nebo kernelů). Při prvním dopředném průchodu model zaznamená tvar vstupu do `Context.InputShape` a zavolá `Layers.SetInputSizeForFirstLayer()`, která projde vrstvy a každé zavolá `LayerAdjustment()`. Každá vrstva si odvodí skutečnou velikost vstupu z výstupu předchozí vrstvy a teprve pak alokuje parametry (neurony, kernely). Proto váhy vrstvy neexistují, dokud model poprvé neuvidí data.

### Dopředný průchod

`model.GetResults(input)` provleče `Tensor` vrstvami v pořadí a každé zavolá `FeedForward()`. Vrstva `Dense` spočítá pro každý neuron `w · x + b` a pak aktivaci; aktivace působící na celou vrstvu, jako `Softmax`, se aplikují na celou vrstvu naráz.

### Zpětný průchod

Zpětnou propagaci řídí `Gradient.RunBackward(target, model)` a každá vrstva nabízí jediný
vstupní bod:

```csharp
Tensor Backward(Tensor gradOutput, bool preActivation = false, bool computeInputGradient = true)
```

Vezme `∂L/∂output` dané vrstvy, naakumuluje gradienty **vlastních** parametrů a vrátí
`∂L/∂input` — tedy `∂L/∂output` vrstvy pod sebou. Jeden průchod zásobníkem vrstev, jedna metoda
na vrstvu.

Dva důsledky stojí za to rozepsat, protože právě je tenhle tvar přináší:

- **Vrstva nikdy nesahá do souseda.** Každá propaguje gradient přes vlastní váhy, takže `Conv`
  a `MaxPool` už nepotřebují zvláštní větev „další vrstva je `Dense`" (dřív četly matici vah
  té vrstvy a ručně přepočítávaly ploché indexy).
- **Žádný skrytý stav mezi voláními.** Neexistuje zvláštní háček „naseeduj výstupní vrstvu"
  ani pole, které by neslo hodnotu z jedné metody backwardu do další.

`preActivation` říká, že `gradOutput` už je `∂L/∂z`, takže se derivace aktivace přeskočí.
Jediný takový případ je fúze softmax + cross-entropy níž, kde softmax patří k loss, ne k vrstvě
— přesně jako `CrossEntropyLoss` v PyTorchi obsahuje log-softmax.

`computeInputGradient: false` znamená, že `∂L/∂input` nikdo nepotřebuje, takže se nepočítá.
Používá se pro první vrstvu modelu — pod ní už nic není — a u `Conv` tím odpadne nejdražší
operace celého backwardu.

Každá vrstva aplikuje derivaci aktivace podle své konvence: `Conv` ano, `RNN` ne (její RTRL
citlivosti ji už obsahují).

Gradienty se **akumulují, ne aplikují**. `UpdateParams()` vydělí naakumulované gradienty počtem viděných vzorků a předá je optimizéru. Proto je několik volání `Fit()` následovaných jedním `UpdateParams()` totéž co jedna minibatch dané velikosti.

### Fúze softmax + cross-entropy

Když je loss `CrossEntropy` (hlásí `RequiresSoftmax`) a výstupní vrstva používá `Softmax`, výstupní gradient se spočítá rovnou jako `výstup − cíl`. Je to fúzovaná, numericky stabilní forma: Jacobián softmaxu se přeskočí a derivace aktivace se záměrně *nenásobí*, jinak by se uplatnila dvakrát. Použití `CrossEntropy` bez softmax výstupní vrstvy vyhodí výjimku.

### Optimizéry

Každý `Neuron` (a každá vrstva `Conv`) vlastní svou instanci optimizéru, naklonovanou z toho v `NetworkContext`u modelu. Stav optimizéru je proto per-parametr a per-model — například `Adam` drží odhady prvního a druhého momentu s bias korekcí pro každou jednotlivou váhu. Metoda `Update(value, gradient, index)` optimizéru vrací novou hodnotu parametru. Před každou dávkou aktualizací vrstva zavolá `EnsureCapacity(parameterCount)`, takže optimizér se stavem si ho alokuje jednou a na přesnou velikost, místo aby rostl po jedné váze; výchozí implementace nedělá nic.

### Rekurentní vrstvy (RTRL)

**Rekurence je diagonální.** Každý neuron dostává jako vstup navíc jen *svůj vlastní*
předchozí výstup — neexistuje matice vah skrytý→skrytý, takže se neurony napříč časovými kroky
nemíchají. Jde o strukturu známou jako *independently recurrent* síť (IndRNN), ne o učebnicové
`h(t) = f(W·x(t) + U·h(t−1))`. Pokud čekáte plně propojenou rekurentní vrstvu, tohle je rozdíl,
o kterém je dobré vědět.

Právě tahle struktura dělá metodu tréninku praktickou. Vrstvy `RNN` se trénují pomocí
**Real-Time Recurrent Learning**, ne backpropagation-through-time. Místo rozvinutí sekvence si
vrstva nese citlivosti dopředu v čase (`∂h/∂weight` a `∂h/∂bias`), které se v každém časovém
kroku posunou uvnitř `FeedForward()`. Při zpětné propagaci se přicházející chyba vynásobí
uloženými citlivostmi. Na začátku každé sekvence zavolejte `model.ResetSequence()`, který
vynuluje skrytý stav i citlivosti.

Pro diagonální rekurenci je RTRL **přesné** a jeho cena je lineární v počtu vah. S plnou maticí
skrytý→skrytý by potřebovalo `∂h_i/∂w_jk` pro každou dvojici, což roste jako `n³` a nad několik
desítek neuronů přestává být praktické — tam se správným nástrojem stává backpropagation-through-time.

Dva důsledky, které je dobré znát:

- Gradienty tekoucí do vrstev umístěných *před* rekurentní vrstvou zůstávají přibližné:
  příspěvek, který putuje zpět v čase, se nezapočítává.
- Nejsou k dispozici hradlované jednotky (LSTM/GRU); ty potřebují backpropagation-through-time.

### Inicializace vah

Neurony vrstev Dense a RNN se inicializují rovnoměrným rozdělením ve stylu Xavier/He — `U(−1, 1) · sqrt(6 / n_inputs)` — a biasy začínají na nule.

### Správnost: numerický gradient check a parita s PyTorchem

Správnost se ověřuje ze dvou nezávislých stran, protože každá chytí, co ta druhá nedokáže.

**Numerický gradient check** ověřuje celý zpětný průchod: analytické gradienty každé vrstvy se
porovnávají s odhady z centrálních diferencí. Zachytí jakoukoli změnu, která rozbije matematiku,
a slouží jako regresní hlídač.

Má ale jedno slepé místo. Gradient check porovnává zpětný průchod s numerickou derivací
*vlastního* dopředného průchodu — takže pokud se dopředný průchod řídí jinou konvencí, než bylo
zamýšleno (jiné pravidlo paddingu, otočený kernel, jiné škálování loss), kontrola projde
a nesoulad zůstane neviditelný.

**Paritní testy proti PyTorchi** tu mezeru zavírají tím, že dodají vnější referenci. Stejný
výpočet se sestaví v PyTorchi s bit po bitu stejnými vahami a vstupy a každé číslo se porovná
prvek po prvku: výstupy vrstev, hodnoty loss, gradienty vah a biasů, gradient vůči vstupu
a kroky optimizéru. Shoda je řádově `1e-15`, tedy na hranici přesnosti `double`.

Referenční hodnoty generuje `tools/torch_fixtures/generate_fixtures.py` a jsou commitnuté jako
JSON, takže spuštění testů Python nepotřebuje.

## Testy

Repozitář obsahuje testovací projekt v xUnit (`MDNN.Tests`) — přes 300 testů, které CI pouští
na Linuxu i Windows. Spustíte je takto:

```
dotnet test
```

Co pokrývají:

- **Numerické gradient checky** pro každou vrstvu, včetně cest regularizace
- **Paritní testy proti PyTorchi** (viz [výše](#správnost-numerický-gradient-check-a-parita-s-pytorchem)) — výstupy vrstev, gradienty a kroky optimizéru porovnané s vnější referencí
- Unit testy aktivačních funkcí, loss funkcí a optimizérů
- Uložení a načtení tam a zpět, včetně toho, co uložený model musí zachovat
- Paralelní trénink proti sekvenčnímu, pro každý typ vrstvy
- Smoke testy tréninku od začátku do konce

Gradient checky slouží jako regresní hlídač: každá změna, která rozbije matematiku pod tím,
se automaticky zachytí.
