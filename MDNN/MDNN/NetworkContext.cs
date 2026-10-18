using My_DNN.Loss_functions;
using My_DNN.Optimizers;

namespace My_DNN
{
    // Instanční konfigurace modelu — postupně nahrazuje globální GeneralNeuralNetworkSettings (Fáze 3).
    // Každý model má svůj NetworkContext → dva modely v procesu si nešlapou po sobě.
    public class NetworkContext
    {
        public Loss Loss { get; set; } = new MSE();
        private Optimizer _optimizer = new SGD(0.0001);

        // Optimizer modelu. Každý neuron (a každá Conv vrstva) má vlastní klon — stav je
        // per-parametr. Změna po sestavení modelu se PROPÍŠE do všech vrstev: dostanou nový
        // optimizer s prázdným stavem. Dřív se nepropsala vůbec — vrstvy trénovaly dál
        // původním, ale do souboru se uložil nový, takže po načtení model trénoval jiným
        // optimizerem než před uložením. Přiřazení téže instance nic nedělá (stav zůstane).
        public Optimizer Optimizer
        {
            get => _optimizer;
            set
            {
                ArgumentNullException.ThrowIfNull(value);
                if (ReferenceEquals(value, _optimizer)) return;

                _optimizer = value;
                OptimizerChanged?.Invoke(value);
            }
        }

        // Kdo se má o změnu optimizeru dozvědět — LayerManager modelu. Jedno místo, ne
        // událost: při znovunačtení vrstev se přepíše, takže starý správce nezůstane viset.
        internal Action<Optimizer>? OptimizerChanged { get; set; }

        // Zdroj náhody pro CELÝ model: inicializace vah/kernelů, míchání datasetu i výběr
        // vzorků v minibatchi. Dřív to byly tři nezávislé zdroje
        // (GeneralNeuralNetworkSettings.rnd, Train._rnd, Random.Shared v ShuffleTensor)
        // a žádný z nich nešel nastavit → stejný experiment nešlo spustit dvakrát.
        //
        // Když se seed nezadá, spadne se zpátky na sdílený globální rnd — chování zůstává
        // přesně jako dřív (žádná změna pro existující kód).
        //
        // POZOR: `Random` není thread-safe. Dnes se z něj tahá jen ze sekvenčních míst
        // (výběr vzorku a míchání jsou mimo Task.Run), ale při data-paralelním minibatchi
        // z Fáze 5 to bude potřeba ošetřit — buď zámek, nebo stream na vlákno.
        private Random? _random;
        private int? _seed;
        private bool _randomAlreadyUsed;

        // Seed s možností přečíst zpět — AutoML si u kandidáta potřebuje poznamenat,
        // čím se běh dá zopakovat, a jinak by ho musel evidovat vedle modelu.
        // null = žádný seed, jede se na sdíleném globálním generátoru (jako dřív).
        public int? Seed
        {
            get => _seed;
            set
            {
                ThrowIfRandomAlreadyUsed(nameof(Seed));
                _seed = value;
                _random = value == null ? null : new Random(value.Value);
            }
        }

        public Random Random
        {
            get
            {
                // Od prvního čerpání je pozdě cokoli přenastavovat — váhy už jsou vylosované.
                _randomAlreadyUsed = true;
                return _random ??= GeneralNeuralNetworkSettings.rnd;
            }
            set
            {
                ThrowIfRandomAlreadyUsed(nameof(Random));
                _random = value;
                _seed = null;   // vlastní generátor → seed neznáme, ať ho Seed nehlásí falešně
            }
        }

        // Bez tohohle guardu byla situace tichá past: LayerManager.Add() losuje váhy výstupní
        // vrstvy HNED, takže seed nastavený až po Add() na ni nedosáhl. Model pak vyšel
        // ČÁSTEČNĚ reprodukovatelný (skryté vrstvy ano, výstupní ne) a nic to nehlásilo —
        // což je horší než zjevná chyba, protože to vypadá, že reprodukovatelnost funguje.
        private void ThrowIfRandomAlreadyUsed(string propertyName)
        {
            if (_randomAlreadyUsed)
            {
                throw new InvalidOperationException(
                    $"{propertyName} cannot be set after the generator has already been drawn from " +
                    "(weights or kernels are already initialized). The model would only be partly " +
                    "reproducible. Pass the seed to the constructor instead: " +
                    "new MDNN(layer, optimizer, loss, seed: 42), or MDNN.LoadModel(path, seed: 42).");
            }
        }
        public bool SequenceTrain { get; set; } = false;
        public bool CalculationViaGpu { get; set; } = false;
        public int[] InputShape { get; set; } = new int[] { 0 };

        // ---------------------------------------------------------------- regularizace

        private double _l2 = 0.0;

        /// <summary>
        /// Koeficient L2 regularizace (weight decay). 0 = vypnuto (výchozí).
        ///
        /// Trestá velké váhy, čímž brání přeučení: k účelové funkci se přičte
        /// <c>λ/2 · Σw²</c>, což v gradientu znamená <c>+ λ·w</c>.
        ///
        /// POZOR na dvě věci:
        /// 1. Aplikuje se JEN na váhy a kernely, NE na biasy. Bias jen posouvá práh,
        ///    jeho zmenšování model nezjednoduší, jen mu sebere schopnost se posunout.
        /// 2. Trénovací loss, kterou knihovna vypisuje, ten člen λ/2·Σw² NEobsahuje —
        ///    stejně jako `weight_decay` v PyTorchi. Vypsaná loss je tedy čistá chyba
        ///    predikce, ne celá minimalizovaná funkce.
        ///
        /// Pozn.: tohle je klasické L2 (přičte se ke gradientu, takže u Adamu projde
        /// i adaptivním škálováním). Oddělený weight decay à la AdamW, který se
        /// aplikuje mimo adaptivní člen, je jiná věc a knihovna ho zatím nemá.
        /// </summary>
        public double L2
        {
            get => _l2;
            set
            {
                if (double.IsNaN(value) || value < 0)
                    throw new ArgumentOutOfRangeException(nameof(L2), value,
                        "The L2 coefficient must be a non-negative number (0 = regularization off).");

                _l2 = value;
            }
        }

        private double _dropout = 0.0;

        /// <summary>
        /// Globální dropout pro celý model. 0 = vypnuto (výchozí).
        ///
        /// Platí pro každou vrstvu, která si NEurčí vlastní hodnotu — takže u pětivrstvé
        /// sítě stačí jeden řádek místo pěti. Konkrétní vrstva ho přebije tím, že si
        /// zadá vlastní (<c>new Dense(64, new ReLu(), dropout: 0.2)</c>), včetně nuly,
        /// kterou se dropout na té vrstvě vypne.
        ///
        /// NEAPLIKUJE SE NA VÝSTUPNÍ VRSTVU. Zahazovat výstupy sítě není regularizace,
        /// je to poškozování predikce — a je to chyba, kterou by globální přepínač jinak
        /// tiše zavedl. Kdo to opravdu chce, musí si to na výstupní vrstvě zadat explicitně.
        /// </summary>
        public double Dropout
        {
            get => _dropout;
            set
            {
                if (double.IsNaN(value) || value < 0 || value >= 1)
                    throw new ArgumentOutOfRangeException(nameof(Dropout), value,
                        "Dropout must be in the range [0, 1). 0 = off; 1 would drop " +
                        "everything and the scaling factor 1/(1-p) would divide by zero.");

                _dropout = value;
            }
        }

        /// <summary>
        /// Je model v trénovacím režimu? Rozhoduje o chování vrstev, které se při
        /// tréninku a při predikci chovají jinak — dnes dropout.
        ///
        /// Nastavuje se automaticky: <c>Fit</c> ho na dobu svého průchodu zapne,
        /// samostatné <c>GetResults</c> ho nechá vypnutý. Díky tomu dá predikce na stejný
        /// vstup vždy stejnou odpověď, i když je dropout zapnutý.
        /// </summary>
        public bool IsTraining { get; set; } = false;

        // Kontext repliky pro paralelní trénink. Stejné nastavení tréninku, ale vlastní
        // akumulátor loss a VLASTNÍ generátor náhody — `Random` není thread-safe a z dropoutu
        // se z něj tahá v každém vzorku. Seed dodá volající z generátoru originálu, takže
        // je to se seedem modelu pořád reprodukovatelné.
        internal NetworkContext CreateReplica(int randomSeed)
        {
            return new NetworkContext
            {
                Loss = Loss.CreateReplica(),
                Optimizer = Optimizer,   // replika parametry neaktualizuje, jen kvůli úplnosti
                SequenceTrain = SequenceTrain,
                InputShape = InputShape,
                L2 = L2,
                Dropout = Dropout,
                Random = new Random(randomSeed),
            };
        }
    }
}
