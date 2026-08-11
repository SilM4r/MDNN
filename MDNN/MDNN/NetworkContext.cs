using My_DNN.Loss_functions;
using My_DNN.Optimizers;

namespace My_DNN
{
    // Instanční konfigurace modelu — postupně nahrazuje globální GeneralNeuralNetworkSettings (Fáze 3).
    // Každý model má svůj NetworkContext → dva modely v procesu si nešlapou po sobě.
    public class NetworkContext
    {
        public Loss Loss { get; set; } = new MSE();
        public Optimizer Optimizer { get; set; } = new SGD(0.0001);

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
                    $"{propertyName} nelze nastavit potom, co se z generátoru už čerpalo " +
                    "(váhy nebo kernely jsou vylosované). Model by vyšel jen částečně " +
                    "reprodukovatelný. Předej seed rovnou konstruktoru: " +
                    "new MDNN(vrstva, optimizer, loss, seed: 42), případně MDNN.LoadModel(cesta, seed: 42).");
            }
        }
        public bool SequenceTrain { get; set; } = false;
        public bool CalculationViaGpu { get; set; } = false;
        public int[] InputShape { get; set; } = new int[] { 0 };
    }
}
