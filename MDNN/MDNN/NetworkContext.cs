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
        public Random Random
        {
            get => _random ??= GeneralNeuralNetworkSettings.rnd;
            set => _random = value;
        }
        public bool SequenceTrain { get; set; } = false;
        public bool CalculationViaGpu { get; set; } = false;
        public int[] InputShape { get; set; } = new int[] { 0 };
    }
}
