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
        public bool SequenceTrain { get; set; } = false;
        public bool CalculationViaGpu { get; set; } = false;
        public int[] InputShape { get; set; } = new int[] { 0 };
    }
}
