using My_DNN.Activation_functions;
using My_DNN.Loss_functions;
using My_DNN.Optimizers;


namespace My_DNN
{
    public static class GeneralNeuralNetworkSettings
    {
        public static Activation_func output_activation_func = new Linear();

        public static Activation_func hidden_layers_activation_func = new ReLu();

        public static Loss loss_func = new MSE();

        public static Optimizer optimizer = new SGD(0.000000001);

        public static bool calculationViaGpu = false;

        public static bool AutoSaveInTrainLoop = true;

        public static string AutoSaveInTrainLoopFileName = "AutoSave";

        public static bool SequenceTrain = false;

        public static int[] modelInputSizeAndShape = new int[] { 0 };

        public static Random rnd = new Random();

    }
}
