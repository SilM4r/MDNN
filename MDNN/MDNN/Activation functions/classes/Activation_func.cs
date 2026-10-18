using My_DNN.Activation_functions;

namespace My_DNN
{
    public abstract class Activation_func
    {
        public abstract string Name { get; }
        public abstract double Apply(double value);
        public abstract double Derivative(double value);
        public abstract bool Apply_to_layer { get; }

        // Aktivace podle jména uloženého v souboru s modelem.
        //
        // Dřív všechno neznámé tiše propadlo do `default: return new Linear()`. Vlastní
        // aktivace se tak po načtení stala lineární, a stejně dopadly staré soubory s Conv
        // vrstvou: ta jméno ukládala přes ToString(), tedy celé jméno typu
        // „My_DNN.Activation_functions.ReLu" (opraveno v 2c-9). Model se načetl bez chyby
        // a predikoval jinou funkcí, než se trénoval.
        //
        // Teď: celé jméno typu se zkrátí na samotné jméno třídy (staré Conv soubory se tím
        // načtou SPRÁVNĚ, ne lineárně) a neznámé jméno je chyba, ne Linear.
        public static Activation_func inicialization_activation_func(string? name_of_activation_function)
        {
            if (string.IsNullOrEmpty(name_of_activation_function))
            {
                throw new ArgumentException("The model file is missing the name of a layer's activation function.");
            }

            string name = name_of_activation_function;
            int lastDot = name.LastIndexOf('.');
            if (lastDot >= 0)
            {
                name = name.Substring(lastDot + 1);
            }

            switch (name.ToLower())
            {
                case "linear":
                    return new Linear();
                case "sigmoid":
                    return new Sigmoid();
                case "tanh":
                    return new Tanh();
                case "relu":
                    return new ReLu();
                case "leak_relu":
                    return new Leak_ReLu();
                case "softmax":
                    return new Softmax();
                default:
                    throw new ArgumentException(
                        $"Activation function '{name_of_activation_function}' is not built in. A model " +
                        "with a custom activation cannot be loaded from a file yet — loading looks it up " +
                        "by name and only knows Linear, Sigmoid, Tanh, ReLu, Leak_ReLu and Softmax.");
            }
        }

    }
}
