namespace My_DNN.Layers.classes
{
    public abstract class Layer
    {
        // Reference na kontext modelu, do kterého vrstva patří (nastaví LayerManager při připojení).
        public My_DNN.NetworkContext? Context { get; set; }

        abstract public string Name { get; }

        abstract public int[] Input_size_and_shape { get; }
        abstract public int[] Output_size_and_shape { get; }
        abstract public Tensor? Layer_output { get; }
        abstract public Tensor? Layer_raw_output { get; }
        abstract public Activation_func Activation_Func { get; set; }
        abstract public Tensor FeedForward(Tensor input_values);
        abstract public Tensor CalculateLayerGradients(Tensor next_layer_e, Layer next_layer);
        abstract public void BackPropagation(Tensor de);
        abstract public void UpdateParams();
        abstract public void LayerAdjustment(int? number_of_elements = null, int[]? number_of_input = null);

        // Gradient shora pro VÝSTUPNÍ vrstvu. Prostřední vrstvy si vnitřní stav pro backward
        // (Conv.dOutput, RNN.gImm) naplní v CalculateLayerGradients — jenže tu poslední vrstva
        // nikdy nedostane (Gradient jede jen count-2..0), takže bez tohohle hooku zůstal její
        // stav nulový a vrstva se TIŠE neučila.
        //
        // `dLossDOutput[i]` = ∂L/∂output_i, tedy BEZ derivace aktivace — každá vrstva si ji
        // dodá podle své konvence (Conv ano, RNN ne, protože RTRL citlivosti ji už obsahují).
        // Default je no-op: Dense i MaxPool tenhle stav nemají, Dense.BackPropagation pracuje
        // rovnou se svým argumentem.
        virtual public void SeedOutputGradient(double[] dLossDOutput) { }

        virtual public async Task<Tensor> FeedForwardAsync(Tensor input_values)
        {
            return FeedForward(input_values);
        }
        virtual public async Task<Tensor> CalculateLayerGradientsAsync(Tensor next_layer_e, Layer next_layer)
        {
            return CalculateLayerGradients(next_layer_e,next_layer);
        }
        virtual public async Task BackPropagationAsync(Tensor de)
        {
            BackPropagation(de);
        }
        virtual public async Task UpdateParamsAsync()
        {
            UpdateParams();
        }
    }
}
