namespace My_DNN
{
    // Dvojice vstupy + očekávané výstupy. Existuje proto, aby JEDNO bez DRUHÉHO nešlo předat.
    //
    // Dřív se valid/test dodávaly čtyřmi nezávislými settery (`ValidDataInputs`,
    // `ValidDataCurrentOutput`, …). Nastavit vstupy a zapomenout na cíle bylo tiše přípustné:
    // dělení datasetu si s null poradilo (`originalValidOutput?.Slice(...)`), trénink
    // nastartoval a spadl až u prvního valid reportu — tedy daleko od místa, kde chyba vznikla.
    // Když jsou vstupy a cíle svázané v jednom objektu, ten stav nejde vyrobit.
    //
    // Pozn. k názvu: NE `Dataset`/`DataSet` — `Train.cs` má `using System.Data`, kde `DataSet`
    // je něco úplně jiného.
    public sealed class LabeledData
    {
        public Tensor Inputs { get; }
        public Tensor Targets { get; }

        // Kolik vzorků (u sekvenčního tréninku kolik sekvencí).
        public int Count => Inputs.Shape[0];

        public LabeledData(Tensor inputs, Tensor targets)
        {
            ArgumentNullException.ThrowIfNull(inputs);
            ArgumentNullException.ThrowIfNull(targets);

            if (inputs.Shape.Length == 0 || targets.Shape.Length == 0)
            {
                throw new ArgumentException("Vstupy i cíle musí mít aspoň jednu dimenzi.");
            }

            if (inputs.Shape[0] != targets.Shape[0])
            {
                throw new ArgumentException(
                    $"Počet vzorků se neshoduje: vstupy {inputs.Shape[0]}, cíle {targets.Shape[0]}.");
            }

            Inputs = inputs;
            Targets = targets;
        }

        // Pohodlná varianta pro double[][], double[,] a spol. — ať uživatel nemusí
        // převádět na Tensor ručně.
        public LabeledData(Array inputs, Array targets)
            : this(Tensor.ConvertArrayToTensor(inputs), Tensor.ConvertArrayToTensor(targets))
        {
        }
    }
}
