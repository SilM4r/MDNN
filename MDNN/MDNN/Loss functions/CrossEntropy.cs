namespace My_DNN.Loss_functions
{
    public class CrossEntropy: Loss
    {
        public override string Name => "Cross Entropy";
        // Kategorická cross-entropy: L = -Σ t_i · ln(s_i)
        //
        // Guard proti ln(0) je CLAMP, ne přičtení epsilonu. Dřív tu bylo `value + 1e-15`,
        // což hodnotu loss tiše ZASTROPOVALO na −ln(1e-15) ≈ 34,5: jakmile softmax vydal
        // pravděpodobnost menší než epsilon, epsilon ji přebil a loss dál nerostla.
        // U saturovaného softmaxu (s ≈ 1e-181, tedy „síť si je jistá a mýlí se") vycházelo
        // 34,5 místo 414,7 — a protože se podle valid loss vybírá nejlepší model a rozhoduje
        // early stopping, byla to chyba, kterou nebylo jak poznat. Gradient je jiná cesta
        // (fúze s − t) a zasažený nebyl, takže se model učil správně a jen ŠPATNĚ REPORTOVAL.
        //
        // Clamp na 1e-300 udrží skutečnou hodnotu až k hranici, kde double podteče na nulu,
        // a přitom pořád brání −Infinity. Odhaleno paritním testem proti PyTorchi
        // (fixture dense_softmax_ce_saturated).
        public override double LossFunction(double value, double targetValue)
        {
            return -targetValue * Math.Log(Math.Max(value, 1e-300));
        }

        // Fúzovaný gradient softmax + kategorická CE: dL/dz = s - t.
        // Platí jen pro softmax výstup; backprop díky FusedWithSoftmax bere
        // tuto hodnotu jako hotovou deltu a NEnásobí ji softmax derivací.
        public override double DerivativeOfLossFunction(double value, double targetValue)
        {
            return value - targetValue;
        }

        public override bool RequiresSoftmax => true;
    }
}
