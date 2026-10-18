using My_DNN.Save_neural_network;


namespace My_DNN.Optimizers
{
    public abstract class Optimizer
    {
        public abstract string Name { get; }
        public abstract double[] Hyperparameters { get; }
        public abstract double Update(double w, double gradient,  int paramIndex);

        // Připraví stav pro parametry 0..parameterCount-1 dopředu. Vrstva ho volá před
        // dávkou Update, protože jen ona ví, kolik parametrů má — bez toho by stav rostl
        // po jedné váze (přealokování na každou váhu = kvadraticky při první dávce).
        // Už nasbíraný stav zůstává. Výchozí implementace nic nedělá — SGD stav nemá.
        public virtual void EnsureCapacity(int parameterCount) { }

        // Nová instance téhož optimizeru: stejné hyperparametry, PRÁZDNÝ stav. Každý neuron
        // (a každá Conv vrstva) potřebuje vlastní, protože stav je per-parametr.
        //
        // Abstraktní schválně. Výchozí implementace přes MemberwiseClone by byla mělká kopie,
        // takže by klony sdílely pole se stavem (momenty Adamu) — tichá chyba, kterou hlídá
        // Two_models_with_adam_do_not_share_momentum. Optimizer sám ví, jak se postavit.
        public abstract Optimizer CreateFresh();

        protected static double Rms(double val)
        {
            double sqrt = Math.Sqrt(val);
            if (sqrt == 0)
            {
                sqrt += double.Epsilon;
            }
            return sqrt;
        }

        public static Optimizer Refactor_optimizer(ExportOptimizer Optimizer)
        {
            return GetOptimizer(Optimizer.Name, Optimizer.Hyperparameters);
        }

        // Dřív se klon stavěl podle jména přes GetOptimizer, a ten zná jen vestavěné
        // optimizery — vlastní optimizer (README slibuje, že jde) proto spadl na
        // „this optimizer does not exist" hned při sestavení první vrstvy.
        public static Optimizer Clone_optimizer(Optimizer Optimizer)
        {
            return Optimizer.CreateFresh();
        }

        // Jen pro načtení ze souboru: tam je k dispozici jméno, ne instance.
        private static Optimizer GetOptimizer(string name, double[] hyperparameters)
        {
            switch (name)
            {
                case "SGD":
                    return new SGD(hyperparameters[0]);
                case "Adam":
                    return new Adam(hyperparameters[0], hyperparameters[1], hyperparameters[2]);
                case "Momentum":
                    return new Momentum(hyperparameters[0], hyperparameters[1]);
                default: throw new ArgumentException($"this optimizer ({name}) does not exist");
            }
        }
    }
}
