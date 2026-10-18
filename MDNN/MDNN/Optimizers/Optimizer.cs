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

        // Stav optimizeru pro uložení modelu: pojmenovaná pole (u Adamu "m", "v", "t"),
        // jedna hodnota na parametr. Bez něj se načtený model rozjíždí od nuly — Adam má
        // vynulované momenty a krok 0, takže první kroky po načtení jsou velké jako na
        // začátku tréninku a navázání není plynulé.
        //
        // null = optimizer stav nemá (SGD), nebo ho neumí uložit (vlastní optimizer, který
        // tyhle dvě metody nepřepíše) — načte se pak s prázdným stavem jako dřív.
        public virtual Dictionary<string, double[]>? ExportState() => null;

        // Protějšek ExportState. Dostane KOPII polí, takže si je smí nechat.
        public virtual void ImportState(Dictionary<string, double[]> state) { }

        // Pole ze souboru: musí existovat a mít stejnou délku jako ostatní. Jinak je soubor
        // poškozený a pokračovat by znamenalo počítat s nesmyslným stavem.
        protected static double[] RequireState(Dictionary<string, double[]> state, string key, int? expectedLength)
        {
            if (!state.TryGetValue(key, out double[]? values))
            {
                throw new ArgumentException($"Ve stavu optimizeru chybí pole „{key}\".");
            }

            if (expectedLength != null && values.Length != expectedLength)
            {
                throw new ArgumentException(
                    $"Pole „{key}\" ve stavu optimizeru má {values.Length} hodnot, ostatní {expectedLength}.");
            }

            return (double[])values.Clone();
        }

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

        // Umí ho načtení postavit jen podle jména ze souboru?
        internal static bool IsBuiltIn(string name) => name is "SGD" or "Adam" or "Momentum";

        // Optimizer pro načítaný model. `own` = optimizer, který má volající po ruce:
        // předaný do LoadModel, nebo ten, který model už má (obnova snapshotu v TrainLoop).
        //   - předaný do LoadModel vyhrává vždy, jen musí sedět jméno (jinak by se stav
        //     ze souboru načítal do optimizeru, který s ním nemá nic společného);
        //   - vestavěný se jinak postaví ze jména a hyperparametrů ze souboru;
        //   - vlastní, který postavit neumíme, převezme `own` se stejným jménem.
        internal static Optimizer ResolveForLoad(ExportOptimizer saved, Optimizer? own, bool ownWins)
        {
            if (own != null && own.Name == saved.Name && (ownWins || !IsBuiltIn(saved.Name)))
            {
                return own;
            }

            if (own != null && ownWins)
            {
                throw new ArgumentException(
                    $"Model se ukládal s optimizerem „{saved.Name}\", předaný je „{own.Name}\". " +
                    "Stav optimizeru ze souboru by se načítal do jiného optimizeru.");
            }

            if (!IsBuiltIn(saved.Name))
            {
                throw new ArgumentException(
                    $"Optimizer „{saved.Name}\" není vestavěný, načtení ho podle jména neumí postavit. " +
                    "Předej jeho instanci: MDNN.LoadModel(cesta, optimizer: new …).");
            }

            return Refactor_optimizer(saved);
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
