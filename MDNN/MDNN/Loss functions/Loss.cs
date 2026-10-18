namespace My_DNN.Loss_functions
{
    public abstract class Loss
    {
        public double[]? Losses
        {
            get { return losses; }
        }
        private int iteration = 0;
        private double totalLossSum = 0;
        private double[]? losses;
        public abstract string Name { get; }

        public abstract double LossFunction(double value, double target_value);

        public abstract double DerivativeOfLossFunction(double value, double target_value);

        // True pokud DerivativeOfLossFunction vrací rovnou fúzovaný gradient dL/dz
        // pro softmax výstup (tj. s - t). Backprop pak přeskočí násobení softmax derivací.
        public virtual bool RequiresSoftmax => false;

        public void CalculateLoss(double[] value, double[] target_value)
        {
            int n = value.Length;

            losses = new double[n];

            for (int i = 0; i < n; i++)
            {
                double lossValue = LossFunction(value[i], target_value[i]);
                losses[i] = lossValue;
                totalLossSum += lossValue;
            }

            iteration++;

        }

        public double CalculateAndGetLoss(double[] value, double[] target_value)
        {
            int n = value.Length;

            double lossSum = 0;

            for (int i = 0; i < n; i++)
            {
                lossSum += LossFunction(value[i], target_value[i]);
            }

            return lossSum;
        }

        public double GetResetAverageLossPerIteration()
        {
            if (iteration == 0)
            {
                return 0;
            }

            double output = totalLossSum / iteration;
            iteration = 0;
            totalLossSum = 0;

            return output;
        }

        public double GetAverageLossPerIteration()
        {  
            if (iteration == 0)
            {
                return 0;
            }

            return totalLossSum / iteration;
        }

        public void ResetAverageLossPerIteration()
        {
            iteration = 0;
            totalLossSum = 0;
        }

        // Replika pro paralelní trénink: tatáž funkce, vlastní (prázdné) akumulátory.
        // Loss je kromě akumulátorů bezstavová, takže mělká kopie stačí a funguje
        // i pro vlastní loss funkce — na rozdíl od stavění podle jména.
        internal Loss CreateReplica()
        {
            Loss replica = (Loss)MemberwiseClone();
            replica.iteration = 0;
            replica.totalLossSum = 0;
            replica.losses = null;
            return replica;
        }

        // Přičte nasbíranou loss repliky k vlastní a repliku vynuluje. `Losses` (po prvcích,
        // z posledního vzorku) se převezme, jen když replika nějaký vzorek počítala.
        internal void TakeAccumulatedLossFrom(Loss replica)
        {
            totalLossSum += replica.totalLossSum;
            iteration += replica.iteration;

            if (replica.losses != null)
            {
                losses = replica.losses;
            }

            replica.totalLossSum = 0;
            replica.iteration = 0;
            replica.losses = null;
        }

        internal static bool IsBuiltIn(string? name) => name is "Cross Entropy" or "MSE";

        // Loss pro načítaný model — stejná pravidla jako Optimizer.ResolveForLoad.
        internal static Loss ResolveForLoad(string? savedName, Loss? own, bool ownWins)
        {
            if (own != null && own.Name == savedName && (ownWins || !IsBuiltIn(savedName)))
            {
                return own;
            }

            if (own != null && ownWins)
            {
                throw new ArgumentException(
                    $"Model se ukládal s loss funkcí „{savedName}\", předaná je „{own.Name}\".");
            }

            if (!IsBuiltIn(savedName) && !string.IsNullOrEmpty(savedName))
            {
                throw new ArgumentException(
                    $"Loss funkce „{savedName}\" není vestavěná, načtení ji podle jména neumí postavit. " +
                    "Předej její instanci: MDNN.LoadModel(cesta, loss: new …).");
            }

            return inicialization_Loss_func(savedName);
        }

        // Loss podle jména uloženého v souboru s modelem. Dřív neznámé jméno tiše propadlo
        // do MSE: model s vlastní loss se načetl bez chyby a dotrénoval úplně jinou účelovou
        // funkcí. Stejná chyba jako u optimizeru, jen bez výjimky. Teď je to chyba.
        public static Loss inicialization_Loss_func(string? name_of_loss_function)
        {
            switch (name_of_loss_function)
            {
                case "Cross Entropy":
                    return new CrossEntropy();
                case "MSE":
                    return new MSE();
                case null or "":
                    throw new ArgumentException("V souboru s modelem chybí jméno loss funkce.");
                default:
                    throw new ArgumentException(
                        $"Loss funkce „{name_of_loss_function}\" není vestavěná. Model s vlastní " +
                        "loss zatím nejde načíst ze souboru — načítání ji hledá podle jména " +
                        "a zná jen MSE a Cross Entropy.");
            }
        }

    }
}
