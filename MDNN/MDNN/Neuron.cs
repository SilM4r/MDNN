using My_DNN.Save_neural_network;
using My_DNN.Optimizers;

namespace My_DNN
{

    public class Neuron
    {
        public double[] Weights
        {
            get { return weights; }
        }
        public double Bias
        {
            get { return bias; }
        }

        
        // Stav neuronu jde zvenku ČÍST (výpis vah, výstupů… při učení se, jak síť počítá),
        // ale zapisuje do něj jen knihovna. Dřív to byla veřejná pole a kdo nastavil třeba
        // `inputs` nebo `optimizer` mimo pořadí, které vrstva čeká, dostal tiše špatný výpočet.
        //
        // Výstup a vstupy jsou z POSLEDNÍHO průchodu neuronem.
        public double Output { get; internal set; }
        public double RawOutput { get; internal set; }   // před aktivací (w · x + b)
        public double[] Inputs { get; internal set; }

        public Activation_func Activation_Func { get; internal set; }
        public Optimizer Optimizer { get; internal set; }

        private double[] weights;
        private double bias;
        
        internal double[] gradientsW;
        // internal kvůli paritním testům proti PyTorchi — bez čtení gradientu biasu
        // by se dal porovnat jen gradient vah. Veřejné API zůstává beze změny.
        internal double gradientsB;

        private int mini_batch_size = 0;

        // `random` = zdroj náhody pro inicializaci vah. Předává ho vrstva ze svého
        // NetworkContextu (kvůli reprodukovatelnosti přes seed); null = samostatný neuron
        // bez modelu, spadne se na sdílený globální generátor jako dřív.
        public Neuron(int Number_of_input, Activation_func activation_function, Random? random = null)
        {
            Random rnd = random ?? GeneralNeuralNetworkSettings.rnd;

            weights = new double[Number_of_input];
            Inputs = new double[Number_of_input];
            gradientsW = new double[Number_of_input];

            Output = 0;

            // limit je pro všechny váhy stejný → počítat ho ve smyčce bylo zbytečné
            double limit = Number_of_input > 0 ? Math.Sqrt(6.0 / Number_of_input) : 0;

            for (int i = 0; i < Number_of_input; i++)
            {
                Weights[i] = (rnd.NextDouble() * 2 - 1) * limit;
            }

            bias = 0;

            inicializationGradients();
            Optimizer = Optimizer.Clone_optimizer(GeneralNeuralNetworkSettings.optimizer);
            Activation_Func = activation_function;
        }

        public Neuron(ExportNeuron exportNeuron, Activation_func activation_function)
        {

            weights = exportNeuron.Weights;
            Inputs = new double[Weights.Length];
            gradientsW = new double[Weights.Length];

            bias = exportNeuron.Bias;

            Output = 0;

            inicializationGradients();
            Optimizer = Optimizer.Clone_optimizer(GeneralNeuralNetworkSettings.optimizer);
            Activation_Func = activation_function;

            // Navázání tréninku: optimizer pokračuje se stavem, se kterým se model uložil.
            if (exportNeuron.OptimizerState != null)
            {
                Optimizer.ImportState(exportNeuron.OptimizerState);
            }
        }


        // Nastavení konkrétních vah a biasu zvenčí. Potřebné pro paritní testy proti
        // PyTorchi: obě strany musí startovat z BITOVĚ stejných parametrů, jinak by se
        // porovnávala náhodná inicializace, ne výpočet. `Weights` je sice veřejné, ale
        // jen ke čtení (vrací pole, takže by šlo mutovat prvek po prvku) a `Bias` je
        // scalar bez setteru — tohle obojí obsluhuje na jednom místě.
        internal void SetParamsForTests(double[] newWeights, double newBias)
        {
            if (newWeights.Length != weights.Length)
                throw new ArgumentException(
                    $"Expected {weights.Length} weights, got {newWeights.Length}.");

            Array.Copy(newWeights, weights, weights.Length);
            bias = newBias;
        }

        public double feedForward(double[] values)
        {
            Inputs = values;

            // Skalární součin w · x. Dvě věci, obě kvůli rychlosti:
            //
            // 1) Sčítá se do LOKÁLNÍCH proměnných, ne rovnou do pole `raw_output`. Pole objektu
            //    znamená zápis do paměti v každé iteraci; lokální součet zůstane v registru
            //    (změřeno: 3,75× rychlejší).
            //
            // 2) Čtyři nezávislé součty místo jednoho. S jedním každé sčítání čeká, až doběhne
            //    předchozí (sčítání v plovoucí čárce má latenci několika taktů), takže procesor
            //    stojí. Čtyři řetězy běží souběžně (změřeno: 2,45× rychlejší, SIMD navíc nedá
            //    nic). Matematicky je to tentýž součet, jen v jiném pořadí — výsledek se proto
            //    může lišit v posledním bitu (~1e-16) od sčítání zleva doprava.
            double s0 = 0, s1 = 0, s2 = 0, s3 = 0;
            int i = 0;
            for (; i <= values.Length - 4; i += 4)
            {
                s0 += values[i] * weights[i];
                s1 += values[i + 1] * weights[i + 1];
                s2 += values[i + 2] * weights[i + 2];
                s3 += values[i + 3] * weights[i + 3];
            }

            double sum = (s0 + s1) + (s2 + s3);
            for (; i < values.Length; i++)   // zbytek, když počet vstupů není dělitelný 4
            {
                sum += values[i] * weights[i];
            }

            RawOutput = sum + bias;
            Output = Activation_Func.Apply(RawOutput);

            return Output;
        }

        public void backPropagation(double e, double L)
        {

            for (int i = 0; i < Weights.Length; i++)
            {
                Weights[i] -= L * e * Inputs[i];
            }
            bias -= L * e * 1;
        }
        // ---------------------------------------------------------------- repliky
        //
        // Replika pro data-paralelní trénink (Train.Workers > 1). Pole vah SDÍLÍ s originálem
        // — během dávky se váhy jen čtou a mění je až UpdateParams originálu, po dávce, kdy
        // repliky stojí. Díky tomu se váhy mezi dávkami nemusí kopírovat. Vlastní má
        // všechno, co se mění během průchodu: gradienty, čítač dávky, vstupy a výstupy.
        // Bias je skalár, takže sdílet nejde — přenáší ho SyncBiasFrom před každou dávkou.
        internal Neuron CreateReplica()
        {
            Neuron replica = (Neuron)MemberwiseClone();
            replica.Inputs = new double[weights.Length];
            replica.Output = 0;
            replica.RawOutput = 0;
            replica.gradientsW = new double[weights.Length];
            replica.gradientsB = 0;
            replica.mini_batch_size = 0;
            return replica;
        }

        internal void SyncBiasFrom(Neuron original)
        {
            bias = original.bias;
        }

        // Přičte gradienty a čítač repliky k tomuto neuronu a repliku vynuluje. Čítač se
        // přičítat MUSÍ — Update_weights_bias jím dělí, takže bez něj by se gradient
        // celé dávky vydělil jen počtem vzorků, které prošly originálem (tedy nulou).
        internal void TakeGradientsFrom(Neuron replica)
        {
            for (int i = 0; i < gradientsW.Length; i++)
            {
                gradientsW[i] += replica.gradientsW[i];
                replica.gradientsW[i] = 0;
            }
            gradientsB += replica.gradientsB;
            replica.gradientsB = 0;

            mini_batch_size += replica.mini_batch_size;
            replica.mini_batch_size = 0;
        }

        // Neuron byl v tomhle vzorku vypnutý dropoutem: nepočítal se ve forwardu a jeho
        // gradient je nulový, takže není co akumulovat. Čítač dávky se ale zvýšit MUSÍ —
        // gradienty se dělí právě jím, a kdyby vypnuté vzorky nepočítal, dělilo by se
        // menším číslem a gradient by vyšel nafouknutý.
        internal void SkipSample()
        {
            mini_batch_size++;
        }

        public void Calculate_gradients_of_W_B(double e)
        {

            for (int i = 0; i < Weights.Length; i++)
            {
                gradientsW[i] += e * Inputs[i];
            }
            gradientsB += e ;

            mini_batch_size++;
        }

        // RTRL: gradient parametru = okamžitý gradient shora (gImm) × citlivost tohoto parametru.
        // sensW[i] = ∂h/∂Weights[i], sensB = ∂h/∂bias (obojí drží a posouvá RNN vrstva).
        public void AccumulateGradient(double gImm, double[] sensW, double sensB)
        {
            for (int i = 0; i < Weights.Length; i++)
            {
                gradientsW[i] += gImm * sensW[i];
            }
            gradientsB += gImm * sensB;

            mini_batch_size++;
        }

        // weightDecay = koeficient L2. Vrstva ho bere z NetworkContext.L2; 0 = vypnuto.
        public void Update_weights_bias(double weightDecay = 0)
        {
            // Prázdná dávka je legitimní stav (UpdateParams bez předchozího backpropu).
            // Bez tohohle se dělilo nulou → 0/0 = NaN → tiše celý model NaN.
            if (mini_batch_size == 0)
            {
                return;
            }

            // váhy + bias (bias má index Weights.Length, viz níž)
            Optimizer.EnsureCapacity(Weights.Length + 1);

            for (int i = 0; i < Weights.Length; i++)
            {
                // L2 se přičítá k UŽ ZPRŮMĚROVANÉMU gradientu, ne k součtu přes dávku —
                // jinak by síla regularizace závisela na velikosti minibatche.
                double gradient = gradientsW[i] / mini_batch_size + weightDecay * Weights[i];
                Weights[i] = Optimizer.Update(Weights[i], gradient, i);
            }

            // Bias se NEregularizuje: netvaruje rozhodovací hranici, jen ji posouvá,
            // takže jeho tažení k nule model nezjednoduší — jen zhorší.
            bias = Optimizer.Update(bias, gradientsB / mini_batch_size,Weights.Length);

            mini_batch_size = 0;

            inicializationGradients();
        }

        // `randomSource`: stejná logika jako v konstruktoru — když ho volající nedodá,
        // použije se globální generátor. (Neuron sám Context nevidí, takže si ho musí
        // podat ten, kdo mutaci spouští.)
        public void Mutate_params(int chance_of_mutation, int percent_mutation, Random? randomSource = null)
        {
            Random random = randomSource ?? GeneralNeuralNetworkSettings.rnd;

            for (int i = 0; i < Weights.Length; i++)
            {
                int randomValueBetween0And100 = random.Next(101);
                if (randomValueBetween0And100 < chance_of_mutation)
                {
                    // Rozmezí je -percentage/2 až +percentage/2
                    double minPercentage = -percent_mutation / 100.0 / 2.0;
                    double maxPercentage = percent_mutation / 100.0 / 2.0;

                    // Vygeneruj náhodnou hodnotu mezi minPercentage a maxPercentage
                    double randomPercentage = minPercentage + (random.NextDouble() * (maxPercentage - minPercentage));

                    // Změň hodnotu x o náhodné procento
                    Weights[i] += Weights[i] * randomPercentage;
                }
            }
        }

        private void inicializationGradients()
        {
            for (int i = 0; i < gradientsW.Length; i++)
            {
                gradientsW[i] = 0;
            }
            gradientsB = 0;
        }
    }
}