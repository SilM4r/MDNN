using My_DNN.Optimizers;

namespace My_DNN.Layers.classes
{
    public abstract class Layer
    {
        // Reference na kontext modelu, do kterého vrstva patří (nastaví LayerManager při připojení).
        public My_DNN.NetworkContext? Context { get; set; }

        // Je tohle poslední vrstva v modelu? Udržuje LayerManager při každé změně řetězu.
        // Dnes to potřebuje jediná věc: GLOBÁLNÍ dropout se na výstupní vrstvu NESMÍ
        // aplikovat — zahazovat výstupy sítě není regularizace, je to poškozování predikce.
        // Explicitně zadaný dropout na konkrétní vrstvě tenhle zákaz přebije (je to volba).
        public bool IsOutputLayer { get; internal set; }

        abstract public string Name { get; }

        abstract public int[] Input_size_and_shape { get; }
        abstract public int[] Output_size_and_shape { get; }
        abstract public Tensor? Layer_output { get; }
        abstract public Tensor? Layer_raw_output { get; }
        abstract public Activation_func Activation_Func { get; set; }
        abstract public Tensor FeedForward(Tensor input_values);
        abstract public void UpdateParams();
        // Stavba vrstvy je rozdělená na DVĚ fáze, které spolu nesouvisí:
        //
        //   WireShapes()           dopočítá tvary (vstup, výstup, buffery). Potřebuje to
        //                          LayerManager, aby mohl zřetězit vrstvy, a smí se to
        //                          volat kolikrát chce — nic to nelosuje.
        //   MaterializeParameters() vytvoří a VYLOSUJE parametry. Volá se právě jednou,
        //                          jedním průchodem zleva doprava, před prvním forwardem.
        //
        // Proč rozdělené: dokud to bylo v jedné metodě, materializace se spouštěla už při
        // skládání modelu. Pořadí losování tak bylo funkcí HISTORIE volání (Add/Insert),
        // ne výsledné architektury — dvě stejné sítě se stejným seedem daly různé váhy podle
        // toho, jak jsi je poskládal. Výstupní vrstva se navíc při přidání dvou vrstev
        // materializovala třikrát. A vložená vrstva se nezapojila vůbec, což u načteného
        // modelu končilo IndexOutOfRange ve forwardu.
        public void LayerAdjustment(int? number_of_elements = null, int[]? number_of_input = null)
        {
            WireShapes(number_of_elements, number_of_input);
            MaterializeParameters();
        }

        abstract public void WireShapes(int? number_of_elements = null, int[]? number_of_input = null);

        // Idempotentní: když parametry pro aktuální tvar už existují, nechá je být
        // (jinak by přepsala natrénované váhy).
        abstract public void MaterializeParameters();

        // Má vrstva parametry odpovídající svému aktuálnímu tvaru? False znamená
        // „tvar je zapojený, ale parametry ještě nejsou" nebo „tvar zatím neznáme".
        abstract public bool IsMaterialized { get; }

        // ---------------------------------------------------------------- konvence B
        //
        // Jediný vstupní bod do backwardu, který postupně nahradí trojici výš
        // (CalculateLayerGradients + SeedOutputGradient + BackPropagation).
        //
        //   vezme  ∂L/∂output této vrstvy
        //   udělá  naakumuluje gradienty VLASTNÍCH parametrů
        //   vrátí  ∂L/∂input, tedy ∂L/∂output vrstvy pod sebou
        //
        // Proti dnešku mizí posun o jednu: vrstva propaguje gradient přes SVÉ váhy
        // a o sousedovi nemusí vědět. Tím odpadají větve „další vrstva je Dense",
        // které si dnes Conv a MaxPool musí psát samy, i skrytý stav (Conv.dOutput,
        // RNN.gImm), který dnes existuje jen kvůli tomu rozdělení na dvě metody.
        //
        // `preActivation` = gradOutput UŽ JE ∂L/∂z, takže se derivace aktivace přeskočí.
        // Jediný případ je fúze softmax+CrossEntropy, kde loss vrací rovnou `s − t`:
        // softmax tam logicky patří k loss funkci, ne k vrstvě (stejně jako
        // CrossEntropyLoss v PyTorchi obsahuje log-softmax). Díky tomu zůstává
        // jeden vstupní bod, ne dva.
        //
        // `computeInputGradient` = false znamená „∂L/∂input nikdo nepotřebuje, nepočítej ho".
        // Platí pro PRVNÍ vrstvu v modelu: pod ní už nic není, takže její vstupní gradient
        // jde rovnou do koše. U Conv je to plná korelace přes celý obrázek, tedy nejdražší
        // operace backwardu — vyhozená zbytečně. (Totéž dělal i kód před konvencí B, jen
        // to nebylo vidět.) Obdoba `needs_input_grad` v PyTorchi.
        //
        // Default je true, tedy vždycky správně; vypnout to smí jen ten, kdo VÍ, že
        // návratovou hodnotu nepoužije. Při false se vrací PRÁZDNÝ tenzor, ne nesmyslný.
        abstract public Tensor Backward(Tensor gradOutput, bool preActivation = false,
                                        bool computeInputGradient = true);

        // ---------------------------------------------------------------- repliky
        //
        // Data-paralelní trénink (Train.Workers > 1): každé vlákno má vlastní repliku modelu
        // a počítá svůj kus dávky. Replika SDÍLÍ parametry s originálem (během dávky se jen
        // čtou) a má VLASTNÍ všechno, co se mění během průchodu — aktivace, masky, gradienty.
        //
        // null = vrstva paralelní trénink neumí. Výchozí hodnota je schválně null, ne
        // MemberwiseClone: mělká kopie by sdílela pracovní buffery a dvě vlákna by si
        // navzájem přepisovala aktivace — tiše, bez pádu. Trénink to proto odmítne.
        internal virtual Layer? CreateReplica() => null;

        // Před každou dávkou: přenést do repliky parametry, které nejdou sdílet odkazem.
        internal virtual void SyncReplica(Layer replica) { }

        // Po dávce: přičíst gradienty repliky k vlastním a repliku vynulovat.
        internal virtual void TakeGradientsFrom(Layer replica) { }

        // Model dostal jiný optimizer (NetworkContext.Optimizer) — vrstva si vezme jeho
        // čerstvý klon. Vrstvy bez parametrů nemají co měnit.
        internal virtual void ReplaceOptimizer(Optimizer prototype) { }

    }
}
