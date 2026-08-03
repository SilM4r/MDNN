namespace My_DNN.Activation_functions
{
    public abstract class LayerActivationFunc : Activation_func
    {
        public override bool Apply_to_layer => true;

        public abstract double[] ApplyToLayer(double[] values);

        // Backward pro aktivaci působící na CELOU vrstvu: dostane surové výstupy vrstvy
        // a gradient shora (∂L/∂output) a vrátí ∂L/∂raw_output.
        //
        // Nahrazuje dřívější DerivativeForLayer(), která vracela jen DIAGONÁLU Jacobiánu
        // a per-prvkově se pak dosazovala přes Derivative(double). Pro softmax je to špatně:
        // ∂L/∂z_i = Σ_j (∂L/∂s_j)·J[j,i], tedy součin s CELÝM Jacobiánem, ne jen s J[i,i].
        // (Změřeno proti numerice: ~24 % chyba u softmax+MSE. Fúzovaná cesta softmax+CE
        // sem nechodí, proto to dosud nikdo neviděl.)
        public abstract double[] BackwardForLayer(double[] rawOutput, double[] gradFromAbove);
    }
}
