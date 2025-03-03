using System.Runtime.InteropServices;


namespace mdnn
{
    
    public static class GPUManager
    {
        [DllImport("gpu.dll", CallingConvention = CallingConvention.Cdecl)]
        public static extern void LayerCalculation(float[] a, float[] b, float[] c, float[] result, int size, int quantity);

        public static float[] GPUCalculation(float[] a, float[] b, float[] c, float[] result, int size, int quantity) 
        {
            if (a.Length != size)
                throw new ArgumentException("Array 'a' length must match 'size'.");
            if (b.Length != size * quantity)
                throw new ArgumentException("Array 'b' length must match 'size * quantity'.");
            if (c.Length != quantity)
                throw new ArgumentException("Array 'c' length must match 'quantity'.");
            if (result.Length != quantity)
                throw new ArgumentException("Array 'result' length must match 'quantity'.");

            LayerCalculation(a, b, c, result, size, quantity);
            return result;
        }
    }
}
