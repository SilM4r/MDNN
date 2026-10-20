namespace MDNN.Layers
{
    internal struct PoolingIndex
    {
        public int Row;
        public int Col;

        public PoolingIndex(int row, int col)
        {
            Row = row; 
            Col = col;
        }
    }
}
