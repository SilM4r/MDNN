using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace My_DNN.Layers.classes
{
    public struct PoolingIndex
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
