using System;
using System.Collections.Generic;
using System.Text;
using Microsoft.ML.Data;

namespace AnomalyDetection
{
    class ProductSalesData
    {
        [LoadColumn(0)]
        public string Month;

        [LoadColumn(1)]
        public float NumSales;
    }

    class ProductSalesPrediction
    {
        // vector to hold alert, score, p-value, values
        [VectorType(3)]
        public double[] Prediction { get; set; }
    }
}
