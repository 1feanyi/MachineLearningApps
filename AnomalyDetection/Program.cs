using System;
using System.IO;
using Microsoft.ML;
using System.Collections.Generic;
using Microsoft.ML.Transforms.TimeSeries;

namespace AnomalyDetection
{
    internal class Program
    {
        /// <summary>
        /// Detect anomaly in product sales data
        /// ------------------------------------
        /// Load the data
        /// Create a transform for spike anomaly detection
        /// Detect spike anomalies with the transform
        /// Create a transform for change point anomaly detection
        /// Detect change point anomalies with the transform
        /// </summary>
        static readonly string _dataPath = Path.Combine(Environment.CurrentDirectory, "Data", "product-sales.csv");
        const int _docsize = 36; // assign the number of records in dataset file to constant variable

        private static void Main(string[] args)
        {
            var mlContext = new MLContext();

            // Load Data
            var dataView = mlContext.Data.LoadFromTextFile<ProductSalesData>(path: _dataPath, hasHeader: true, separatorChar: ',');
            //DetectSpike(mlContext, _docsize, dataView);
            DetectChangePoint(mlContext, _docsize, dataView);
        }

        /// <summary>
        /// Spike detection identifies sudden yet temporary bursts that differ significantly from the majority of the time series data values
        /// </summary>
        /// <param name="mlContext"></param>
        /// <param name="docSize"></param>
        /// <param name="productSales"></param>
        static void DetectSpike(MLContext mlContext, int docSize, IDataView productSales)
        {
            var iidSpikeEstimator = mlContext.Transforms.DetectIidSpike(outputColumnName: nameof(ProductSalesPrediction.Prediction), inputColumnName: nameof(ProductSalesData.NumSales), confidence: 95, pvalueHistoryLength: docSize / 4);
            var iidSpikeTransform = iidSpikeEstimator.Fit(CreateEmptyView(mlContext));
            var tranformedData = iidSpikeTransform.Transform(productSales);
            var predictions = mlContext.Data.CreateEnumerable<ProductSalesPrediction>(tranformedData, reuseRowObject: false);

            // Alert: spike alert for given datapoint; Score: productSales value for given datapoint in dataset; P-Value: probability, closer to zero is anomaly
            Console.WriteLine("Alert\tScore\tP-Value");

            foreach (var p in predictions)
            {
                var results = $"{p.Prediction[0]}\t{p.Prediction[1]:f2}\t{p.Prediction[2]:f2}";

                if (p.Prediction[0] == 1)
                {
                    results += "<---- Spike detected";
                }

                Console.WriteLine(results);
            }
            Console.WriteLine("");
        }

        /// <summary>
        /// Change point detection identifies persistent changes in a time series event stream distribution of values like level changes/trends
        /// </summary>
        /// <param name="mlContext"></param>
        /// <param name="docSize"></param>
        /// <param name="productSales"></param>
        static void DetectChangePoint(MLContext mlContext, int docSize, IDataView productSales)
        {
            var iidChangePointEstimator = mlContext.Transforms.DetectIidChangePoint(outputColumnName: nameof(ProductSalesPrediction.Prediction), inputColumnName: nameof(ProductSalesData.NumSales), confidence: 95, changeHistoryLength: docSize / 4);
            var iidChangePointTransform = iidChangePointEstimator.Fit(CreateEmptyView(mlContext));
            var transformedData = iidChangePointTransform.Transform(productSales);
            var predictions = mlContext.Data.CreateEnumerable<ProductSalesPrediction>(transformedData, reuseRowObject: false);

            // Alert: change point alert for given data point; 
            // Score: ProductSales value for a given data point in dataset;
            // P-value: probability value, closer to zero likely to be anomaly; 
            // Martingale value: shows how weird a datapoint is based on the sequence of P-Values
            Console.WriteLine("Alert\tScore\tP-Value\tMartingale Value");

            foreach (var p in predictions)
            {
                var results = $"{p.Prediction[0]}\t{p.Prediction[1]:f2}\t{p.Prediction[2]:F2}\t{p.Prediction[3]:F2}";

                if (p.Prediction[0] == 1)
                {
                    results += "<--- Alert is on, predicted changepoint";
                }
                Console.WriteLine(results);
            }
            Console.WriteLine("");
        }

        /// <summary>
        /// Product an empty dataview object with correct schema to be used as input to IEstimator.Fit()
        /// </summary>
        /// <param name="mlContext"></param>
        /// <returns></returns>
        static IDataView CreateEmptyView(MLContext mlContext)
        {
            var enumerableData = new List<ProductSalesData>();
            return mlContext.Data.LoadFromEnumerable(enumerableData);
        }
    }
}