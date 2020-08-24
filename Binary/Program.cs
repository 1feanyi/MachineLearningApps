using Microsoft.ML;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using static Microsoft.ML.DataOperationsCatalog;

namespace SentimentAnalysis
{
    internal class Program
    {
        /// <summary>
        /// Classify and analyze sentiment from website comments.
        /// </summary>
        private static readonly string _dataPath = Path.Combine(Environment.CurrentDirectory, "Data", "yelp_labelled.txt");

        private static void Main(string[] args)
        {
            var mlContext = new MLContext();
            TrainTestData splitDataView = LoadData(mlContext);
            ITransformer model = BuildAndTrainModel(mlContext, splitDataView.TrainSet);

            Evaluate(mlContext, model, splitDataView.TestSet);
            UseModelWithSingleItem(mlContext, model);
            UseModelWithBatchItems(mlContext, model);

            Console.ReadLine();
        }

        /// <summary>
        /// Load data
        /// Split loaded dataset into train and test datasets
        /// Return the split train and test datasets
        /// </summary>
        /// <param name="mlContext"></param>
        /// <returns></returns>
        public static TrainTestData LoadData(MLContext mlContext)
        {
            IDataView dataView = mlContext.Data.LoadFromTextFile<SentimentData>(_dataPath, hasHeader: false);

            // Specify the percentage of data for the test set with the testFractionparameter. The default is 10%.
            TrainTestData splitDataView = mlContext.Data.TrainTestSplit(dataView, testFraction: 0.2);

            return splitDataView;
        }

        /// <summary>
        /// Extract and transform data
        /// Train model
        /// Predict sentiment based on test data
        /// Return model
        /// </summary>
        /// <param name="mlContext"></param>
        /// <param name="splitTrainSet"></param>
        /// <returns></returns>
        public static ITransformer BuildAndTrainModel(MLContext mlContext, IDataView splitTrainSet)
        {
            var estimator = mlContext.Transforms.Text.FeaturizeText(outputColumnName: "Features", inputColumnName: nameof(SentimentData.SentimentText))
                                     .Append(mlContext.BinaryClassification.Trainers.SdcaLogisticRegression(labelColumnName: "Label", featureColumnName: "Features"));

            //Console.WriteLine("================= Create and Train the Model =================");
            var model = estimator.Fit(splitTrainSet);
            //Console.WriteLine("================= End of Training =================");
            //Console.WriteLine();

            return model;
        }

        /// <summary>
        /// Loads the test dataset.
        /// Creates the BinaryClassification evaluator.
        /// Evaluates the model and creates metrics.
        /// Displays the metrics.
        /// </summary>
        /// <param name="mlContext"></param>
        /// <param name="model"></param>
        /// <param name="splitTestSet"></param>
        public static void Evaluate(MLContext mlContext, ITransformer model, IDataView splitTestSet)
        {
            Console.WriteLine("================= Evaluating Model Accuracy with Test Data =================");
            IDataView predictions = model.Transform(splitTestSet);
            var metrics = mlContext.BinaryClassification.Evaluate(predictions, "Label");

            Console.WriteLine();
            Console.WriteLine("Model quality metrics evaluation");
            Console.WriteLine("--------------------------------");
            Console.WriteLine($"Accuracy: {metrics.Accuracy:P2}");      // measure accuracy
            Console.WriteLine($"Auc: {metrics.AreaUnderRocCurve:P2}");  // measure confidence. Desire: close to 1
            Console.WriteLine($"F1Score: {metrics.F1Score:P2}");        // measure balance between precision and recall. Desire: close to 1
            Console.WriteLine("================= End of model evaluation =================");
        }

        /// <summary>
        /// Creates a single comment of test data.
        /// Predicts sentiment based on test data.
        /// Combines test data and predictions for reporting.
        /// Displays the predicted results.
        /// </summary>
        /// <param name="mlcontext"></param>
        /// <param name="model"></param>
        public static void UseModelWithSingleItem(MLContext mlcontext, ITransformer model)
        {
            var predictionFunction = mlcontext.Model.CreatePredictionEngine<SentimentData, SentimentPrediction>(model);

            var sampleStatement = new SentimentData
            {
                SentimentText = "This was a very bad steak"
            };

            var resultPrediction = predictionFunction.Predict(sampleStatement);

            Console.WriteLine();
            Console.WriteLine("================= Prediction Test of Model with a Single Sample and Test Dataset =================");
            Console.WriteLine($"Sentiment: {resultPrediction.SentimentText} | Prediction: {(Convert.ToBoolean(resultPrediction.Prediction) ? "Positive" : "Negative")} | Probability: {resultPrediction.Probability}");
            Console.WriteLine("================= End of Predictions =================");
            Console.WriteLine();
        }

        /// <summary>
        /// Creates batch test data.
        /// Predicts sentiment based on test data.
        /// Combines test data and predictions for reporting.
        /// Displays the predicted results.
        /// </summary>
        /// <param name="mlContext"></param>
        /// <param name="model"></param>
        public static void UseModelWithBatchItems(MLContext mlContext, ITransformer model)
        {
            IEnumerable<SentimentData> sentiments = new[]
            {
                new SentimentData
                {
                    SentimentText = "This was a horrible meal"
                },
                new SentimentData
                {
                    SentimentText = "I love this spaghetti."
                }
            };

            var batchComments = mlContext.Data.LoadFromEnumerable(sentiments);

            var predictions = model.Transform(batchComments);

            var predictedResults = mlContext.Data.CreateEnumerable<SentimentPrediction>(predictions, reuseRowObject: false);

            Console.WriteLine();
            Console.WriteLine("================= Prediction Test of loaded Model with multiple samples =================");

            foreach (SentimentPrediction prediction in predictedResults)
            {
                Console.WriteLine($"Sentiment: {prediction.SentimentText} | Prediction: {(Convert.ToBoolean(prediction.Prediction) ? "Positive" : "Negative")} | Probability: {prediction.Probability}");
            }

            Console.WriteLine("================= End of Predictions =================");
        }
    }
}