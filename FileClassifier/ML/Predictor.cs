using FileClassifier.ML.Base;
using FileClassifier.ML.Objects;
using Microsoft.ML;
using System;
using System.IO;

namespace FileClassifier.ML
{
    public class Predictor : BaseML
    {
        public void Predict(string inputDataFile)
        {
            // validate that input file exists
            if (!File.Exists(inputDataFile))
            {
                Console.WriteLine($"Failed to find input data at {inputDataFile}");
                return;
            }

            ITransformer mlModel;

            using (var stream = new FileStream(ModelPath, FileMode.Open, FileAccess.Read, FileShare.Read))
            {
                mlModel = mlContext.Model.Load(stream, out _);
            }

            if (mlModel == null)
            {
                Console.WriteLine("Failed to load model");
                return;
            }

            var predictionEngine = mlContext.Model.CreatePredictionEngine<FileInput, FilePrediction>(mlModel);

            var prediction = predictionEngine.Predict(new FileInput
            {
                Strings = GetStrings(File.ReadAllBytes(inputDataFile))
            });

            Console.WriteLine(
                $"Based on the file ({inputDataFile}) the file is classified as {(prediction.IsMalicious ? "malicious" : "benign")}" +
                $" at a confidence level of {prediction.Probability:P0}");
        }
    }
}