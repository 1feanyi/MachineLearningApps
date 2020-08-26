using FileTypeClassifier.Enums;
using FileTypeClassifier.ML.Base;
using FileTypeClassifier.ML.Objects;
using Microsoft.ML;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;

namespace FileTypeClassifier.ML
{
    public class Predictor : BaseML

    {
        private Dictionary<uint, FileTypes> GetClusterToMap(PredictionEngineBase<FileData, FileTypePrediction> predictionEngine)
        {
            var map = new Dictionary<uint, FileTypes>();

            var fileTypes = Enum.GetValues(typeof(FileTypes)).Cast<FileTypes>();

            foreach (var fileType in fileTypes)
            {
                var fileData = new FileData(fileType);
                var prediction = predictionEngine.Predict(fileData);
                map.Add(prediction.PredictedClusterId, fileType);
            }

            return map;
        }

        public void Predict(string inputDataFile)
        {
            if (!File.Exists(ModelPath))
            {
                Console.WriteLine($"Failed to find model at {ModelPath}");
                return;
            }

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

            var predictionEngine = mlContext.Model.CreatePredictionEngine<FileData, FileTypePrediction>(mlModel);

            var fileData = new FileData(File.ReadAllBytes(inputDataFile));

            var prediction = predictionEngine.Predict(fileData);

            var mapping = GetClusterToMap(predictionEngine);

            Console.WriteLine(
                $"Based on input file: {inputDataFile}{Environment.NewLine}{Environment.NewLine}" +
                $"Feature Extraction: {fileData}{Environment.NewLine}{Environment.NewLine}" +
                $"The file is predicted to be a {mapping[prediction.PredictedClusterId]}{Environment.NewLine}");

            Console.WriteLine("Distances from all clusters:");

            for (uint i = 0; i < prediction.Distances.Length; i++)
            {
                Console.WriteLine($"{mapping[i + 1]}: {prediction.Distances[i]}");
            }
        }
    }
}