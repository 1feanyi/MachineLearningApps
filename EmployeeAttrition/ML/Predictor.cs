using EmployeeAttrition.ML.Base;
using EmployeeAttrition.ML.Objects;
using Microsoft.ML;
using Newtonsoft.Json;
using System;
using System.IO;

namespace EmployeeAttrition.ML
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

            var predictionEngine = mlContext.Model.CreatePredictionEngine<EmploymentHistory, EmploymentHistoryPrediction>(mlModel);

            var json = File.ReadAllText(inputDataFile);

            var prediction = predictionEngine.Predict(JsonConvert.DeserializeObject<EmploymentHistory>(json));

            Console.WriteLine(
                $"Based on input json: {Environment.NewLine}" +
                $"{json}{Environment.NewLine}" +
                $"The employee is predicted to work for {prediction.DurationInMonths:#.##} months");
        }
    }
}