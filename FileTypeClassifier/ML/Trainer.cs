using FileTypeClassifier.ML.Base;
using FileTypeClassifier.ML.Objects;
using Microsoft.ML;
using Microsoft.ML.Data;
using System;
using System.IO;

namespace FileTypeClassifier.ML
{
    public class Trainer : BaseML
    {
        // Helper method to build IDataView object
        private IDataView GetDataView(string fileName)
        {
            return mlContext.Data.LoadFromTextFile(path: fileName,
                columns: new[]
                {
                    new TextLoader.Column(nameof(FileData.Label), DataKind.Single, 0),
                    new TextLoader.Column(nameof(FileData.isBinary), DataKind.Single, 1),
                    new TextLoader.Column(nameof(FileData.isMZHeader), DataKind.Single, 2),
                    new TextLoader.Column(nameof(FileData.isPKHeader), DataKind.Single, 3)
                },
                hasHeader: false, separatorChar: ',');
        }

        public void Train(string trainingFileName, string testingFileName)
        {
            // Check if files exist
            if (!File.Exists(trainingFileName))
            {
                Console.WriteLine($"Failed to find training data file ({trainingFileName})");
                return;
            }

            if (!File.Exists(testingFileName))
            {
                Console.WriteLine($"Failed to find test data file ({testingFileName})");
                return;
            }

            // Build data process pipeline by transforming the columns into a single Features column
            var trainingDataView = GetDataView(trainingFileName);

            var dataProcessPipeline = mlContext.Transforms.Concatenate(
                FEATURES, nameof(FileData.isBinary), nameof(FileData.isMZHeader), nameof(FileData.isPKHeader));

            // Create trainer with a cluster size of 3 and save created model
            var trainer = mlContext.Clustering.Trainers.KMeans(featureColumnName: FEATURES, numberOfClusters: 3);
            var trainingPipeline = dataProcessPipeline.Append(trainer);
            var trainedModel = trainingPipeline.Fit(trainingDataView);

            mlContext.Model.Save(trainedModel, trainingDataView.Schema, ModelPath);

            // Evaluate the trained model
            var testingDataView = GetDataView(testingFileName);

            IDataView testDataView = trainedModel.Transform(testingDataView);

            var modelMetrics = mlContext.Clustering.Evaluate(
                data: testDataView,
                labelColumnName: "Label",
                scoreColumnName: "Score",
                featureColumnName: FEATURES);

            // Display metrics
            Console.WriteLine(
                $"Average Distance: {modelMetrics.AverageDistance}{Environment.NewLine}" +
                $"Davies Bould Index: {modelMetrics.DaviesBouldinIndex}{Environment.NewLine}" +
                $"Normalized Mutual Information: {modelMetrics.NormalizedMutualInformation}");
        }
    }
}