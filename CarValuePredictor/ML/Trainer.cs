using CarValuePredictor.Common;
using CarValuePredictor.ML.Base;
using CarValuePredictor.ML.Objects;
using Microsoft.ML;
using System;
using System.IO;

namespace CarValuePredictor.ML
{
    public class Trainer : BaseML
    {
        public void Train(string trainingFileName, string testFileName)
        {
            if (!File.Exists(trainingFileName))
            {
                Console.WriteLine($"Failed to find training data file ({trainingFileName})");
                return;
            }

            if (!File.Exists(testFileName))
            {
                Console.WriteLine($"Failed to find test data file ({testFileName})");
                return;
            }

            var trainingDataView = mlContext.Data.LoadFromTextFile<CarInventory>(trainingFileName, ',', hasHeader: false);

            // Create pipeline and model then save model
            var dataProcessPipeline = mlContext.Transforms
                                               .Concatenate("Features", typeof(CarInventory)
                                                    .ToPropertyList<CarInventory>(nameof(CarInventory.Label)))
                                               .Append(mlContext.Transforms
                                                    .NormalizeMeanVariance(
                                                        inputColumnName: "Features",
                                                        outputColumnName: "FeaturesNormalizedByMeanVar"));

            var trainer = mlContext.BinaryClassification.Trainers.FastTree(
                labelColumnName: nameof(CarInventory.Label),
                featureColumnName: "FeaturesNormalizedByMeanVar",
                numberOfLeaves: 2,
                numberOfTrees: 1000,
                minimumExampleCountPerLeaf: 1,
                learningRate: 0.2);

            var trainingPipeline = dataProcessPipeline.Append(trainer);

            var trainedModel = trainingPipeline.Fit(trainingDataView);

            mlContext.Model.Save(trainedModel, trainingDataView.Schema, ModelPath);

            // Evaluate model
            var evaluationPipeline = trainedModel.Append(mlContext.Transforms
                .CalculateFeatureContribution(trainedModel.LastTransformer)
                .Fit(dataProcessPipeline.Fit(trainingDataView).Transform(trainingDataView)));

            var testDataView = mlContext.Data.LoadFromTextFile<CarInventory>(testFileName, ',', hasHeader: false);

            var testSetTransform = evaluationPipeline.Transform(testDataView);

            var modelMetrics = mlContext.BinaryClassification.Evaluate(
                    data: testSetTransform,
                    labelColumnName: nameof(CarInventory.Label),
                    scoreColumnName: "Score");

            // Output Display
            Console.WriteLine($"Accuracy: {modelMetrics.Accuracy:P2}");
            Console.WriteLine($"Area Under Curve: {modelMetrics.AreaUnderRocCurve:P2}");
            Console.WriteLine($"Area under Precision recall Curve: {modelMetrics.AreaUnderPrecisionRecallCurve:P2}");
            Console.WriteLine($"F1Score: {modelMetrics.F1Score:P2}");
            Console.WriteLine($"LogLoss: {modelMetrics.LogLoss:#.##}");
            Console.WriteLine($"LogLossReduction: {modelMetrics.LogLossReduction:#.##}");
            Console.WriteLine($"PositivePrecision: {modelMetrics.PositivePrecision:#.##}");
            Console.WriteLine($"PositiveRecall: {modelMetrics.PositiveRecall:#.##}");
            Console.WriteLine($"NegativePrecision: {modelMetrics.NegativePrecision:#.##}");
            Console.WriteLine($"NegativeRecall: {modelMetrics.NegativeRecall:P2}");
        }
    }
}