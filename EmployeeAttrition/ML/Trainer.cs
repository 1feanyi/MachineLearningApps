using EmployeeAttrition.Common;
using EmployeeAttrition.ML.Base;
using EmployeeAttrition.ML.Objects;
using Microsoft.ML;
using System;
using System.IO;

namespace EmployeeAttrition.ML
{
    public class Trainer : BaseML
    {
        public void Train(string trainingFileName)
        {
            if (!File.Exists(trainingFileName))
            {
                Console.WriteLine($"Failed to find training data file ({trainingFileName})");
                return;
            }

            var trainingDataView = mlContext.Data.LoadFromTextFile<EmploymentHistory>(trainingFileName, ',');
            
            var dataSplit = mlContext.Data.TrainTestSplit(trainingDataView, testFraction: 0.2);

            var dataProcessPipeline = mlContext.Transforms.CopyColumns("Label", nameof(EmploymentHistory.DurationInMonths))
                .Append(mlContext.Transforms.NormalizeMeanVariance(nameof(EmploymentHistory.IsMarried)))
                .Append(mlContext.Transforms.NormalizeMeanVariance(nameof(EmploymentHistory.BSDegree)))
                .Append(mlContext.Transforms.NormalizeMeanVariance(nameof(EmploymentHistory.MSDegree)))
                .Append(mlContext.Transforms.NormalizeMeanVariance(nameof(EmploymentHistory.YearsExperience)))
                .Append(mlContext.Transforms.NormalizeMeanVariance(nameof(EmploymentHistory.AgeAtHire)))
                .Append(mlContext.Transforms.NormalizeMeanVariance(nameof(EmploymentHistory.HasKids)))
                .Append(mlContext.Transforms.NormalizeMeanVariance(nameof(EmploymentHistory.WithinMonthOfVesting)))
                .Append(mlContext.Transforms.NormalizeMeanVariance(nameof(EmploymentHistory.DeskDecorations)))
                .Append(mlContext.Transforms.NormalizeMeanVariance(nameof(EmploymentHistory.LongCommute)))
                .Append(mlContext.Transforms.Concatenate("Features", typeof(EmploymentHistory).ToPropertyList<EmploymentHistory>(nameof(EmploymentHistory.DurationInMonths))));

            var trainer = mlContext.Regression.Trainers.Sdca(labelColumnName: "Label", featureColumnName: "Features");

            var trainingPipeline = dataProcessPipeline.Append(trainer);

            ITransformer trainedModel = trainingPipeline.Fit(dataSplit.TestSet);
            mlContext.Model.Save(trainedModel, dataSplit.TrainSet.Schema, ModelPath);

            var testSetTransform = trainedModel.Transform(dataSplit.TestSet);

            var modelMetrics = mlContext.Regression.Evaluate(testSetTransform);

            Console.WriteLine(
                $"Loss Function: {modelMetrics.LossFunction:0.##}{Environment.NewLine}" +
                $"Mean Absolute Error: {modelMetrics.MeanAbsoluteError:#.##}{Environment.NewLine}" +
                $"Mean Squared Error: {modelMetrics.MeanSquaredError:#.##}{Environment.NewLine}" +
                $"RSquared: {modelMetrics.RSquared:0.##}{Environment.NewLine}" +
                $"Root Mean Squared Error: {modelMetrics.RootMeanSquaredError:#.##}");
        }
    }
}