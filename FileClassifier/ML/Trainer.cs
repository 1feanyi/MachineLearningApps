using FileClassifier.ML.Base;
using FileClassifier.ML.Objects;
using Microsoft.ML;
using System;
using System.IO;

namespace FileClassifier.ML
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

            var trainingDataView = mlContext.Data.LoadFromTextFile<FileInput>(trainingFileName);

            var dataSplit = mlContext.Data.TrainTestSplit(trainingDataView, testFraction: 0.2);

            var dataProcessPipeline = mlContext.Transforms.CopyColumns("Label", nameof(FileInput.Label))
                .Append(mlContext.Transforms.Text.FeaturizeText("NGrams", nameof(FileInput.Strings)))
                .Append(mlContext.Transforms.Concatenate("Features", "NGrams"));

            var trainer = mlContext.BinaryClassification.Trainers.SdcaLogisticRegression(labelColumnName: "Label", featureColumnName: "Features");

            var trainingPipeline = dataProcessPipeline.Append(trainer);

            ITransformer trainedModel = trainingPipeline.Fit(dataSplit.TestSet);
            mlContext.Model.Save(trainedModel, dataSplit.TrainSet.Schema, ModelPath);
        }
    }
}