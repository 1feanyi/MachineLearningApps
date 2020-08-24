using Microsoft.ML;
using System;
using System.IO;
using System.Threading.Tasks;

namespace Classification
{
    internal class Program
    {
        /// <summary>
        /// Classification is a machine learning task that uses data to determine the category, type, or class of an item
        /// or row of data.
        /// Below is a classifier to train a model that classifies and predicts the Area label for a GitHub issue
        /// Multiclass classification: multiple categories that can be predicted by using a single model.
        /// </summary>
        static string _trainDataPath => Path.Combine(Environment.CurrentDirectory, "Data", "issues_train.tsv");

        static string _testDataPath => Path.Combine(Environment.CurrentDirectory, "Data", "issues_test.tsv");
        static string _modelPath => Path.Combine(Environment.CurrentDirectory, "Models", "model.zip");

        private static MLContext _mlContext;
        private static PredictionEngine<GitHubIssue, IssuePrediction> _predEngine;
        private static ITransformer _trainedModel;
        private static IDataView _trainingDataView;

        private static async Task Main(string[] args)
        {
            _mlContext = new MLContext(seed: 0);
            _trainingDataView = _mlContext.Data.LoadFromTextFile<GitHubIssue>(_trainDataPath, hasHeader: true);

            var pipeline = ProcessData();
            var trainingPipeline = await BuildAndTrainModelAsync(_trainingDataView, pipeline);

            await EvaluateAsync(_trainingDataView.Schema);
            PredictIssue();

            Console.ReadLine();
        }

        /// <summary>
        /// Extracts and transforms the data.
        /// Returns the processing pipeline.
        /// </summary>
        /// <returns></returns>
        public static IEstimator<ITransformer> ProcessData()
        {
            var pipeline = _mlContext.Transforms.Conversion.MapValueToKey(inputColumnName: "Area", outputColumnName: "Label")
                .Append(_mlContext.Transforms.Text.FeaturizeText(inputColumnName: "Title", outputColumnName: "TitleFeaturized"))
                .Append(_mlContext.Transforms.Text.FeaturizeText(inputColumnName: "Description", outputColumnName: "DescriptionFeaturized"))
                .Append(_mlContext.Transforms.Concatenate("Features", "TitleFeaturized", "DescriptionFeaturized"))
                .AppendCacheCheckpoint(_mlContext); // never use for large datasets

            return pipeline;
        }

        /// <summary>
        /// Creates the training algorithm class.
        /// Trains the model.
        /// Predicts area based on training data.
        /// Returns the model.
        /// </summary>
        /// <param name="trainingDataView"></param>
        /// <param name="pipeline"></param>
        /// <returns></returns>
        public static async Task<IEstimator<ITransformer>> BuildAndTrainModelAsync(IDataView trainingDataView, IEstimator<ITransformer> pipeline)
        {
            var trainingPipeline = pipeline.Append(_mlContext.MulticlassClassification.Trainers.SdcaMaximumEntropy("Label", "Features"))
                                                 .Append(_mlContext.Transforms.Conversion.MapKeyToValue("PredictedLabel"));

            _trainedModel = await Task.Run(() => trainingPipeline.Fit(trainingDataView));
            _predEngine = _mlContext.Model.CreatePredictionEngine<GitHubIssue, IssuePrediction>(_trainedModel);

            var issue = new GitHubIssue()
            {
                Title = "WebSockets communication is slow on my machine",
                Description = "The WebSockets communication used under the covers by SignalR looks like it is going slow in my development machine.."
            };

            var prediction = _predEngine.Predict(issue);

            Console.WriteLine($"=============== Single Prediction just-trained-model - Result: {prediction.Area} ===============");

            return trainingPipeline;
        }

        /// <summary>
        /// Loads the test dataset.
        /// Creates the multiclass evaluator.
        /// Evaluates the model and create metrics.
        /// Displays the metrics.
        /// </summary>
        /// <param name="trainingDataViewSchema"></param>
        public static async Task EvaluateAsync(DataViewSchema trainingDataViewSchema)
        {
            var testDataView = await Task.Run(() => _mlContext.Data.LoadFromTextFile<GitHubIssue>(_testDataPath, hasHeader: true));
            var testMetrics = _mlContext.MulticlassClassification.Evaluate(_trainedModel.Transform(testDataView));

            Console.WriteLine($"******************************************************************************************");
            Console.WriteLine($"*       Metrics for Multi-class Classification model - Test Data     ");
            Console.WriteLine($"*-----------------------------------------------------------------------------------------");
            Console.WriteLine($"*       MicroAccuracy:    {testMetrics.MicroAccuracy:0.###}");
            Console.WriteLine($"*       MacroAccuracy:    {testMetrics.MacroAccuracy:0.###}");
            Console.WriteLine($"*       LogLoss:          {testMetrics.LogLoss:#.###}");
            Console.WriteLine($"*       LogLossReduction: {testMetrics.LogLossReduction:#.###}");
            Console.WriteLine($"******************************************************************************************");

            SaveModelAsFile(_mlContext, trainingDataViewSchema, _trainedModel);
        }

        /// <summary>
        /// Serialize and store the trained model as a zip file.
        /// </summary>
        /// <param name="mlContext"></param>
        /// <param name="trainingDataViewSchema"></param>
        /// <param name="model"></param>
        private static void SaveModelAsFile(MLContext mlContext, DataViewSchema trainingDataViewSchema, ITransformer model)
        {
            mlContext.Model.Save(model, trainingDataViewSchema, _modelPath);
        }

        /// <summary>
        /// Loads the saved model
        /// Creates a single issue of test data.
        /// Predicts Area based on test data.
        /// Combines test data and predictions for reporting.
        /// Displays the predicted results.
        /// </summary>
        private static void PredictIssue()
        {
            var loadedModel = _mlContext.Model.Load(_modelPath, out var modelInputSchema);

            var singleIssue = new GitHubIssue() { Title = "Entity Framework crashes", Description = "When connecting to the database, EF is crashing" };

            // PredictionEngine is not thread-safe. Use in prototype
            // In production use PredictionEnginePool instead
            _predEngine = _mlContext.Model.CreatePredictionEngine<GitHubIssue, IssuePrediction>(loadedModel);

            var prediction = _predEngine.Predict(singleIssue);

            Console.WriteLine($"=============== Single Prediction - Result: {prediction.Area} ===============");
        }
    }
}