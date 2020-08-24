using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Transforms;
using System;
using System.Collections.Generic;
using System.IO;

namespace BinaryTF
{
    partial class Program
    {
        /// <summary>
        /// Classify and analyze sentiment from website comments using pre-trained TensorFlow model.
        /// </summary>
        public const int FeatureLength = 600;

        private static readonly string _modelPath = Path.Combine(Environment.CurrentDirectory, "Model");

        private static void Main(string[] args)
        {
            var mlContext = new MLContext();
            ITransformer model = Train(mlContext);

            PredictSentiment(mlContext, model);
        }

        private static ITransformer Train(MLContext mlContext)
        {
            IDataView lookupMap = LoadAndMapData(mlContext);
            var ResizeFeaturesAction = ResizeFeatures();
            var tensorFlowModel = mlContext.Model.LoadTensorFlowModel(_modelPath);
            var pipeline = CreatePipeline(mlContext, lookupMap, ResizeFeaturesAction, tensorFlowModel);

            // Create model from pipeline
            // no need to fit model(empty) since TF model is pre-trained
            var dataView = mlContext.Data.LoadFromEnumerable(new List<MovieReview>());
            ITransformer model = pipeline.Fit(dataView);
            return model;
        }

        private static Action<VariableLength, FixedLength> ResizeFeatures()
        {
            // resize the variable length word integer array to an integer array of fixed size
            return (s, f) =>
            {
                var features = s.VariableLengthFeatures;
                Array.Resize(ref features, FeatureLength);
                f.Features = features;
            };
        }

        /// <summary>
        /// Create a dictionary to encode words as integers
        /// by using the LoadFromTextFile method to load mapping data from a file
        /// Create lookup map
        /// </summary>
        /// <param name="mlContext"></param>
        /// <returns></returns>
        private static IDataView LoadAndMapData(MLContext mlContext)
        {
            return mlContext.Data.LoadFromTextFile(Path.Combine(_modelPath, "imdb_word_index.csv"),
                columns: new[]
                {
                    new TextLoader.Column("Words", DataKind.String, 0),
                    new TextLoader.Column("Ids", DataKind.Int32, 1),
                },
                separatorChar: ',');
        }

        /// <summary>
        /// Create pipeline and split input text into words
        /// using TokenizeIntoWords transform to break the text into words
        /// </summary>
        /// <param name="mlContext"></param>
        /// <param name="lookupMap"></param>
        /// <param name="ResizeFeaturesAction"></param>
        /// <param name="tensorFlowModel"></param>
        /// <returns></returns>
        private static IEstimator<ITransformer> CreatePipeline(MLContext mlContext, IDataView lookupMap, Action<VariableLength, FixedLength> ResizeFeaturesAction, TensorFlowModel tensorFlowModel)
        {
            return mlContext.Transforms.Text.TokenizeIntoWords("TokenizedWords", "ReviewText")
                .Append(mlContext.Transforms.Conversion.MapValue("VariableLengthFeatures", lookupMap,
                            lookupMap.Schema["Words"], lookupMap.Schema["Ids"], "TokenizedWords"))
                .Append(mlContext.Transforms.CustomMapping(ResizeFeaturesAction, "Resize"))
                .Append(tensorFlowModel.ScoreTensorFlowModel("Prediction/Softmax", "Features"))
                .Append(mlContext.Transforms.CopyColumns("Prediction", "Prediction/Softmax"));
        }

        public static void PredictSentiment(MLContext mlContext, ITransformer model)
        {
            var engine = mlContext.Model.CreatePredictionEngine<MovieReview, MovieReviewSentimentPrediction>(model);

            do
            {
                Console.WriteLine("What did you think about the movie 'Home Alone 2?'");
                var uInput = Console.ReadLine();

                var review = new MovieReview()
                {
                    ReviewText = uInput
                };

                var sentimentPrediction = engine.Predict(review);

                Console.WriteLine("Review: {0}", review.ReviewText.ToString());
                Console.WriteLine("Is sentiment/review positive? {0}", sentimentPrediction.Prediction[1] >= 0.5 ? "Yes." : "No.");
                Console.WriteLine("Prediction Score: {0}", sentimentPrediction.Prediction[1]);
                Console.WriteLine("Disclaimer: This tensorflow sentiment model is utter shit - Iffy");
                Console.WriteLine("Press 'q' to quit");
            } while (Console.ReadLine().ToLower() != "q");
        }
    }
}