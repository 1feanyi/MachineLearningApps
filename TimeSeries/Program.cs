using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Transforms.TimeSeries;
using System;
using System.Data.SqlClient;
using System.IO;
using System.Linq;

namespace TimeSeries
{
    internal class Program
    {
        /// <summary>
        /// Forecast demand for bike rentals using a univariate time series analysis algorithm known as Singular Spectrum Analysis.
        /// -----------------------------------------------------------------------------------------------------------------------
        /// Load data from a database
        /// Create a forecasting model
        /// Evaluate forecasting model
        /// Save a forecasting model
        /// Use a forecasting model
        /// </summary>
        
        private static readonly string rootDir = Path.GetFullPath(Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "../../../"));

        private static readonly string dbFilePath = Path.Combine(rootDir, "Data", "DailyDemand.mdf");
        private static readonly string modelPath = Path.Combine(rootDir, "Data", "MLModel.zip");
        private static readonly string connectionString = $"Data Source=(LocalDB)\\MSSQLLocalDB;AttachDbFilename={dbFilePath};Integrated Security=True;Connect Timeout=30;";

        private static MLContext _mlContext;
        private static TimeSeriesPredictionEngine<ModelInput, ModelOutput> _forecastEngine;

        private static void Main(string[] args)
        {
            _mlContext = new MLContext();

            // Load data from db
            IDataView dataView = LoadDataFromDB();
            var firstYearData = _mlContext.Data.FilterRowsByColumn(dataView, "Year", upperBound: 1);
            var secondYearData = _mlContext.Data.FilterRowsByColumn(dataView, "Year", lowerBound: 1);

            // Create forecasting model
            var forecaster = ProcessData(firstYearData);

            // Evaluate forecasting model
            Evaluate(secondYearData, forecaster);

            // Save model to zip file
            _forecastEngine = forecaster.CreateTimeSeriesEngine<ModelInput, ModelOutput>(_mlContext);
            _forecastEngine.CheckPoint(_mlContext, modelPath);

            // Use model for forecast
            Forecast(secondYearData, 7, _forecastEngine);
        }

        private static IDataView LoadDataFromDB()
        {
            var loader = _mlContext.Data.CreateDatabaseLoader<ModelInput>();
            string query = "SELECT RentalDate, CAST(Year as REAL) as Year, CAST(TotalRentals as REAL) as TotalRentals FROM Rentals";
            var dbSource = new DatabaseSource(SqlClientFactory.Instance, connectionString, query);
            var dataView = loader.Load(dbSource);

            return dataView;
        }

        private static SsaForecastingTransformer ProcessData(IDataView trainingDataView)
        {
            var forecastingPipeline = _mlContext.Forecasting.ForecastBySsa(
                            outputColumnName: "ForecastedRentals",
                            inputColumnName: "TotalRentals",
                            windowSize: 7,
                            seriesLength: 30,
                            trainSize: 365,
                            horizon: 7,
                            confidenceLevel: 0.95f,
                            confidenceLowerBoundColumn: "LowerBoundRentals",
                            confidenceUpperBoundColumn: "UpperBoundRentals");

            return forecastingPipeline.Fit(trainingDataView);
        }

        public static void Evaluate(IDataView testData, ITransformer model)
        {
            var predictions = model.Transform(testData);

            var actual = _mlContext.Data.CreateEnumerable<ModelInput>(testData, true)
                                        .Select(observed => observed.TotalRentals);

            var forecast = _mlContext.Data.CreateEnumerable<ModelOutput>(predictions, true)
                                         .Select(prediction => prediction.ForecastedRentals[0]);

            // Calculate 'error' : difference between actual and forecast values
            var metrics = actual.Zip(forecast, (actualValue, forecastValue) => actualValue - forecastValue);

            // Measure performance by computing MAE and RMSE
            // MAE: Measures how close predictions are to the actual value. Value 0 -> infinity. Close to 0 is better.
            // RMSE: Summarizes error in the model. Values 0 -> infinity. Close to 0 is better.
            var MAE = metrics.Average(error => Math.Abs(error)); // Mean Absolute Error
            var RMSE = Math.Sqrt(metrics.Average(error => Math.Pow(error, 2))); // Root Mean Squared Error

            Console.WriteLine("Evaluation Metrics");
            Console.WriteLine("---------------------");
            Console.WriteLine($"Mean Absolute Error: {MAE:F3}");
            Console.WriteLine($"Root Mean Squared Error: {RMSE:F3}\n");
        }

        private static void Forecast(IDataView testData, int horizon, TimeSeriesPredictionEngine<ModelInput, ModelOutput> forecaster)
        {
            var forecast = forecaster.Predict();

            // Align actual & forecast values for seven periods
            var forecastOutput = _mlContext.Data.CreateEnumerable<ModelInput>(testData, reuseRowObject: false)
                .Take(horizon)
                .Select((ModelInput rental, int index) =>
                {
                    string rentalDate = rental.RentalDate.ToShortDateString();
                    float actualRentals = rental.TotalRentals;
                    float lowerEstimate = Math.Max(0, forecast.LowerBoundRentals[index]);
                    float estimate = forecast.ForecastedRentals[index];
                    float upperEstimate = forecast.UpperBoundRentals[index];
                    return $"Date: {rentalDate}\n" +
                    $"Actual Rentals: {actualRentals}\n" +
                    $"Lower Estimate: {lowerEstimate}\n" +
                    $"Forecast: {estimate}\n" +
                    $"Upper Estimate: {upperEstimate}";
                });

            Console.WriteLine("Rental Forecast");
            Console.WriteLine("--------------------");
            foreach (var prediction in forecastOutput)
            {
                Console.WriteLine(prediction);
                Console.WriteLine("");
            }
        }
    }
}