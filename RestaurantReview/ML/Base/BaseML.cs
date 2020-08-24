using Microsoft.ML;
using RestaurantReview.Common;
using System;
using System.IO;

namespace RestaurantReview.ML.Base
{
    public class BaseML
    {
        protected static string ModelPath => Path.Combine(AppContext.BaseDirectory, Constants.MODEL_FILENAME);

        protected readonly MLContext mlContext;

        protected BaseML()
        {
            mlContext = new MLContext(2020);
        }
    }
}