using EmailCategorizer.Common;
using Microsoft.ML;
using System;
using System.IO;

namespace EmailCategorizer.ML.Base
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