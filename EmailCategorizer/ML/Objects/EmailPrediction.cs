using Microsoft.ML.Data;

namespace EmailCategorizer.ML.Objects
{
    public class EmailPrediction
    {
        [ColumnName("PredictedLabel")]
        public string Category { get; set; }
    }
}