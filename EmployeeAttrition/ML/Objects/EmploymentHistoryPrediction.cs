using Microsoft.ML.Data;

namespace EmployeeAttrition.ML.Objects
{
    public class EmploymentHistoryPrediction
    {
        [ColumnName("Score")]
        public float DurationInMonths;
    }
}