using Microsoft.ML.Data;
using System;
using System.Collections.Generic;
using System.Text;

namespace FileTypeClassifier.ML.Objects
{
    public class FileTypePrediction
    {
        [ColumnName("PredictedLabel")]
        public uint PredictedClusterId { get; set; }

        [ColumnName("Score")]
        public float[] Distances { get; set; }
    }
}
