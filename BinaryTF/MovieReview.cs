using Microsoft.ML.Data;

namespace BinaryTF
{
    partial class Program
    {
        /// <summary>
        /// Class to hold original sentiment data.
        /// </summary>
        public class MovieReview
        {
            public string ReviewText { get; set; }
        }

        /// <summary>
        /// Class to hold the variable length feature vector. Used to define the
        /// column names used as input to the custom mapping action.
        /// </summary>
        public class VariableLength
        {
            /// <summary>
            /// This is a variable length vector designated by VectorType attribute.
            /// Variable length vectors are produced by applying operations such as 'TokenizeWords' on strings
            /// resulting in vectors of tokens of variable lengths.
            /// </summary>
            [VectorType]
            public int[] VariableLengthFeatures { get; set; }
        }

        /// <summary>
        /// Class to hold the fixed length feature vector. Used to define the
        /// column names used as output from the custom mapping action,
        /// </summary>
        public class FixedLength
        {
            /// <summary>
            /// This is a fixed length vector designated by VectorType attribute.
            /// </summary>
            [VectorType(FeatureLength)]
            public int[] Features { get; set; }
        }

        /// <summary>
        /// Class to contain the output values from the transformation.
        /// </summary>
        public class MovieReviewSentimentPrediction
        {
            [VectorType(2)]
            public float[] Prediction { get; set; }
        }
    }
}