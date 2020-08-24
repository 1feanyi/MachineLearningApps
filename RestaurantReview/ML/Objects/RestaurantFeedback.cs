using Microsoft.ML.Data;
using System;
using System.Collections.Generic;
using System.Text;

namespace RestaurantReview.ML.Objects
{
    // Input properties
    public class RestaurantFeedback
    {
        [LoadColumn(0)]
        public bool Label { get; set; } // 0 => positive feedback; 1=> negative feedback

        [LoadColumn(1)]
        public string Text { get; set; }
    }
}
