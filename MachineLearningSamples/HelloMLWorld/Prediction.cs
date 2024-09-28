using Microsoft.ML.Data;

namespace SentimentAnalysis
{
    public class Prediction
    {
        [ColumnName("Score")]
        public float Price { get; set; }
    }
}
