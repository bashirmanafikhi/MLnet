using Microsoft.Extensions.ML;
using SentimentAnalysisWebAPI.models;

var builder = WebApplication.CreateBuilder(args);

builder.Services.AddPredictionEnginePool<ModelInput, ModelOutput>()
    .FromFile(modelName: "SentimentAnalysisModel", filePath: "sentiment_model.zip", watchForChanges: true);

var app = builder.Build();

app.MapPost("/predict", async (PredictionEnginePool<ModelInput, ModelOutput> predictionEnginePool, ModelInput input) =>
        await Task.FromResult(predictionEnginePool.Predict(modelName: "SentimentAnalysisModel", input)));

app.Run();
