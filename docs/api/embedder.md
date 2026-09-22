# Embeddings API

`YoloNASEmbedder` turns images — or boxes within them — into fixed-length vectors from
the backbone and neck, without ever running the detection head. For boxes *and*
embeddings from a single forward pass, use
[`YoloNASDetector.predict`](detector.md) with `Task` flags instead.

See the [embeddings guide](../guides/embeddings.md) for layer choice, the single-pass
API and why letterbox padding is excluded from pooling.

## Task flags

::: modern_yolonas.inference.embed.Task

::: modern_yolonas.inference.embed.Prediction

## YoloNASEmbedder

::: modern_yolonas.inference.embed.YoloNASEmbedder

## Pooling

::: modern_yolonas.inference.embed.FeaturePooler

## Raw feature maps

If you would rather pool the feature maps yourself, the model exposes them directly.

::: modern_yolonas.model.YoloNAS.forward_features
