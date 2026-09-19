# YoloNASEmbedder API

`YoloNASEmbedder` turns images — or boxes within them — into fixed-length vectors
from the backbone and neck, without ever running the detection head. See the
[embeddings guide](../guides/embeddings.md) for what to do with them.

::: modern_yolonas.inference.embed.YoloNASEmbedder

## Raw feature maps

If you would rather pool the feature maps yourself, the model exposes them directly.

::: modern_yolonas.model.YoloNAS.forward_features
