# Extending modern-yolonas

## Custom dataset

Create a dataset class with a `load_raw(index)` method that returns `(image, targets)`:

```python
import numpy as np
from torch.utils.data import Dataset

class MyDataset(Dataset):
    def __init__(self, root, transforms=None, input_size=640):
        self.root = root
        self.transforms = transforms
        self.input_size = input_size
        self.samples = self._load_samples()

    def _load_samples(self):
        # Return list of (image_path, label_path) tuples
        ...

    def load_raw(self, index):
        """Load raw image and targets without transforms."""
        img_path, label_path = self.samples[index]
        image = cv2.imread(img_path)
        targets = np.loadtxt(label_path).reshape(-1, 5)  # [cls, x, y, w, h]
        return image, targets

    def __getitem__(self, index):
        image, targets = self.load_raw(index)
        if self.transforms:
            image, targets = self.transforms(image, targets)
        return image, targets

    def __len__(self):
        return len(self.samples)
```

## Custom transform

Transforms follow a simple protocol — a callable that takes and returns `(image, targets)`:

```python
class MyAugmentation:
    def __init__(self, probability=0.5):
        self.p = probability

    def __call__(self, image, targets):
        if random.random() < self.p:
            # Modify image and targets
            image = my_transform(image)
        return image, targets
```

Use with `Compose`:

```python
from modern_yolonas.data.transforms import Compose, LetterboxResize, Normalize

transforms = Compose([
    MyAugmentation(probability=0.3),
    LetterboxResize(640),
    Normalize(),
])
```

## Frozen backbone fine-tuning

```python
from modern_yolonas import yolo_nas_s

model = yolo_nas_s(pretrained=True)

# Freeze backbone
for param in model.backbone.parameters():
    param.requires_grad = False

# Only train neck + heads
trainable = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.AdamW(trainable, lr=1e-4)
```

## Custom number of classes

```python
from modern_yolonas.model import YoloNAS

model = YoloNAS.from_config("yolo_nas_s", num_classes=5)
```

When changing `num_classes`, the pretrained classification heads no longer match
(different tensor shapes). The preferred way to handle this is `transfer_to()`,
which loads the backbone and neck **strictly** and then swaps in freshly
initialised heads:

```python
from modern_yolonas.weights import transfer_to

model = transfer_to("yolo_nas_s", num_classes=5)
```

Because the pretrained part is loaded strictly, nothing can be silently skipped.

The alternative is a partial load, which keeps every tensor whose shape matches
the model and leaves the reshaped heads at their initial values:

```python
from modern_yolonas.weights import load_pretrained

model = YoloNAS.from_config("yolo_nas_s", num_classes=5)
load_pretrained(model, "yolo_nas_s", strict=False)
```

Prefer `transfer_to()` unless you specifically need the partial-load behaviour — a
non-strict load will also quietly tolerate a checkpoint that does not belong to
this architecture at all.
