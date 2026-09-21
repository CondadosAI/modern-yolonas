"""Self-supervised pretraining of the backbone, on images with no labels.

The Apache-2.0 weights plan reaches the backbone twice from outside the detection
loss: stage 1 distils a DINOv3 teacher, and this module contrasts two views of the
same image. They are not alternatives. Distillation needs a teacher and inherits
whatever that teacher saw; SSL needs only images, which is the situation most
people are actually in -- a pile of unlabelled frames from the domain they care
about, and a COCO checkpoint that was trained on something else.

The method is DenseCL (Wang et al., 2021, arXiv:2011.09157). MoCo-style
contrastive learning optimises one vector per image, which is the wrong granularity
for a detector: it teaches the backbone to summarise a photo, not to keep positions
apart. DenseCL adds a second contrastive term between *pixels* of the two views,
matched by feature similarity, and reports +1.1 AP over MoCo-v2 when both are
pretrained on COCO. That dense term is the reason this module exists rather than a
three-line SimCLR.

Building blocks come from ``lightly`` (MIT), installed via the ``ssl`` extra. The
correspondence step is written here: ``lightly`` ships DenseCL's heads and its
transform, but not the cross-view matching.

**Known departure from the paper.** MoCo shuffles BatchNorm statistics across GPUs
so the query and key encoders cannot communicate through their BN buffers. On a
single GPU there is nothing to shuffle, and the backbone has BN everywhere, so the
key branch's statistics leak. The published fix needs multi-GPU; on one card the
usual mitigation is a large memory bank, which is what the default gives. Expect
this to cost some of the paper's margin.
"""

from __future__ import annotations

import copy
from pathlib import Path

import lightning as L
import torch
from torch import Tensor, nn

from modern_yolonas.training.scheduler import cosine_with_warmup

_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _require_lightly():
    """Import lightly, or explain how to get it.

    It is an optional dependency: pretraining is one workflow among many and most
    installs never run it.
    """
    try:
        from lightly.loss import NTXentLoss
        from lightly.models.modules.heads import DenseCLProjectionHead
        from lightly.models.utils import deactivate_requires_grad, update_momentum
        from lightly.transforms import DenseCLTransform
    except ImportError as exc:  # pragma: no cover - exercised by the install, not the suite
        raise ImportError(
            "`yolonas pretrain` needs lightly. Install it with:\n"
            "    uv pip install 'modern-yolonas[ssl]'\n"
            "lightly is MIT-licensed, so it places no condition on the weights."
        ) from exc
    return (
        NTXentLoss,
        DenseCLProjectionHead,
        deactivate_requires_grad,
        update_momentum,
        DenseCLTransform,
    )


class UnlabelledImageFolder(torch.utils.data.Dataset):
    """Every image under *root*, recursively, with no annotations.

    Two augmented views per item, as the contrastive loss requires. The transform
    is built by the caller so the normalisation stays in one place.
    """

    def __init__(self, root: str | Path, transform):
        self.root = Path(root)
        self.paths = sorted(
            p for p in self.root.rglob("*") if p.suffix.lower() in _IMAGE_SUFFIXES
        )
        if not self.paths:
            raise ValueError(
                f"No images under {self.root}. Looked for {sorted(_IMAGE_SUFFIXES)} "
                f"recursively."
            )
        self.transform = transform

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int):
        from PIL import Image

        image = Image.open(self.paths[index]).convert("RGB")
        view0, view1 = self.transform(image)
        return view0, view1


def build_transform(input_size: int = 448, min_scale: float = 0.2):
    """DenseCL's augmentations, without ImageNet normalisation.

    ``normalize=None`` is deliberate and load-bearing. The detection path scales
    ``uint8`` by 1/255 on the device and applies no mean/std, so a backbone
    pretrained on ImageNet-normalised inputs would meet a shifted input
    distribution at the first convolution the moment it was fine-tuned. Matching
    the two is free here and silent to debug later.
    """
    (_, _, _, _, DenseCLTransform) = _require_lightly()
    return DenseCLTransform(input_size=input_size, min_scale=min_scale, normalize=None)


class DenseCLPretrainModule(L.LightningModule):
    """DenseCL over the backbone's stride-16 features.

    The detection model comes along whole, held at ``self.model``, so the
    checkpoint carries ``model.backbone.*`` keys and ``--init-backbone`` reads it
    with no translation -- the same contract the distillation stage writes to.
    Only the backbone is optimised; the neck and heads are untouched passengers.

    Args:
        model: A ``YoloNAS``. Its backbone is trained.
        lr: Peak learning rate.
        momentum: EMA factor for the key encoder.
        temperature: Softmax temperature for both contrastive terms.
        memory_bank_size: Negatives for the global term. The dense term gets this
            scaled by ``dense_bank_multiplier``, because it draws one query per
            *pixel*: a bank sized for images turns over every few steps and its
            negatives degenerate into near-duplicates of the current batch.
        dense_bank_multiplier: How much larger the dense bank is.
        lambda_dense: Weight of the dense term. The paper uses 0.5.
        warmup_steps: Linear warmup before the cosine decay.
        max_steps: Total steps, for the schedule.
    """

    def __init__(
        self,
        model: nn.Module,
        lr: float = 0.03,
        weight_decay: float = 1e-4,
        momentum: float = 0.999,
        temperature: float = 0.2,
        memory_bank_size: int = 4096,
        dense_bank_multiplier: int = 4,
        lambda_dense: float = 0.5,
        projection_dim: int = 128,
        warmup_steps: int = 1000,
        max_steps: int = 100_000,
    ):
        super().__init__()
        (
            NTXentLoss,
            DenseCLProjectionHead,
            deactivate_requires_grad,
            update_momentum,
            _,
        ) = _require_lightly()
        self._update_momentum = update_momentum

        self.model = model
        channels = model.backbone.out_channels[2]

        # Two heads, as in the paper: one over the pooled vector, one per pixel.
        # Sharing a head would force one projection to serve both granularities,
        # which is the failure DenseCL exists to fix.
        self.global_head = DenseCLProjectionHead(channels, channels, projection_dim)
        self.dense_head = DenseCLProjectionHead(channels, channels, projection_dim)

        self.backbone_momentum = copy.deepcopy(model.backbone)
        self.global_head_momentum = copy.deepcopy(self.global_head)
        self.dense_head_momentum = copy.deepcopy(self.dense_head)
        for module in (
            self.backbone_momentum,
            self.global_head_momentum,
            self.dense_head_momentum,
        ):
            deactivate_requires_grad(module)

        self.global_loss = NTXentLoss(
            temperature=temperature, memory_bank_size=(memory_bank_size, projection_dim)
        )
        self.dense_loss = NTXentLoss(
            temperature=temperature,
            memory_bank_size=(memory_bank_size * dense_bank_multiplier, projection_dim),
        )

        self.lr = lr
        self.weight_decay = weight_decay
        self.momentum = momentum
        self.lambda_dense = lambda_dense
        self.warmup_steps = warmup_steps
        self.max_steps = max_steps
        self.save_hyperparameters(ignore=["model"])

    def _query(self, images: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Backbone tokens, global projection, dense projection."""
        _, _, c4, _ = self.model.backbone(images)
        tokens = c4.flatten(2).transpose(1, 2)  # [B, HW, C]
        pooled = self.global_head(c4.mean(dim=(2, 3)))  # [B, D]
        dense = self.dense_head(tokens)  # [B, HW, D]
        return tokens, pooled, dense

    @torch.no_grad()
    def _key(self, images: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """The momentum branch. No gradient reaches it by construction."""
        _, _, c4, _ = self.backbone_momentum(images)
        tokens = c4.flatten(2).transpose(1, 2)
        pooled = self.global_head_momentum(c4.mean(dim=(2, 3)))
        dense = self.dense_head_momentum(tokens)
        return tokens, pooled, dense

    @staticmethod
    def match(query_tokens: Tensor, key_tokens: Tensor) -> Tensor:
        """For each query pixel, the index of its most similar key pixel.

        The match is computed on the *backbone* features, not the projections.
        This is the step that makes DenseCL dense: the two views are different
        crops, so pixel ``i`` of one has no positional counterpart in the other,
        and the correspondence has to be found by appearance.

        Args:
            query_tokens: ``[B, HW, C]``.
            key_tokens: ``[B, HW, C]``.

        Returns:
            ``[B, HW]`` indices into the key's pixels.
        """
        q = nn.functional.normalize(query_tokens, dim=2)
        k = nn.functional.normalize(key_tokens, dim=2)
        return torch.einsum("bic,bjc->bij", q, k).argmax(dim=2)

    def _step(self, batch, stage: str) -> Tensor:
        view0, view1 = batch
        if view0.dtype == torch.uint8:  # pragma: no cover - transform emits float
            view0 = view0.float().div_(255.0)
            view1 = view1.float().div_(255.0)

        query_tokens, query_global, query_dense = self._query(view0)
        key_tokens, key_global, key_dense = self._key(view1)

        global_loss = self.global_loss(query_global, key_global)

        # lambda_dense = 0 is documented as "reduces to MoCo-v2", so it has to
        # actually skip the dense branch. Weighting it by zero instead would still
        # run the [B, HW, HW] correspondence matmul and still consume slots in the
        # dense memory bank -- DenseCL with a dead term, not MoCo-v2.
        if self.lambda_dense == 0:
            self.log(f"{stage}/loss", global_loss, prog_bar=True, sync_dist=True)
            self.log(f"{stage}/global_loss", global_loss, sync_dist=True)
            return global_loss

        indices = self.match(query_tokens, key_tokens)
        batch_size, pixels, dim = query_dense.shape
        matched = torch.gather(
            key_dense, 1, indices.unsqueeze(-1).expand(-1, -1, dim)
        )
        dense_loss = self.dense_loss(
            query_dense.reshape(batch_size * pixels, dim),
            matched.reshape(batch_size * pixels, dim),
        )

        loss = (1 - self.lambda_dense) * global_loss + self.lambda_dense * dense_loss
        self.log(f"{stage}/loss", loss, prog_bar=True, sync_dist=True)
        self.log(f"{stage}/global_loss", global_loss, sync_dist=True)
        self.log(f"{stage}/dense_loss", dense_loss, sync_dist=True)
        return loss

    def training_step(self, batch, batch_idx) -> Tensor:
        # Before the forward, so the key encoder the loss sees is the updated one.
        self._update_momentum(self.model.backbone, self.backbone_momentum, m=self.momentum)
        self._update_momentum(self.global_head, self.global_head_momentum, m=self.momentum)
        self._update_momentum(self.dense_head, self.dense_head_momentum, m=self.momentum)
        return self._step(batch, "train")

    def validation_step(self, batch, batch_idx) -> Tensor:
        return self._step(batch, "val")

    def configure_optimizers(self):
        parameters = (
            list(self.model.backbone.parameters())
            + list(self.global_head.parameters())
            + list(self.dense_head.parameters())
        )
        # SGD with momentum, as MoCo/DenseCL use; AdamW is the distillation stage's
        # choice because a regression loss behaves differently from a contrastive one.
        optimizer = torch.optim.SGD(
            parameters, lr=self.lr, momentum=0.9, weight_decay=self.weight_decay
        )
        scheduler = cosine_with_warmup(
            optimizer, warmup_steps=self.warmup_steps, total_steps=self.max_steps
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
        }
