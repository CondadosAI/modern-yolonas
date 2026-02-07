"""Tests for quantization: PTQ preparation, QAT preparation, conversion."""

import pytest
import torch

from modern_yolonas import yolo_nas_s
from modern_yolonas.nn.repvgg import QARepVGGBlock
from modern_yolonas.quantization.config import default_qconfig_mapping, qat_qconfig_mapping
from modern_yolonas.quantization.prepare import (
    QuantizableBackboneNeck,
    fuse_repvgg_for_quantization,
    prepare_model_ptq,
    prepare_model_qat,
)
from modern_yolonas.quantization.convert import convert_quantized


@pytest.fixture
def model():
    return yolo_nas_s(pretrained=False)


# ── RepVGG fusion ──────────────────────────────────────────────────


class TestRepVGGFusion:
    def test_partial_fusion_all_blocks(self, model):
        fuse_repvgg_for_quantization(model)
        for m in model.modules():
            if isinstance(m, QARepVGGBlock):
                assert m.partially_fused
                assert not m.fully_fused
                assert not hasattr(m, "branch_3x3")
                assert not hasattr(m, "branch_1x1")

    def test_fused_model_output_matches(self, model):
        model.eval()
        x = torch.randn(1, 3, 640, 640)
        with torch.no_grad():
            bboxes_before, scores_before = model(x)

        fuse_repvgg_for_quantization(model)
        with torch.no_grad():
            bboxes_after, scores_after = model(x)

        torch.testing.assert_close(bboxes_before, bboxes_after, atol=1e-4, rtol=1e-4)
        torch.testing.assert_close(scores_before, scores_after, atol=1e-4, rtol=1e-4)


# ── FX tracing ─────────────────────────────────────────────────────


class TestBackboneNeckTracing:
    def test_fx_traceable(self, model):
        fuse_repvgg_for_quantization(model)
        bn = QuantizableBackboneNeck(model.backbone, model.neck)
        bn.eval()

        from torch.fx import symbolic_trace

        traced = symbolic_trace(bn)
        assert traced is not None

        x = torch.randn(1, 3, 640, 640)
        with torch.no_grad():
            p3, p4, p5 = traced(x)
        assert p3.shape[0] == 1
        assert p4.shape[0] == 1
        assert p5.shape[0] == 1


# ── PTQ preparation ───────────────────────────────────────────────


class TestPTQPrepare:
    def test_output_shapes(self, model):
        ptq_model = prepare_model_ptq(model)
        ptq_model.eval()
        x = torch.randn(1, 3, 640, 640)
        with torch.no_grad():
            pred_bboxes, pred_scores = ptq_model(x)
        assert pred_bboxes.shape == (1, 8400, 4)
        assert pred_scores.shape == (1, 8400, 80)

    def test_has_observers(self, model):
        ptq_model = prepare_model_ptq(model)
        observer_count = sum(
            1
            for m in ptq_model.backbone_neck.modules()
            if "observer" in type(m).__name__.lower()
        )
        assert observer_count > 0

    def test_original_model_unchanged(self, model):
        has_unfused = any(
            isinstance(m, QARepVGGBlock) and not m.partially_fused
            for m in model.modules()
        )
        prepare_model_ptq(model)
        still_has_unfused = any(
            isinstance(m, QARepVGGBlock) and not m.partially_fused
            for m in model.modules()
        )
        assert has_unfused == still_has_unfused


# ── QAT preparation ───────────────────────────────────────────────


class TestQATPrepare:
    def test_output_shapes_eval(self, model):
        qat_model = prepare_model_qat(model)
        qat_model.eval()
        x = torch.randn(1, 3, 640, 640)
        with torch.no_grad():
            pred_bboxes, pred_scores = qat_model(x)
        assert pred_bboxes.shape == (1, 8400, 4)
        assert pred_scores.shape == (1, 8400, 80)

    def test_has_fakequant(self, model):
        qat_model = prepare_model_qat(model)
        fq_count = sum(
            1
            for m in qat_model.backbone_neck.modules()
            if "fakequant" in type(m).__name__.lower()
        )
        assert fq_count > 0

    def test_gradient_flow(self, model):
        qat_model = prepare_model_qat(model)
        qat_model.train()
        x = torch.randn(1, 3, 640, 640)
        out = qat_model(x)
        decoded, _raw = out
        pred_bboxes, pred_scores = decoded
        loss = pred_scores.sum()
        loss.backward()
        has_grad = any(
            p.grad is not None
            for p in qat_model.backbone_neck.parameters()
            if p.requires_grad
        )
        assert has_grad


# ── Conversion ─────────────────────────────────────────────────────


class TestConvert:
    def test_convert_ptq(self, model):
        ptq_model = prepare_model_ptq(model)
        ptq_model.eval()
        x = torch.randn(2, 3, 640, 640)
        with torch.no_grad():
            ptq_model(x)  # calibration pass

        quantized = convert_quantized(ptq_model)
        quantized.eval()
        with torch.no_grad():
            pred_bboxes, pred_scores = quantized(x)
        assert pred_bboxes.shape == (2, 8400, 4)
        assert pred_scores.shape == (2, 8400, 80)


# ── QConfig factories ─────────────────────────────────────────────


class TestQConfigMappings:
    @pytest.mark.parametrize("backend", ["x86", "qnnpack"])
    def test_default_qconfig(self, backend):
        mapping = default_qconfig_mapping(backend)
        assert mapping is not None

    @pytest.mark.parametrize("backend", ["x86", "qnnpack"])
    def test_qat_qconfig(self, backend):
        mapping = qat_qconfig_mapping(backend)
        assert mapping is not None
