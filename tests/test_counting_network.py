import torch
import torch.nn.functional as F
from torch import nn

from models_counting_network import CountingNetwork
from models_mae_pretraining import CounTRMaskedAutoencoder
from models_mae_vit import CounTRMAEViTBackbone


class TinyDINO(nn.Module):
    """Small stand-in that exposes DINOv2's intermediate-layer API."""

    def __init__(self, channels: int = 32) -> None:
        super().__init__()
        self.stem = nn.Conv2d(3, channels, kernel_size=1)
        self.blocks = nn.ModuleList(
            [nn.Conv2d(channels, channels, kernel_size=3, padding=1) for _ in range(12)]
        )
        self.last_requested_layers = None

    def get_intermediate_layers(
        self,
        images,
        n=4,
        reshape=True,
        return_class_token=False,
        norm=True,
    ):
        del reshape, return_class_token, norm
        self.last_requested_layers = n
        features = F.avg_pool2d(images, kernel_size=14, stride=14)
        features = self.stem(features)
        outputs = []
        for block in self.blocks:
            features = F.gelu(block(features))
            outputs.append(features)
        if isinstance(n, int):
            return outputs[-n:]
        return [outputs[index] for index in n]


def make_model(
    architecture_version: str = "v2", *, v4_query_mode: str = "repetition"
) -> CountingNetwork:
    return CountingNetwork(
        backbone=TinyDINO(),
        backbone_dim=32,
        decoder_dim=32,
        decoder_channels=(24, 16),
        query_dim=16,
        num_feature_levels=4,
        freeze_backbone=False,
        enable_text_conditioning=False,
        text_dropout_p=0.0,
        architecture_version=architecture_version,
        v4_query_mode=v4_query_mode,
    )


def test_forward_is_device_agnostic_nonnegative_and_resolution_preserving():
    model = make_model().eval()
    images = torch.rand(2, 3, 111, 137)
    query = torch.randn(2, 16)

    with torch.no_grad():
        result = model(images, query_features=query, return_aux=True)

    assert result["density"].shape == (2, 111, 137)
    assert result["proposal_density"].shape == (2, 111, 137)
    assert result["verification"].shape == (2, 111, 137)
    assert result["base_density"].shape == (2, 111, 137)
    assert result["count_correction"].shape == (2,)
    assert torch.all(result["density"] >= 0)
    assert torch.all((result["verification"] >= 0) & (result["verification"] <= 1))
    assert torch.all((result["verification_factor"] >= 0.5) & (result["verification_factor"] <= 1.5))
    assert torch.allclose(
        result["level_weights"].sum(dim=1),
        torch.ones_like(result["level_weights"][:, 0]),
    )
    assert torch.allclose(result["count_correction"], torch.ones(2))


def test_query_conditioning_changes_the_spatial_prediction():
    torch.manual_seed(7)
    model = make_model().eval()
    images = torch.rand(1, 3, 96, 128)
    first_query = torch.randn(1, 16)
    second_query = -first_query

    with torch.no_grad():
        first = model(images, query_features=first_query)
        second = model(images, query_features=second_query)

    assert not torch.allclose(first, second)


def test_reference_less_mode_uses_an_image_derived_query():
    model = make_model().eval()
    images = torch.rand(1, 3, 95, 101)

    with torch.no_grad():
        density = model(images)

    assert density.shape == (1, 95, 101)
    assert torch.isfinite(density).all()


def test_v1_path_remains_checkpoint_compatible():
    model = make_model("v1").eval()
    restored = make_model("v1").eval()
    restored.load_state_dict(model.state_dict(), strict=True)
    image = torch.rand(1, 3, 81, 93)

    with torch.no_grad():
        result = restored(image, return_aux=True)

    assert result["density"].shape == (1, 81, 93)
    assert result["level_weights"].shape == (4,)


def test_partial_backbone_fine_tuning_only_unfreezes_requested_blocks():
    backbone = TinyDINO()
    model = CountingNetwork(
        backbone=backbone,
        backbone_dim=32,
        decoder_dim=32,
        query_dim=16,
        num_feature_levels=4,
        freeze_backbone=True,
        trainable_backbone_blocks=2,
        enable_text_conditioning=False,
    )

    assert all(not parameter.requires_grad for parameter in backbone.stem.parameters())
    assert all(
        not parameter.requires_grad
        for block in backbone.blocks[:2]
        for parameter in block.parameters()
    )
    assert all(
        parameter.requires_grad
        for block in backbone.blocks[-2:]
        for parameter in block.parameters()
    )


def test_v3_is_an_exact_zero_initialized_refinement_of_v1():
    torch.manual_seed(11)
    baseline = make_model("v1").eval()
    refined = make_model("v3").eval()
    compatible = {
        key: value
        for key, value in baseline.state_dict().items()
        if key in refined.state_dict() and refined.state_dict()[key].shape == value.shape
    }
    refined.load_state_dict(compatible, strict=False)
    image = torch.rand(1, 3, 97, 109)
    query = torch.randn(1, 16)

    with torch.no_grad():
        baseline_density = baseline(image, query_features=query)
        refined_density = refined(image, query_features=query)

    assert torch.allclose(baseline_density, refined_density, atol=1e-6, rtol=1e-6)


def test_explicit_dino_layer_indices_are_forwarded_to_the_backbone():
    backbone = TinyDINO()
    model = CountingNetwork(
        backbone=backbone,
        backbone_dim=32,
        decoder_dim=32,
        decoder_channels=(24, 16),
        query_dim=16,
        num_feature_levels=4,
        freeze_backbone=False,
        enable_text_conditioning=False,
        text_dropout_p=0.0,
        architecture_version="v3",
        dino_layer_indices=(2, 5, 8, 11),
    ).eval()

    with torch.no_grad():
        model(torch.rand(1, 3, 83, 97))

    assert backbone.last_requested_layers == (2, 5, 8, 11)


def test_v4_dpt_repetition_path_is_nonnegative_and_resolution_preserving():
    model = make_model("v4").eval()
    images = torch.rand(2, 3, 111, 137)

    with torch.no_grad():
        result = model(images, return_aux=True)

    assert result["density"].shape == (2, 111, 137)
    assert result["similarity"].shape == (2, 111, 137)
    assert result["repetition"].shape == (2, 111, 137)
    assert result["level_weights"].shape == (4,)
    assert torch.all(result["density"] >= 0)
    assert torch.isfinite(result["density"]).all()
    assert torch.isfinite(result["similarity"]).all()
    assert torch.allclose(result["count_correction"], torch.ones(2))


def test_image_only_architectures_reject_external_queries():
    image = torch.rand(1, 3, 83, 97)
    for architecture in ("v4", "v5", "v6"):
        model = make_model(architecture).eval()
        with torch.no_grad():
            try:
                model(image, query_features=torch.randn(1, 16))
            except ValueError as error:
                assert "image-only" in str(error)
            else:
                raise AssertionError(
                    f"{architecture} accepted an external query"
                )


def test_all_v4_query_ablation_modes_run():
    image = torch.rand(1, 3, 83, 97)
    for query_mode in ("image_film", "none", "repetition"):
        model = make_model("v4", v4_query_mode=query_mode).eval()
        with torch.no_grad():
            result = model(image, return_aux=True)
        assert result["density"].shape == (1, 83, 97)
        assert torch.isfinite(result["density"]).all()


def test_v5_featup_jbu_path_runs_and_preserves_resolution():
    model = make_model("v5").eval()
    image = torch.rand(1, 3, 83, 97)

    with torch.no_grad():
        result = model(image, return_aux=True)

    assert result["density"].shape == (1, 83, 97)
    assert torch.isfinite(result["density"]).all()
    assert hasattr(model, "featup_adapter")


def test_countr_mae_vit_exposes_requested_spatial_blocks_without_cls_token():
    backbone = CounTRMAEViTBackbone(
        img_size=64,
        patch_size=16,
        embed_dim=32,
        depth=4,
        num_heads=4,
        pretrained=False,
    ).eval()

    with torch.no_grad():
        features = backbone.get_intermediate_layers(
            torch.rand(2, 3, 64, 64),
            n=(0, 1, 2, 3),
            reshape=True,
            norm=True,
        )

    assert len(features) == 4
    assert all(feature.shape == (2, 32, 4, 4) for feature in features)
    assert not hasattr(backbone, "cls_token")


def test_countr_mae_pretraining_reconstructs_native_patch_grid():
    model = CounTRMaskedAutoencoder(
        img_size=64,
        patch_size=16,
        embed_dim=32,
        depth=2,
        num_heads=4,
        decoder_embed_dim=16,
        decoder_depth=1,
        decoder_num_heads=4,
    ).eval()
    images = torch.rand(2, 3, 64, 64)

    with torch.no_grad():
        loss, prediction, mask = model(images, mask_ratio=0.5)
        reconstruction = model.unpatchify(prediction)

    assert prediction.shape == (2, 16, 16 * 16 * 3)
    assert mask.shape == (2, 16)
    assert torch.all(mask.sum(dim=1) == 8)
    assert reconstruction.shape == images.shape
    assert torch.isfinite(loss)
    assert not hasattr(model, "cls_token")


def test_v6_countr_mae_vit_head_path_matches_v5_spatial_contract():
    model = make_model("v6").eval()
    image = torch.rand(1, 3, 83, 97)

    with torch.no_grad():
        result = model(image, return_aux=True)

    assert model.patch_size == 16
    assert result["density"].shape == (1, 83, 97)
    assert result["repetition"].shape == (1, 83, 97)
    assert torch.isfinite(result["density"]).all()
    assert hasattr(model, "featup_adapter")
