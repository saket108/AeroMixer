"""Small runtime helpers for the active image-only detector stack."""


def unwrap_model(model):
    """Return the underlying module for DDP-wrapped models."""

    return model.module if hasattr(model, "module") else model


def get_backbone(model):
    """Return the model backbone if available."""

    base_model = unwrap_model(model)
    return getattr(base_model, "backbone", None)


def has_text_encoder(model):
    """Check whether the model backbone exposes a configurable text encoder."""

    backbone = get_backbone(model)
    return backbone is not None and hasattr(backbone, "text_encoder")


def configure_text_encoder(model, vocabulary):
    """Load a dataset vocabulary into the model text encoder when available."""

    if vocabulary is None or not has_text_encoder(model):
        return False

    backbone = get_backbone(model)
    backbone.text_encoder.set_vocabulary(vocabulary)
    return True
