from src.deepal.models.base import LinearSeq


def get_net(
    model_architecture: str,
):
    """Create a model from a configuration."""
    if model_architecture == "linear":
        return LinearSeq
    else:
        raise NotImplementedError
