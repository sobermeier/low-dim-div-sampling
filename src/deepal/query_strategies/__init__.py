from .kmeans_sampling import KMeansSampling
from .kcenter_greedy import KCenterGreedy
from .kmeanspp import KMeansPP


def get_strategy(name, x, y, labelled, model, d_handler, d_args):
    strategy_params = {}
    klass = globals()[name]
    if name in ["KMeansSampling", "KMeansPP", "KCenterGreedy"]:
        strategy_params["emb"] = "pca"  # latent | output | pca
        instance = klass(x, y, labelled, model, d_handler, d_args, **strategy_params)
        return instance, strategy_params
    else:
        print("Strategy not found")
        raise NotImplementedError()

