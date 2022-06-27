import json
import os
import pathlib
import statistics
import time

import click
import numpy as np
import pandas as pd

from src.deepal.dataset import get_handler, get_dataset, get_transform
from src.deepal.models.utils import get_net
from src.deepal.database import MLFlowLogger
from src.deepal.query_strategies import get_strategy
from src.deepal.utils import str_list, set_seed
from src.deepal.settings import RUN_CONFIG


@click.command()
@click.option('--seeds', default=[1, 2, 3, 4, 5], type=str_list)  # 5 seeds
@click.option('--tracking_uri', required=True, help='MLflow tracking uri.')
def main(seeds, tracking_uri):
    ds = "MNIST"
    acq_strategies = ['KCenterGreedy', 'KMeansSampling', 'KMeansPP']
    for acq_strategy in acq_strategies:
        # load config
        config_path = os.path.join(RUN_CONFIG, f'{ds.lower()}.json')
        with open(config_path, 'r') as config_file:
            config = json.load(config_file)
        n_initials, n_queries, budget = config["al_params"]["n_initials"], config["al_params"]["n_queries"], config["al_params"]["budget"]

        # load dataset
        X_tr, Y_tr, X_te, Y_te, class_counts = get_dataset(ds)
        if "openml" in ds:
            config["model_params"]["output_dim"] = int(max(Y_tr) + 1)
            config["model_params"]["input_dims"] = np.shape(X_tr)[-1]
        config['ds_params']['transform'] = get_transform(ds)
        n_pool = len(Y_tr)

        # model settings
        model_architecture = config["model_architecture"]

        # evaluation loop
        for _initials in n_initials:
            _initials = int(_initials)
            for _queries in n_queries:
                _queries = int(_queries)
                n_rounds = int((budget - _initials) / _queries)
                if (budget - _initials) % _queries != 0:
                    print(f"--> budget - initials) % queries = {(budget - _initials) % _queries}")
                    n_rounds += 1
                m_params_sampling = config['model_params']
                settings = {
                    "strategy": acq_strategy,
                    "dataset": ds,
                    "n_queries": _queries,
                    "n_rounds": n_rounds,
                    "n_initials": _initials,
                    "model_architecture": model_architecture,
                    "model_params": m_params_sampling,
                    "seed": seeds,  # save all seeds to mlflow for reproduction
                    "epochs": config["train_params"]["epochs"],
                    "batch_size_tr": config["ds_params"]["loader_tr_args"]["batch_size"],
                    "batch_size_te": config["ds_params"]["loader_te_args"]["batch_size"],
                    "learning_rate": config["train_params"]["lr"],
                    "weight_decay": config["train_params"]["wd"]
                }
                #  One mlflow run over all seeds
                database = MLFlowLogger(experiment_name=f'low-dim-div-sampling', tracking_uri=tracking_uri)
                run_id, output_path = database.init_experiment(hyper_parameters=settings)
                output_path = pathlib.Path(output_path)
                print(f"Started experiment: \n  - run_id: {run_id} \n  - queries: {_queries} \n  - rounds: {n_rounds} \n  - initials: {n_initials} \n  - architecture: {model_architecture}")
                accuracies_per_seed = {iteration: [] for iteration in range(0, n_rounds + 1)}
                times_per_seed = {iteration: [] for iteration in range(0, n_rounds)}
                for seed in seeds:
                    set_seed(seed)
                    remaining_budget = budget - _initials
                    print(f"-> Eval with seed={seed}")
                    # generate initial labeled pool; dependent on seed
                    idxs_lb = np.zeros(n_pool, dtype=bool)
                    idxs_tmp = np.arange(n_pool)
                    np.random.shuffle(idxs_tmp)
                    idxs_lb[idxs_tmp[:_initials]] = True
                    # load network and data handler
                    net_sampling = get_net(model_architecture=model_architecture)
                    handler = get_handler(ds)

                    # strategy requires data, currently labeled indices, the model, the data handler, and custom args
                    strategy, s_params = get_strategy(
                        acq_strategy,
                        X_tr, Y_tr, idxs_lb,
                        {"net": net_sampling, "net_args": m_params_sampling},
                        handler,
                        config
                    )
                    database.log_params({"strategy_parameters": s_params})  # save strategy params to mlflow
                    strategy.set_path(output_path / str(seed))
                    strategy.set_total_rounds(n_rounds)
                    # save initially labeled instances
                    strategy.save_stats(pd.DataFrame([list(idxs_tmp[:_initials])], index=["img_id"]).T)

                    # round 0 accuracy
                    tr_acc, tr_loss = strategy.train()  # train sampling model for acquisition
                    P = strategy.predict(X_te, Y_te)
                    acc = np.zeros(n_rounds + 1)
                    times = np.zeros(n_rounds)
                    acc[0] = 1.0 * (Y_te == P).sum().item() / len(Y_te)
                    result = {f"acc_{seed}": acc[0], f"tr_loss_{seed}": tr_loss, f"tr_acc_{seed}": tr_acc}
                    accuracies_per_seed[0].append((acc[0], np.sum(idxs_lb)))
                    database.log_results(result=result, step=np.sum(idxs_lb))
                    print('Round 0\ntesting accuracy {}'.format(acc[0]))

                    rd = 1
                    query_size = _queries
                    while remaining_budget > 0:
                        if remaining_budget < _queries:
                            query_size = remaining_budget

                        print(f'Round {rd} - Budget {remaining_budget}')
                        # query
                        strategy.set_current_round(rd)
                        start_time = time.time()
                        q_idxs, end_time = strategy.query(query_size)
                        seconds_total = end_time - start_time
                        times[rd-1] = seconds_total
                        times_per_seed[rd-1].append((times[rd-1], np.sum(idxs_lb)))

                        idxs_lb[q_idxs] = True

                        # update
                        strategy.update(idxs_lb)
                        tr_acc, tr_loss = strategy.train()

                        P = strategy.predict(X_te, Y_te)
                        acc[rd] = 1.0 * (Y_te == P).sum().item() / len(Y_te)
                        accuracies_per_seed[rd].append((acc[rd], np.sum(idxs_lb)))
                        result = {
                            f"acc_{seed}": acc[rd],
                            f"tr_loss_{seed}": tr_loss,
                            f"tr_acc_{seed}": tr_acc,
                            f"sec_{seed}": seconds_total
                        }
                        database.log_results(result=result, step=np.sum(idxs_lb))
                        print('testing accuracy {}'.format(acc[rd]))

                        remaining_budget -= _queries
                        rd += 1

                print('--- accuracies per round & seed: --- ')
                print(f"  {accuracies_per_seed}")
                # save average over seeds to mlflow
                for key, val in accuracies_per_seed.items():
                    con_accs = [tup[0] for tup in val]
                    con_samples = [tup[1] for tup in val]
                    database.log_results(result={"avg_acc": statistics.mean(con_accs)}, step=con_samples[0])
                    database.log_results(result={"var_acc": statistics.variance(con_accs)}, step=con_samples[0])
                    database.log_results(result={"std_acc": statistics.stdev(con_accs)}, step=con_samples[0])

                for key, val in times_per_seed.items():
                    times = [tup[0] for tup in val]
                    samples = [tup[1] for tup in val]
                    database.log_results(result={"avg_query_time": statistics.mean(times)}, step=samples[0])
                    database.log_results(result={"var_query_time": statistics.variance(times)}, step=samples[0])
                    database.log_results(result={"std_query_time": statistics.stdev(times)}, step=samples[0])

                # close experiment after all seeds have been visited
                database.finalise_experiment()


if __name__ == '__main__':
    main()
