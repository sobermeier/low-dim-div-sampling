import numpy as np
from sklearn.decomposition import PCA
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader


class Strategy:
    def __init__(self, X, Y, idxs_lb, net, handler, args):
        self.sampling_info = []
        self.X = X
        self.Y = Y
        self.idxs_lb = idxs_lb
        self.net_args = net["net_args"]
        self.net = net["net"]
        self.handler = handler
        self.args = args
        self.n_pool = len(Y)
        self.current_round = 0
        self.total_rounds = 0
        use_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda" if use_cuda else "cpu")
        self.out_dir = None

    def query(self, n):
        pass

    def update(self, idxs_lb):
        self.idxs_lb = idxs_lb

    def save_stats(self, df):
        file_name = f"{self.current_round}_statistics.csv"
        if self.out_dir is not None:
            df.to_csv(self.out_dir / file_name)
        else:
            print("did not save, no out_dir specified.")

    def set_current_round(self, iteration):
        self.current_round = iteration

    def set_total_rounds(self, total_rounds):
        self.total_rounds = total_rounds

    def set_path(self, out_dir):
        """ create and set output directory for current seed and experiment run. """
        if out_dir.is_dir():
            print(f'Output path already exists! {out_dir}')
        out_dir.mkdir(exist_ok=True, parents=True)
        self.out_dir = out_dir

    def _train(self, loader_tr, optimizer):
        self.clf.train()
        train_accuracy = 0.
        train_loss = 0.
        for batch_idx, (x, y, idxs) in enumerate(loader_tr):
            x, y = x.to(self.device), y.to(self.device)
            optimizer.zero_grad()
            out, e1 = self.clf(x)
            loss = self.clf.loss_fc(out, y)
            train_accuracy += torch.sum((torch.max(out, 1)[1] == y).float()).data.item()
            loss.backward()
            train_loss += loss.item()
            optimizer.step()
        train_accuracy /= len(loader_tr.dataset.X)
        train_loss /= len(loader_tr.dataset.X)
        return train_accuracy, train_loss

    def train(self):
        self.clf = self.net(**self.net_args).to(self.device)

        optimizer = optim.Adam(
            self.clf.parameters(),
            lr=self.args["train_params"]['lr'],
            weight_decay=self.args["train_params"]['wd']
        )

        idxs_train = np.arange(self.n_pool)[self.idxs_lb]
        loader_tr = DataLoader(
            self.handler(self.X[idxs_train], self.Y[idxs_train], transform=self.args['ds_params']['transform']),
            shuffle=True,
            **self.args['ds_params']['loader_tr_args']
        )
        loss_hist, acc_hist = [], []
        tr_loss, tr_acc = 0, 0
        for epoch in range(self.args["train_params"]["epochs"]):
            tr_acc, tr_loss = self._train(loader_tr, optimizer)
            loss_hist.append(tr_loss)
            acc_hist.append(tr_acc)
            print(f'Epoch {epoch:5}: {tr_loss:2.7f} (acc: {tr_acc})')
        return tr_acc, tr_loss

    def predict(self, X, Y):
        loader_te = DataLoader(
            self.handler(X, Y, transform=self.args["ds_params"]['transform']),
            shuffle=False,
            **self.args["ds_params"]['loader_te_args']
        )
        self.clf.eval()
        P = torch.zeros(len(Y), dtype=Y.dtype)
        with torch.no_grad():
            for x, y, idxs in loader_te:
                x, y = x.to(self.device), y.to(self.device)
                out, e1 = self.clf(x)
                pred = torch.max(out, dim=-1)[1]
                P[idxs] = pred.cpu()
        return P

    def predict_prob(self, X, Y):
        loader_te = DataLoader(
            self.handler(X, Y, transform=self.args["ds_params"]['transform']),
            shuffle=False,
            **self.args["ds_params"]['loader_te_args']
        )

        self.clf.eval()
        probs = torch.zeros([len(Y), len(np.unique(Y))])
        with torch.no_grad():
            for x, y, idxs in loader_te:
                x, y = x.to(self.device), y.to(self.device)
                out, e1 = self.clf(x)
                prob = F.softmax(out, dim=1)
                probs[idxs] = prob.cpu()

        return probs

    def get_embedding(self, X, Y):
        loader_te = DataLoader(
            self.handler(X, Y, transform=self.args["ds_params"]['transform']),
            shuffle=False,
            **self.args["ds_params"]['loader_te_args']
        )

        self.clf.eval()
        embedding = torch.zeros([len(Y), self.clf.get_embedding_dim()])
        with torch.no_grad():
            for x, y, idxs in loader_te:
                x, y = x.to(self.device), y.to(self.device)
                out, e1 = self.clf(x)
                embedding[idxs] = e1.cpu()
        return embedding

    def get_diversity_embeddings(self, emb_type: str, x, y) -> np.ndarray:
        """
        get vectors on which diversity sampling is performed
        """
        if emb_type == "latent":
            print("latent")
            embedding = self.get_embedding(x, y)
            return embedding.numpy()
        elif emb_type == "pca":
            print("pca")
            embedding = self.get_embedding(x, y).numpy()
            pca = PCA(n_components=10)
            pca.fit(embedding)
            return pca.transform(embedding)
        elif emb_type == "output":
            print("output")
            embedding = self.predict_prob(x, y)
            return embedding.numpy()
        else:
            print("Embedding Type Not Found. Must be one of 'latent', 'pca', 'output'")
            raise NotImplementedError

