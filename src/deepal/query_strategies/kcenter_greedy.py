import time

import numpy as np
import pandas as pd
from .strategy import Strategy
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.metrics import pairwise_distances


class KCenterGreedy(Strategy):
	def __init__(self, X, Y, idxs_lb, net, handler, args, emb="latent"):
		super(KCenterGreedy, self).__init__(X, Y, idxs_lb, net, handler, args)
		self.emb = emb

	def query(self, n):
		idxs_unlabeled = np.arange(self.n_pool)[~self.idxs_lb]
		idxs_lb = np.arange(self.n_pool)[self.idxs_lb]

		embedding = self.get_diversity_embeddings(emb_type=self.emb, x=self.X, y=self.Y)

		dist_mat = euclidean_distances(embedding[idxs_unlabeled], embedding[idxs_lb])
		min_dists = np.min(dist_mat, axis=-1)
		ind = min_dists.argmax()
		indsAll = [ind]
		features = [embedding[idxs_unlabeled[ind]]]
		while len(indsAll) < n:
			new_dist = pairwise_distances(embedding[idxs_unlabeled], [features[-1]]).ravel().astype(float)
			for i in range(len(embedding[idxs_unlabeled])):
				if min_dists[i] > new_dist[i]:
					min_dists[i] = new_dist[i]
			ind = min_dists.argmax()
			features.append(embedding[idxs_unlabeled[ind]])
			indsAll.append(ind)

		chosen = idxs_unlabeled[indsAll]
		end_time = time.time()
		sampling_df = pd.DataFrame([list(chosen)], index=["img_id"]).T
		self.save_stats(sampling_df)
		return chosen, end_time
