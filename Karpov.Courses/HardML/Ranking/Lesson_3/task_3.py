import math
import numpy as np
import torch
from torch import FloatTensor, Tensor
from torch import nn
from catboost.datasets import msrank_10k
from sklearn.preprocessing import StandardScaler
from typing import List
from time import time



class ListNet(torch.nn.Module):
    def __init__(self, num_input_features: int, hidden_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        # укажите архитектуру простой модели здесь
        self.model = nn.Sequential(
            nn.Linear(num_input_features, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, 1)
        )

    def forward(self, input_1: torch.Tensor) -> torch.Tensor:
        logits = self.model(input_1)
        return logits


class Solution:
    def __init__(self, n_epochs: int = 5, listnet_hidden_dim: int = 12,
                 lr: float = 0.001, ndcg_top_k: int = 10):
        self._prepare_data()
        self.num_input_features = self.X_train.shape[1]
        self.ndcg_top_k = ndcg_top_k
        self.n_epochs = n_epochs

        self.model = self._create_model(
            self.num_input_features, listnet_hidden_dim)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)


    def _get_data(self) -> List[np.ndarray]:
        train_df, test_df = msrank_10k()

        # 0,1 - target/queryGroup columns
        X_train = train_df.drop([0, 1], axis=1).values
        y_train = train_df[0].values
        query_ids_train = train_df[1].values.astype(int)

        X_test = test_df.drop([0, 1], axis=1).values
        y_test = test_df[0].values
        query_ids_test = test_df[1].values.astype(int)

        return [X_train, y_train, query_ids_train, X_test, y_test, query_ids_test]


    def _prepare_data(self) -> None:
        (X_train, y_train, self.query_ids_train,
            X_test, y_test, self.query_ids_test) = self._get_data()
        X_train = self._scale_features_in_query_groups(X_train, self.query_ids_train)
        X_test = self._scale_features_in_query_groups(X_test, self.query_ids_train)
        # Tensors
        self.X_train = FloatTensor(X_train)
        self.X_test = FloatTensor(X_test)
        self.ys_train = FloatTensor(y_train)
        self.ys_test = FloatTensor(y_test)


    def _scale_features_in_query_groups(self, inp_feat_array: np.ndarray,
                                        inp_query_ids: np.ndarray) -> np.ndarray:
        for query_id in np.unique(inp_query_ids):
            mask = query_id == inp_query_ids
            inp_feat_array[mask] = StandardScaler().fit_transform(inp_feat_array[mask])

        return inp_feat_array


    def _create_model(self, listnet_num_input_features: int, listnet_hidden_dim: int) -> torch.nn.Module:
        torch.manual_seed(0)
        net = ListNet(num_input_features=listnet_num_input_features,
                      hidden_dim=listnet_hidden_dim)
        return net


    def fit(self) -> List[float]:
        scores = []
        for epoch in range(self.n_epochs):
            self._train_one_epoch()
            eval_test_score = self._eval_test_set()
            scores.append(eval_test_score)
            print(f"epoch: {epoch + 1}.\t"
                  f"Avg nDCG: {eval_test_score:.4f}")
        return scores


    def _calc_loss(self, batch_ys: torch.FloatTensor,
                   batch_pred: torch.FloatTensor) -> torch.FloatTensor:
        P_y_i = torch.softmax(batch_ys, dim=0)
        P_z_i = torch.softmax(batch_pred, dim=0)
        return -torch.sum(P_y_i * torch.log(P_z_i / P_y_i))


    def _train_one_epoch(self) -> None:
        self.model.train()
        for query_id in np.unique(self.query_ids_train):
            mask = query_id == self.query_ids_train
            batch_X, batch_ys = self.X_train[mask], self.ys_train[mask]
            
            self.optimizer.zero_grad()

            batch_pred = self.model(batch_X).reshape(batch_ys.shape)
            batch_loss = self._calc_loss(batch_ys, batch_pred)

            batch_loss.backward()
            self.optimizer.step()


    def _eval_test_set(self) -> float:
        with torch.no_grad():
            self.model.eval()
            ndcgs = []
            for query_id in np.unique(self.query_ids_test):
                mask = query_id == self.query_ids_test
                pred = self.model(self.X_test[mask])
                ndcg_value = self._ndcg_k(self.ys_test[mask], pred, self.ndcg_top_k)
                
                if type(ndcg_value) == np.nan:
                    ndcgs.append(0)
                else:
                    ndcgs.append(ndcg_value)

            return np.mean(ndcgs)


    def _ndcg_k(self, ys_true: Tensor, ys_pred: Tensor, ndcg_top_k: int) -> float:

        def dcg(ys_true: Tensor, ys_pred: Tensor, ndcg_top_k: int=ndcg_top_k)-> float:
            # Sort ys
            ys_pred_sorted, indices = torch.sort(ys_pred, dim=0, descending=True)
            indices = indices[:ndcg_top_k]
            ys_true_sorted = ys_true[indices]
            # Metric value
            dcg_value = sum([(2 ** y_true - 1) / math.log2(i+1) for i, y_true in enumerate(ys_true_sorted, 1)])
            return float(dcg_value)

        empirical_dcg, ideal_dcg = dcg(ys_true, ys_pred), dcg(ys_true, ys_true)
        return empirical_dcg / ideal_dcg


if __name__ == '__main__':

    scores = Solution().fit()
    print(f"Metric values per epoch: {list(np.around(np.array(scores), 4))}")