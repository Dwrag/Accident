import numpy as np
import torch
import torch.nn as nn
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.inspection import permutation_importance


class _MLP(nn.Module):
    def __init__(self, input_dim, hidden_dims, num_classes, dropout=0.25):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers += [
                nn.Linear(prev, h),
                nn.BatchNorm1d(h),
                nn.ReLU(),
                nn.Dropout(dropout),
            ]
            prev = h
        layers.append(nn.Linear(prev, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class TorchMLPClassifier(BaseEstimator, ClassifierMixin):
    """
    Minimal sklearn-style wrapper around a PyTorch MLP so it can sit
    inside train_model.py's `models[name] = ...` dict and be trained /
    evaluated / joblib-pickled exactly like the sklearn models.

    Inherits from BaseEstimator/ClassifierMixin (rather than just
    duck-typing fit/predict) purely so that sklearn utilities like
    permutation_importance — which on sklearn>=1.6 look up
    `__sklearn_tags__` to figure out if something is a classifier —
    work without special-casing this model.
    """

    def __init__(self, hidden_dims=(128, 64, 32), dropout=0.25, lr=1e-3,
                 weight_decay=1e-4, epochs=200, batch_size=64, patience=15,
                 val_split=0.15, random_state=42, verbose=True):
        self.hidden_dims = hidden_dims
        self.dropout = dropout
        self.lr = lr
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.batch_size = batch_size
        self.patience = patience
        self.val_split = val_split
        self.random_state = random_state
        self.verbose = verbose

        self.model_ = None
        self.classes_ = None
        self.feature_importances_ = None
        self.feature_names_ = None
        self.device_ = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ------------------------------------------------------------------
    def _to_tensor(self, X, dtype=torch.float32):
        if hasattr(X, "values"):
            X = X.values
        return torch.tensor(np.asarray(X), dtype=dtype)

    # ------------------------------------------------------------------
    def fit(self, X, y):
        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)

        if hasattr(X, "columns"):
            self.feature_names_ = list(X.columns)

        y = np.asarray(y)
        self.classes_ = np.unique(y)
        num_classes = len(self.classes_)
        input_dim = X.shape[1]

        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=self.val_split, random_state=self.random_state,
            stratify=y if num_classes > 1 else None,
        )

        Xt_train = self._to_tensor(X_train).to(self.device_)
        yt_train = torch.tensor(np.asarray(y_train), dtype=torch.long).to(self.device_)
        Xt_val = self._to_tensor(X_val).to(self.device_)
        yt_val = torch.tensor(np.asarray(y_val), dtype=torch.long).to(self.device_)

        # class weights, same intent as class_weight="balanced" in sklearn
        class_counts = np.bincount(y_train, minlength=num_classes).astype(np.float32)
        class_counts[class_counts == 0] = 1.0
        class_weights = (class_counts.sum() / (num_classes * class_counts))
        class_weights_t = torch.tensor(class_weights, dtype=torch.float32).to(self.device_)

        self.model_ = _MLP(input_dim, self.hidden_dims, num_classes, self.dropout).to(self.device_)
        optimizer = torch.optim.Adam(self.model_.parameters(), lr=self.lr,
                                      weight_decay=self.weight_decay)
        criterion = nn.CrossEntropyLoss(weight=class_weights_t)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=5
        )

        best_val_loss = float("inf")
        best_state = None
        epochs_no_improve = 0
        n = Xt_train.shape[0]

        for epoch in range(self.epochs):
            self.model_.train()
            perm = torch.randperm(n)
            epoch_loss = 0.0
            for start in range(0, n, self.batch_size):
                idx = perm[start:start + self.batch_size]
                xb, yb = Xt_train[idx], yt_train[idx]

                optimizer.zero_grad()
                out = self.model_(xb)
                loss = criterion(out, yb)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item() * xb.size(0)
            epoch_loss /= n

            self.model_.eval()
            with torch.no_grad():
                val_out = self.model_(Xt_val)
                val_loss = criterion(val_out, yt_val).item()
            scheduler.step(val_loss)

            if val_loss < best_val_loss - 1e-4:
                best_val_loss = val_loss
                best_state = {k: v.clone() for k, v in self.model_.state_dict().items()}
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1

            if self.verbose and (epoch % 20 == 0 or epoch == self.epochs - 1):
                print(f"    [Deep Learning (MLP)] epoch {epoch:3d} "
                      f"train_loss={epoch_loss:.4f} val_loss={val_loss:.4f}")

            if epochs_no_improve >= self.patience:
                if self.verbose:
                    print(f"    [Deep Learning (MLP)] early stopping at epoch {epoch}")
                break

        if best_state is not None:
            self.model_.load_state_dict(best_state)
        self.model_.eval()

        # Permutation importance so app.py's "why did the model predict
        # this" panel works the same way it does for the tree models.
        try:
            result = permutation_importance(
                self, X_val, y_val, n_repeats=5, random_state=self.random_state,
                scoring="accuracy",
            )
            importances = np.clip(result.importances_mean, 0, None)
            total = importances.sum()
            self.feature_importances_ = importances / total if total > 0 else importances
        except Exception:
            self.feature_importances_ = None

        return self

    # ------------------------------------------------------------------
    def predict_proba(self, X):
        self.model_.eval()
        with torch.no_grad():
            Xt = self._to_tensor(X).to(self.device_)
            logits = self.model_(Xt)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
        return probs

    def predict(self, X):
        probs = self.predict_proba(X)
        idx = probs.argmax(axis=1)
        return self.classes_[idx]

    def score(self, X, y):
        return accuracy_score(y, self.predict(X))

    # ------------------------------------------------------------------
    # Make the object joblib/pickle-friendly: torch tensors inside the
    # nn.Module pickle fine on their own, but we explicitly move the
    # model to CPU before saving so it reloads correctly on any machine
    # (e.g. training on GPU, serving on a CPU-only Streamlit host).
    def __getstate__(self):
        state = self.__dict__.copy()
        if self.model_ is not None:
            state["model_"] = self.model_.to("cpu")
            state["device_"] = torch.device("cpu")
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        if self.model_ is not None:
            self.device_ = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model_.to(self.device_)
