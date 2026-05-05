# Probabilistic-Framework

Pendiente

## Installation

Pendiente

## Sample usage

```python
import numpy as np
from sklearn.naive_bayes import GaussianNB
from ucimlrepo import fetch_ucirepo

# Loads the dataset
iris  = fetch_ucirepo(id=53) 
X, y = iris.data.features, iris.data.targets.squeeze()

# Training and predict
model = GaussianNB().fit(X, y)
result = model.predict_proba(X)

# Calculates the probabilistic confusion matrix and the probabilistic accuracy
prob_conf_matrix = prob_confusion_matrix(y, result)
prob_acc = prob_accuracy(y, result)

print(np.round(prob_conf_matrix,3))
print(f"Acc*:{np.round(prob_acc,5)}\n")


# Calculates the certainty and uncertainty confusion matrix, their probabilistic accuracy and their lambda values
V, U = certainty_matrix(y, result)
V_acc, U_acc = certainty_accuracy(y, result)
lambda_V, lambda_U = certainty_weights(y, result)

print(np.round(V,3))
print(f"Acc_V*:{np.round(V_acc,5)}, lambda_V:{np.round(lambda_V,5)}\n")
print(np.round(U,3))
print(f"Acc_U*:{np.round(U_acc,5)}, lambda_U:{np.round(lambda_U,5)}")
```

## Result sample

### Probabilistic confusion matrix
| Iris-setosa  |  Iris-versicolor | Iris-virginica |
|:------------:|:----------------:|:--------------:|
|      50      |         0        |        0       |
|      0       |       46.06      |       3.94     |
|      0       |        3.93      |      46.07     |

Acc*:0.94754

### Certainty matrix (V)

| Iris-setosa  |  Iris-versicolor | Iris-virginica |
|:------------:|:----------------:|:--------------:|
|      50      |         0        |         0      |
|       0      |       45.374     |       2.314    |
|       0      |        2.644     |      45.715    |

Acc_V*:0.96605, lambda_V:0.97365

### Uncertainty matrix (U)

| Iris-setosa  |  Iris-versicolor | Iris-virginica |
|:------------:|:----------------:|:--------------:|
|      0       |          0       |         0      |
|      0       |        0.686     |       1.626    |
|      0       |        1.285     |       0.356    |

Acc_U*:0.26353, lambda_U:0.02635

## Citation

Pendiente
