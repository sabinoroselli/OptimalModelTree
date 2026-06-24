OptimalModelTree (OMT)

A mixed-integer optimization-based decision tree framework for classification and regression.

OMT implements Optimal Model Trees, where both the tree structure and the prediction models at the leaves are learned jointly by solving a Mixed-Integer Linear Programming (MILP) problem.

⸻

🚀 Features

* Globally optimized decision tree construction via MILP
* Supports:
    * Binary classification
    * Multiclass classification
    * Regression
* Parallel and oblique decision boundaries
* Sparse and regularized leaf models
* Warm-start support for faster optimization
* Scikit-learn-style API (fit, predict, score)

⸻

📦 Installation

Install the latest release from PyPI:

pip install optimal-omt

⸻

⚙️ Solver Backend

OMT is built on top of Google OR-Tools, which provides interfaces to several Mixed-Integer Linear Programming (MILP) solvers.

By default, OMT uses SCIP, an open-source solver that is distributed with OR-Tools and requires no additional installation.

Commercial solvers such as Gurobi are also supported and can substantially improve performance on larger optimization problems. However, Gurobi is not included with OMT and must be installed and licensed separately.

To use Gurobi:

1. Obtain a valid Gurobi license.
2. Install the Python interface:

pip install gurobipy

3. Configure OMT to use Gurobi as the optimization backend.

If Gurobi is not installed, OMT automatically falls back to SCIP.

Note: OMT solves optimization problems exactly and is therefore intended primarily for small-to-medium datasets.

⸻

🧪 Minimal Working Example

The following example trains an Optimal Model Tree on the Wine dataset from scikit-learn.
```

from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from omt import OptimalModelTreeClassifier
X, y = load_wine(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)
clf = OptimalModelTreeClassifier(Splits=2,solver='SCIP')
clf.fit(X_train, y_train)
print("Test accuracy:", clf.score(X_test, y_test))
```

Example output:
```

Test accuracy: 0.944
```

⸻

📚 Citation

If you use OMT in your research, please cite:

```bibtex
@article{roselli2025experiments,
  title={Experiments with Optimal Model Trees},
  author={Roselli, Sabino Francesco and Frank, Eibe},
  journal={arXiv preprint arXiv:2503.12902},
  year={2025}
  url = {https://github.com/sabinoroselli/OptimalModelTree}
}
```


