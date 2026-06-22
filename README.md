# OptimalModelTree (OMT)

A mixed-integer optimization-based decision tree framework for classification and regression.

This package implements **Optimal Model Trees (OMT)**, where the tree structure and prediction rules are learned jointly via a mathematical optimization model (MILP) using Gurobi.

---

## 🚀 Features

- Optimal decision tree construction via mixed-integer programming
- Supports:
  - Classification (binary & multiclass)
  - Regression
- Parallel and oblique splits
- Regularized models (L1-style sparsity control)
- Warm-start support for faster optimization
- Scikit-learn compatible API (`fit`, `predict`)

---

## 📦 Installation

### From PyPI

```bash
pip install optimal-omt