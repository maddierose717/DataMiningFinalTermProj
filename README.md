# CS 634 Data Mining - Final Term Project

**Student:** Madison Rose Lucas  
**UCID:** MRL58  
**Email:** MRL58@njit.edu

## Project Description

This project implements two classification algorithms **from scratch** for Option 1: Supervised Data Mining (Classification).

### Algorithms Implemented

1. **Algorithm 1:** LIBSVM with RBF Kernel (Support Vector Machines)
   - Custom implementation using Sequential Minimal Optimization (SMO)
   - RBF (Gaussian) kernel
   - One-vs-Rest strategy for multiclass classification

2. **Algorithm 2:** Random Forest
   - Custom Decision Tree implementation using Gini impurity
   - Bootstrap sampling for tree diversity
   - Random feature selection
   - Majority voting for predictions

## Implementation Details

Both algorithms are implemented **entirely from scratch** following professor's requirements:
- ✅ Core learning logic implemented manually
- ✅ Uses only NumPy for basic array operations
- ✅ sklearn only used for dataset loading and StandardScaler (preprocessing)
- ✅ No sklearn models, training utilities, or metrics helpers

## Datasets

The algorithms are evaluated on three datasets:
1. **Iris Dataset** - 150 samples, 4 features, 3 classes
2. **Wine Dataset** - 178 samples, 13 features, 3 classes
3. **Breast Cancer Dataset** - 569 samples, 30 features, 2 classes

## Evaluation Method

- **10-fold cross-validation** (implemented from scratch)
- Reports mean accuracy ± standard deviation for each algorithm on each dataset

## Requirements
```
numpy
scikit-learn (only for datasets and StandardScaler)
```

## Usage
```bash
python final_project.py
```

## Results

The script outputs:
- Accuracy scores for each algorithm on each dataset
- Summary table comparing SVM vs Random Forest performance
