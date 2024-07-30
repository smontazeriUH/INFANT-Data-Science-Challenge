# INFANT-Data-Science-Challenge

This repository contains the following files:

1. **Classifiers:**
   - `./classifiers/model1.h5`
   - `./classifiers/model2.h5`
   - `./classifiers/model3.h5`

   These are the proposed CNN classifiers trained on the data-challenge dataset.

2. **Preprocessing Script:**
   - `./01_preprocess.py`

   A Python file containing scripts for preprocessing (filtering and resampling) of the signals.

3. **Testing Sample Preparation:**
   - `./04_read_test_files.py`

   A Python file with scripts for preparing testing samples.

4. **Classifier Testing:**
   - `./05_testCNN.py`

   A Python file with scripts for testing the CNN classifiers.

### Prerequisites

The following libraries and versions should be installed in the environment:

- Python        -> 3.8.13
- Numpy         -> 1.22.3
- TensorFlow    -> 2.4.1
- Keras         -> 2.3.1
- Matplotlib    -> 3.5.1
- MNE           -> 0.24.0
- Scikit-learn  -> 1.0.2
- SciPy         -> 1.7.3
- Pandas        -> 1.4.3
- HDF5          -> 1.10.6

### Installation

You can install the required libraries using pip:

```bash
pip install numpy==1.22.3 tensorflow==2.4.1 keras==2.3.1 matplotlib==3.5.1 mne==0.24.0 scikit-learn==1.0.2 scipy==1.7.3 pandas==1.4.3 hdf5==1.10.6

```

Usage
**Preprocess the data:**
```bash
python 01_preprocess.py```

**Prepare testing samples:**
```bash
python 04_read_test_files.py```

**Test the CNN classifiers:**
```bash
python 05_testCNN.py```
