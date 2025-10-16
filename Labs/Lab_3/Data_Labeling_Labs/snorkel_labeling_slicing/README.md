# Snorkel Labeling and Data Slicing Environment Setup

This guide explains how to set up a stable environment for running Snorkel labeling and slicing tutorials locally.  
It includes environment creation, package installation, troubleshooting, and verification steps to ensure all core Snorkel modules such as `LabelModel` and `PandasLFApplier` import correctly.

## 1. Overview

Snorkel is a weak supervision and data-centric AI framework developed to build and manage labeled datasets programmatically.  
Older versions (up to 0.9.x) include a rich API for labeling functions, label models, and data slicing modules.  
Later versions (0.10 and above) restructured the API, which can break compatibility with official tutorials.  

To reproduce classic Snorkel tutorials and maintain full compatibility, this setup uses:

- Python 3.8
- Snorkel 0.9.9
- PyTorch 1.1.0
- Scikit-learn 1.0.2
- Pandas 1.1.5
- Numpy 1.19.5

## 2. Prerequisites

Before beginning, ensure that:
- You have **Anaconda** or **Miniconda** installed.
- You are running this on macOS or Linux (recommended).  
- If you use `pyenv`, confirm that your Python version is not shadowing the conda environment.

To check which Python interpreter is active:
```bash
pyenv which python
```

## 3. Environment Creation

Run the following commands to create a clean and isolated conda environment for Snorkel.

```bash
conda remove --name snorkel --all -y
conda create -y -n snorkel python=3.8
conda activate snorkel
```

## 4. Package Installation

Install the required dependencies inside the newly created environment:

```bash
pip install snorkel==0.9.9 torch==1.1.0 scikit-learn==1.0.2 pandas==1.1.5 numpy==1.19.5
```

This setup replicates the environment used in the official Snorkel 0.9 tutorials.

## 5. Verification

Once installation is complete, verify that all core Snorkel modules load correctly:

```bash
python -c "from snorkel.labeling import LabelModel, PandasLFApplier; print('Snorkel successfully installed.')"
```

If this command prints `Snorkel successfully installed.`, the environment is working properly.

To confirm the installed version:
```bash
python -m pip show snorkel
```
Expected output:
```
Version: 0.9.9
```

## 6. Common Errors and Fixes

### ImportError: cannot import name 'LabelModel' from 'snorkel.labeling'
**Cause:** You are running Snorkel 0.10 or later.  
**Fix:** Downgrade to version 0.9.9 using pip:
```bash
pip install --force-reinstall snorkel==0.9.9
```

### Conflicting global installs with pyenv
If you see multiple Snorkel paths in your system:
```bash
pip list | grep snorkel
```
Remove the global installation:
```bash
pip uninstall snorkel
```
Then reinstall inside your conda environment only.

## 7. Testing the Installation

You can verify Snorkel end-to-end by running this simple test script:

```python
import pandas as pd
from snorkel.labeling import LabelModel, PandasLFApplier, labeling_function

ABSTAIN = -1
SPAM = 1
HAM = 0

@labeling_function()
def short_comment(x):
    return SPAM if len(x.text.split()) < 5 else ABSTAIN

data = pd.DataFrame({"text": ["Free money now", "Nice video", "Click here", "Awesome content", "Shakira rocks!"]})
lfs = [short_comment]

applier = PandasLFApplier(lfs)
L_train = applier.apply(df=data)

label_model = LabelModel(cardinality=2, verbose=True)
label_model.fit(L_train=L_train, n_epochs=200, seed=42)
preds = label_model.predict(L_train)
print(preds)
```

Expected output: a NumPy array of predicted labels without import or runtime errors.

## 8. Notes for Google Colab Users

Conda environments are not natively supported in Google Colab.  
If you want to run Snorkel inside Colab, install the same versions directly via pip:

```bash
!pip install snorkel==0.9.9 torch==1.1.0 scikit-learn==1.0.2 pandas==1.1.5 numpy==1.19.5
```

Then restart the Colab runtime and verify the import:
```python
from snorkel.labeling import LabelModel
```

## 9. Troubleshooting Commands

To check the currently active Python environment:
```bash
which python
```

To list installed packages:
```bash
pip list
```

To clear cached site-packages if an import error persists:
```bash
pip uninstall snorkel
pip cache purge
pip install snorkel==0.9.9
```

## 10. Summary

After following this guide, you will have a working and isolated environment capable of running all Snorkel labeling and slicing tutorials without dependency or import issues.  

**Environment summary:**
- Python 3.8  
- Snorkel 0.9.9  
- PyTorch 1.1.0  
- Scikit-learn 1.0.2  
- Pandas 1.1.5  
- Numpy 1.19.5  

This configuration is stable and verified against all official Snorkel 0.9 examples.
