# Setting Up Anaconda Environment

## Option 1: Create New Conda Environment from environment.yml

1. **Create the environment:**
   ```bash
   conda env create -f environment.yml
   ```

2. **Activate the environment:**
   ```bash
   conda activate toit_revamp
   ```

3. **Verify installation:**
   ```bash
   conda list
   ```

## Option 2: Create Environment Manually

1. **Create a new conda environment:**
   ```bash
   conda create -n toit_revamp python=3.9
   ```

2. **Activate the environment:**
   ```bash
   conda activate toit_revamp
   ```

3. **Install packages:**
   ```bash
   conda install pandas numpy scikit-learn matplotlib jupyter ipykernel
   ```

   Or using pip:
   ```bash
   pip install -r requirements.txt
   ```

## Linking Environment to Jupyter Notebook

1. **Activate your environment:**
   ```bash
   conda activate toit_revamp
   ```

2. **Install ipykernel in the environment:**
   ```bash
   conda install ipykernel
   # or
   pip install ipykernel
   ```

3. **Register the kernel with Jupyter:**
   ```bash
   python -m ipykernel install --user --name toit_revamp --display-name "Python (toit_revamp)"
   ```

4. **In Jupyter Notebook:**
   - Go to Kernel → Change Kernel → Select "Python (toit_revamp)"

## Linking Environment to VS Code / Cursor

### VS Code / Cursor:

1. **Activate the environment:**
   ```bash
   conda activate toit_revamp
   ```

2. **In VS Code/Cursor:**
   - Press `Cmd+Shift+P` (Mac) or `Ctrl+Shift+P` (Windows/Linux)
   - Type "Python: Select Interpreter"
   - Choose the interpreter from your conda environment:
     - Should show: `~/anaconda3/envs/toit_revamp/bin/python` (or similar path)
     - Or: `Python 3.9.x ('toit_revamp': conda) ~/anaconda3/envs/toit_revamp/bin/python`

3. **For Jupyter Notebooks in VS Code/Cursor:**
   - Open your `.ipynb` file
   - Click on the kernel name in the top right
   - Select "Python (toit_revamp)" or the conda environment interpreter

## Using Existing Anaconda Environment

If you already have a conda environment you want to use:

1. **Activate it:**
   ```bash
   conda activate your_existing_env_name
   ```

2. **Install required packages:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Link to Jupyter (if needed):**
   ```bash
   python -m ipykernel install --user --name your_existing_env_name --display-name "Python (your_existing_env_name)"
   ```

## Verify Environment is Working

Run this in Python to verify all packages are installed:

```python
import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt

print("All packages imported successfully!")
print(f"Pandas version: {pd.__version__}")
print(f"NumPy version: {np.__version__}")
```

## Troubleshooting

### If kernel doesn't appear in Jupyter:
```bash
# Make sure ipykernel is installed
conda activate toit_revamp
conda install ipykernel
python -m ipykernel install --user --name toit_revamp --display-name "Python (toit_revamp)"
```

### If VS Code doesn't detect the environment:
1. Check that the environment is activated in your terminal
2. Restart VS Code/Cursor
3. Manually set the interpreter path in settings.json:
   ```json
   {
     "python.defaultInterpreterPath": "~/anaconda3/envs/toit_revamp/bin/python"
   }
   ```

### If packages are missing:
```bash
conda activate toit_revamp
pip install -r requirements.txt
```
