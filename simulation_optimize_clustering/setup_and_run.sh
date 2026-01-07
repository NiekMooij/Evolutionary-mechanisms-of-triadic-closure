#!/bin/bash

# Step 1: Create a virtual environment
python3 -m venv venv_temp

# Step 2: Activate the virtual environment
source venv_temp/bin/activate

# Step 3: Install dependencies
pip install -r requirements.txt

# Step 4: Run your code
python networks.py
python simulate.py
python analyse.py
python data.py
python plot.py

# Step 5: Deactivate the virtual environment
deactivate

# Step 6: Remove the virtual environment
rm -rf venv_temp