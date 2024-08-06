# Fuzzy Logic Marks Classification

This project implements a fuzzy logic system to classify student marks into different categories using triangular membership functions.
It uses fuzzy logic to classify student marks into categories like Distinction, First Division, Second Division, Third Division, and Not Graded. The classification is based on triangular membership functions defined for each category.

## Features
- **Triangular Membership Functions:** Utilizes triangular membership functions to define fuzzy sets for marks classification.
- **Fuzzy Variable Calculation:** Computes fuzzy values for each student's marks.
- **Defuzzification:** Converts fuzzy values into crisp outputs to determine final classification.
- **Visualization:** Plots membership functions and classification results using matplotlib.
- **Interactive Plot:** Uses mplcursors to display student names on hover in the scatter plot.

## Setup Instructions

### Step 1: Create and Activate a Virtual Environment

For different operating systems, use the following commands:

#### Linux and Mac
```bash
# Create a virtual environment
python3 -m venv venv

# Activate the virtual environment
source venv/bin/activate
```

#### Windows 
```bash
# Create a virtual environment
python -m venv venv

# Activate the virtual environment
venv\Scripts\activate
```


### Step 2: Install Deependencies
```bash
pip install -r requirements.txt
```


### Step 3 : Run Python Script
```bash
cd src
python result.py
```





## Contributors

- **Aditi Kharel** - 077BEI008
- **Asmita Sigdel** - 077BEI013