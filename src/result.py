import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import mplcursors

# Universe of discourse
marks = np.arange(0, 100, 1)

# Triangular membership function for each category
class MF:
    def __init__(self, label, x, lows, mediums, averages, highs):
        self.label = label
        self.x = x
        self.lows = lows
        self.mediums = mediums
        self.averages = averages
        self.highs = highs

    def _low(self, x):
        a, b = self.lows
        if x <= a:
            return 1
        elif a <= x <= b:
            return (b - x) / (b - a)
        else:
            return 0

    def _medium(self, x):
        a, b, c = self.mediums
        if x <= a or x > c:
            return 0
        elif a < x <= b:
            return (x - a) / (b - a)
        elif b < x <= c:
            return (c - x) / (c - b)
        else:
            return 0

    def _average(self, x):
        a, b, c = self.averages
        if x <= a or x > c:
            return 0
        elif a < x <= b:
            return (x - a) / (b - a)
        elif b < x <= c:
            return (c - x) / (c - b)
        else:
            return 0

    def _high(self, x):
        a, b = self.highs
        if x <= a:
            return 0
        elif a < x <= b:
            return (x - a) / (b - a)
        else:
            return 1

# Membership Functions
marks_mf = MF('Marks', marks, (0, 40), (30, 50, 70), (60, 75, 90), (85, 100))


# Triangular membership functions for each category
class MF:
    def __init__(self, label: str, x: np.array, lows: tuple, mediums: tuple, averages: tuple, highs: tuple) -> None:
        self.label = label
        self.x = x
        self.lows = lows
        self.mediums = mediums
        self.averages = averages
        self.highs = highs

    def _low(self, x):
        a, b, c = self.lows
        if x <= a or x > c:
            return 0
        elif a < x <= b:
            return (x - a) / (b - a)
        elif b < x <= c:
            return (c - x) / (c - b)
        else:
            return 0

    def low(self):
        return np.array([self._low(xi) for xi in self.x])

    def _medium(self, x):
        a, b, c = self.mediums
        if x <= a or x > c:
            return 0
        elif a < x <= b:
            return (x - a) / (b - a)
        elif b < x <= c:
            return (c - x) / (c - b)
        else:
            return 0

    def medium(self):
        return np.array([self._medium(xi) for xi in self.x])

    def _average(self, x):
        a, b, c = self.averages
        if x <= a or x > c:
            return 0
        elif a < x <= b:
            return (x - a) / (b - a)
        elif b < x <= c:
            return (c - x) / (c - b)
        else:
            return 0

    def average(self):
        return np.array([self._average(xi) for xi in self.x])

    def _high(self, x):
        a, b = self.highs
        if x <= a:
            return 0
        elif a < x <= b:
            return (x - a) / (b - a)
        else:
            return 1

    def high(self):
        return np.array([self._high(xi) for xi in self.x])



percentage = np.arange(0, 100, 1)

# Membership functions for percentage
plt.figure(figsize=(10, 6))
percentage_mf = MF('Percentage', percentage, (38, 45, 52), (49, 58, 66), (64, 73, 81), (77, 100))
plt.plot(percentage, percentage_mf.low(), label='Low')
plt.plot(percentage, percentage_mf.medium(), label='Medium')
plt.plot(percentage, percentage_mf.average(), label='Average')
plt.plot(percentage, percentage_mf.high(), label='High')

plt.title('Triangular Membership Function')
plt.xlabel('Percentage')
plt.ylabel('Membership Degree')
plt.legend()
plt.grid(True)
plt.show()

# Load dataset
df = pd.read_csv(r"/home/asmita/Desktop/7th Sem/AI/Final/fuzzy-logic/data/marks.csv")

class FuzzyVariable:
    def __init__(self, universe, label):
        self.universe = universe
        self.label = label

    def get_fuzzy_var(self, mark, obj):
        fuzzy_values = [obj._low(mark), obj._medium(mark), obj._average(mark), obj._high(mark)]
        return fuzzy_values

# Computation of fuzzy values
students = df['Name'].values
marks = df['Marks'].values

def fuzzy_values(marks):
    return FuzzyVariable(marks_mf.x, 'Marks').get_fuzzy_var(marks, marks_mf)

# Addition of fuzzy values to DataFrame
fuzzy_df = pd.DataFrame([fuzzy_values(mark) for mark in marks], columns=['Low', 'Medium', 'Average', 'High'])
df = pd.concat([df, fuzzy_df], axis=1)

# Calculation of fuzzy logic rules
def area_tr(mu, a, b, c):
    x1 = mu * (b - a) + a
    x2 = c - mu * (c - b)
    d1 = c - a
    d2 = x2 - x1
    return (1 / 2) * mu * (d1 + d2)

# Defuzzification
def defuzzification(fd_op, basic_op, elem_op, inter_op):
    area_pl = area_pm = area_ps = area_ns = 0
    c_fd = c_basic = c_elem = c_inter = 0

    if fd_op != 0:
        area_pl, c_fd = area_tr(fd_op, 0, 25, 50), 25
    if basic_op != 0:
        area_pm = area_tr(basic_op, 25, 50, 75)
        c_basic = 50
    if elem_op != 0:
        area_ps = area_tr(elem_op, 50, 75, 100)
        c_elem = 75
    if inter_op != 0:
        area_ns = area_tr(inter_op, 75, 100, 100)
        c_inter = 100

    numerator = area_pl * c_fd + area_pm * c_basic + area_ps * c_elem + area_ns * c_inter
    denominator = area_pl + area_pm + area_ps + area_ns

    if denominator == 0:
        return 0
    else:
        return numerator / denominator

def rule(fuzzy_vars):
    low, medium, average, high = zip(*fuzzy_vars)
    fd_op = max(low)
    basic_op = max(medium)
    elem_op = max(average)
    inter_op = max(high)

    return fd_op, basic_op, elem_op, inter_op

# Applying rules and defuzzification to each student
crisp_outputs = []
for i, row in df.iterrows():
    fuzzy_vals = [row['Low'], row['Medium'], row['Average'], row['High']]
    fd_op, basic_op, elem_op, inter_op = rule([fuzzy_vals])
    crisp_output = defuzzification(fd_op, basic_op, elem_op, inter_op)
    crisp_outputs.append(crisp_output)

df['Crisp Output'] = crisp_outputs

# Classifying crisp outputs
def classify_marks(row):
    crisp_output = row['Crisp Output']
    
    if crisp_output >= 80:
        return 'Distinction'
    elif crisp_output >= 70:
        return 'First Division'
    elif crisp_output >= 60:
        return 'Second Division'
    elif crisp_output >= 50:
        return 'Third Division'
    else:
        return 'Not Graded'

df['Class'] = df.apply(classify_marks, axis=1)

# DataFrame to verify new columns
print(df)

# Plotting actual attendance, crisp value, and class
plt.figure(figsize=(12, 8))
colors = {'Distinction': 'green', 'First Division': 'blue', 'Second Division': 'yellow', 'Third Division': 'orange', 'Not Graded': 'red'}

# Scatter Plot
scatter = plt.scatter(df['Marks'], df['Crisp Output'], c=df['Class'].map(colors), label=df['Class'])

# Add mplcursors to show student names on hover with custom annotation box
cursor = mplcursors.cursor(scatter, hover=True)
cursor.connect("add", lambda sel: sel.annotation.set(
    text=df['Name'].iloc[sel.index],
    bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.5')
))

# Creation of legend for the plot
handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color, markersize=10, label=cls) 
           for cls, color in colors.items()]
plt.legend(handles=handles, title='Class')

plt.xlabel('Actual Marks')
plt.ylabel('Crisp Output')
plt.grid(True)
plt.title('Students Marks, Crisp Values, and Class')
plt.tight_layout()
plt.show()

# Validation with sample data
test_marks = [16, 39, 60, 80, 97]
expected_outputs = [32, 48, 71, 97, 97]  

for test_mark, expected in zip(test_marks, expected_outputs):
    test_fuzzy_values = FuzzyVariable(marks_mf.x, 'Marks').get_fuzzy_var(test_mark, marks_mf)
    test_fd_op, test_basic_op, test_elem_op, test_inter_op = rule([test_fuzzy_values])
    test_crisp_output = defuzzification(test_fd_op, test_basic_op, test_elem_op, test_inter_op)
    print(f"\nTest Marks: {test_mark}")
    print(f"Expected Crisp Output: {expected}")
    print(f"Computed Crisp Output: {test_crisp_output}")
