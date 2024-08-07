import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import mplcursors

percentage = np.arange(0, 101, 1)

class MF:
    def __init__(self, label: str, x: np.array, very_lows: tuple, lows: tuple, mediums: tuple, averages: tuple, highs: tuple) -> None:
        self.label = label
        self.x = x
        self.very_lows = very_lows
        self.lows = lows
        self.mediums = mediums
        self.averages = averages
        self.highs = highs

    def _very_low(self, x):
        a, b = self.very_lows
        if x <= a:
            return 1
        elif a < x <= b:
            return (b - x) / (b - a)
        else:
            return 0

    def very_low(self):
        return np.array([self._very_low(xi) for xi in self.x])

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

plt.figure(figsize=(10, 6))

percentage_mf = MF('Percentage', percentage, (0, 39), (38, 45, 52), (49, 58, 66), (64, 73, 81), (77, 100))
plt.plot(percentage, percentage_mf.very_low(), label='Very Low')
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

df = pd.read_csv("../data/percentage.csv")

class FuzzyVariable:
    def __init__(self, universe, label):
        self.universe = universe
        self.label = label

    def get_fuzzy_var(self, value, obj):
        fuzzy_values = [obj._very_low(value), obj._low(value), obj._medium(value), obj._average(value), obj._high(value)]
        return fuzzy_values

def fuzzy_values(percentage):
    return FuzzyVariable(percentage_mf.x, 'Percentage').get_fuzzy_var(percentage, percentage_mf)

fuzzy_df = pd.DataFrame([fuzzy_values(x) for x in df['Percentage']], columns=['Very Low', 'Low', 'Medium', 'Average', 'High'])
df = pd.concat([df, fuzzy_df], axis=1)

def area_tr(mu, a, b, c):
    base = abs(c - a)
    height = mu
    area = (base * height) / 2
    return area

def defuzzification(vl_op, fd_op, basic_op, elem_op, inter_op):
    area_vl = area_pl = area_pm = area_ps = area_ns = 0
    c_vl = c_fd = c_basic = c_elem = c_inter = 0

    if vl_op != 0:
        area_vl= area_tr(vl_op, 0, 0, 39)
        c_vl= 13
    if fd_op != 0:
        area_pl= area_tr(fd_op, 38, 45, 52)
        c_fd=45
    if basic_op != 0:
        area_pm = area_tr(basic_op, 49, 58, 66)
        c_basic = 58
    if elem_op != 0:
        area_ps = area_tr(elem_op, 64, 73, 81)
        c_elem = 73
    if inter_op != 0:
        area_ns = area_tr(inter_op, 77, 100,100)
        c_inter = 92.33

    numerator = area_vl * c_vl + area_pl * c_fd + area_pm * c_basic + area_ps * c_elem + area_ns * c_inter
    denominator = area_vl + area_pl + area_pm + area_ps + area_ns

    return numerator / denominator if denominator != 0 else 0

def rule(fuzzy_vars):
    very_low, low, medium, average, high = zip(*fuzzy_vars)
    vl_op = max(very_low)
    fd_op = max(low)
    basic_op = max(medium)
    elem_op = max(average)
    inter_op = max(high)

    return vl_op, fd_op, basic_op, elem_op, inter_op

crisp_outputs = []
for i, row in df.iterrows():
    fuzzy_vals = [row['Very Low'], row['Low'], row['Medium'], row['Average'], row['High']]
    vl_op, fd_op, basic_op, elem_op, inter_op = rule([fuzzy_vals])
    crisp_output = defuzzification(vl_op, fd_op, basic_op, elem_op, inter_op)
    crisp_outputs.append(crisp_output)

df['Crisp Output'] = crisp_outputs

def classify_marks(row):
    crisp_output = row['Crisp Output']
    
    if crisp_output >= 80:
        return 'Distinction'
    elif crisp_output >= 65:
        return 'First Division'
    elif crisp_output >= 50:
        return 'Second Division'
    elif crisp_output >= 40:
        return 'Pass'
    else:
        return 'Fail'

df['Class'] = df.apply(classify_marks, axis=1)
print(df)

plt.figure(figsize=(12, 8))
colors = {'Distinction': 'green', 'First Division': 'blue', 'Second Division': 'yellow', 'Pass': 'orange', 'Fail': 'red'}

scatter = plt.scatter(df['Percentage'], df['Crisp Output'], c=df['Class'].map(colors), label=df['Class'])

cursor = mplcursors.cursor(scatter, hover=True)
cursor.connect("add", lambda sel: sel.annotation.set(
    text=df['Name'].iloc[sel.index],
    bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.5')
))

handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color, markersize=10, label=cls) 
           for cls, color in colors.items()]
plt.legend(handles=handles, title='Class')

plt.xlabel('Actual Percentage')
plt.ylabel('Crisp Output')
plt.grid(True)
plt.title('Students Percentage, Crisp Values, and Class')
plt.tight_layout()
plt.show()

test_percentages = [16, 39, 60, 78, 79.8, 97]
expected_outputs = [13, 45, 58, 73, 73, 92.33]  

for test_percentage, expected in zip(test_percentages, expected_outputs):
    fuzzy_values = FuzzyVariable(percentage_mf.x, 'Percentage').get_fuzzy_var(test_percentage, percentage_mf)
    vl_op, fd_op, basic_op, elem_op, inter_op = rule([fuzzy_values])
    test_crisp_output = defuzzification(vl_op, fd_op, basic_op, elem_op, inter_op)
    print(f"\nTest Percentage: {test_percentage}")
    print(f"Expected Crisp Output: {expected}")
    print(f"Computed Crisp Output: {test_crisp_output}")
