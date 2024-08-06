import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Define the universe of discourse
attendance = np.arange(0, 31, 1)

# Define triangular membership functions for each category
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

# Define membership functions for attendance
attendance_mf = MF('Attendance', attendance, (0, 10), (8, 15, 22), (18, 22, 26), (24, 30))

# Load dataset
df = pd.read_csv(r"fuzzy-logic/data/attendance.csv")

class FuzzyVariable:
    def __init__(self, universe, label):
        self.universe = universe
        self.label = label

    def get_fuzzy_var(self, mark, obj):
        fuzzy_values = [obj._low(mark), obj._medium(mark), obj._average(mark), obj._high(mark)]
        return fuzzy_values

# Compute fuzzy values and add them to DataFrame
students = df['Name'].values
attendances = df['Attendance'].values

def fuzzy_values(attendance):
    return FuzzyVariable(attendance_mf.x, 'Attendance').get_fuzzy_var(attendance, attendance_mf)

# Add fuzzy values to DataFrame
fuzzy_df = pd.DataFrame([fuzzy_values(att) for att in attendances], columns=['Low', 'Medium', 'Average', 'High'])
df = pd.concat([df, fuzzy_df], axis=1)

# Calculate fuzzy logic rules and defuzzification
def area_tr(mu, a, b, c):
    x1 = mu * (b - a) + a
    x2 = c - mu * (c - b)
    d1 = c - a
    d2 = x2 - x1
    return (1 / 2) * mu * (d1 + d2)

def defuzzification(fd_op, basic_op, elem_op, inter_op):
    area_pl = area_pm = area_ps = area_ns = 0
    c_fd = c_basic = c_elem = c_inter = 0

    if fd_op != 0:
        area_pl, c_fd = area_tr(fd_op, 0, 10, 20), 10
    if basic_op != 0:
        area_pm = area_tr(basic_op, 8, 15, 22)
        c_basic = 15
    if elem_op != 0:
        area_ps = area_tr(elem_op, 18, 22, 26)
        c_elem = 22
    if inter_op != 0:
        area_ns = area_tr(inter_op, 24, 30, 32)
        c_inter = 30

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

# Apply rules and defuzzification to each student
crisp_outputs = []
for i, row in df.iterrows():
    fuzzy_vals = [row['Low'], row['Medium'], row['Average'], row['High']]
    fd_op, basic_op, elem_op, inter_op = rule([fuzzy_vals])
    crisp_output = defuzzification(fd_op, basic_op, elem_op, inter_op)
    crisp_outputs.append(crisp_output)

df['Crisp Output'] = crisp_outputs

# Classify crisp outputs
def classify_attendance(row):
    # Get the fuzzy values for each category
    low = row['Low']
    medium = row['Medium']
    average = row['Average']
    high = row['High']
    
    # Determine the class with the highest fuzzy value
    max_value = max(low, medium, average, high)
    if max_value == low:
        return 'Low'
    elif max_value == medium:
        return 'Medium'
    elif max_value == average:
        return 'Average'
    else:
        return 'High'

df['Class'] = df.apply(classify_attendance, axis=1)

# Print the DataFrame to verify new columns
print(df)

# Plot the students' actual attendance, crisp value, and class
plt.figure(figsize=(12, 8))
colors = {'Low': 'red', 'Medium': 'yellow', 'Average': 'blue', 'High': 'green'}

# Plot the scatter plot
scatter = plt.scatter(df['Attendance'], df['Crisp Output'], c=df['Class'].map(colors), label=df['Class'])

# Create a legend for the plot
handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color, markersize=10, label=cls) 
           for cls, color in colors.items()]
plt.legend(handles=handles, title='Class')

plt.xlabel('Actual Attendance')
plt.ylabel('Crisp Output')
plt.grid(True)
plt.title('Students Attendance, Crisp Values, and Class')
plt.tight_layout()
plt.show()

# Validation with sample data
test_attendances = [5, 12, 19, 25, 30]
expected_outputs = [10, 15, 22, 30, 30]  # Example expected crisp values for the test data

for test_attendance, expected in zip(test_attendances, expected_outputs):
    test_fuzzy_values = FuzzyVariable(attendance_mf.x, 'Attendance').get_fuzzy_var(test_attendance, attendance_mf)
    test_fd_op, test_basic_op, test_elem_op, test_inter_op = rule([test_fuzzy_values])
    test_crisp_output = defuzzification(test_fd_op, test_basic_op, test_elem_op, test_inter_op)
    print(f"\nTest Attendance: {test_attendance}")
    print(f"Expected Crisp Output: {expected}")
    print(f"Computed Crisp Output: {test_crisp_output}")
