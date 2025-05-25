import dynex
import dimod
from pyqubo import Array

# Parameters (customize as needed)
num_staff = 5  # Number of staff members
num_shifts = 3  # Number of shifts (e.g., morning, evening, night)
max_shifts_per_week = 3  # Maximum shifts a staff can work per week
min_shifts_per_day = 1  # Minimum number of staff per shift
availability = [
    [1, 1, 0],  # Staff 1 availability for shifts (can work morning and evening)
    [1, 0, 1],  # Staff 2 availability for shifts (can work morning and night)
    [0, 1, 1],  # Staff 3 availability for shifts (can work evening and night)
    [1, 1, 1],  # Staff 4 availability for all shifts
    [1, 0, 0],  # Staff 5 availability for morning only
]

# Binary variable for each staff-member and shift assignment (0 or 1)
x = Array.create('x', (num_staff, num_shifts), 'BINARY')

# Objective Function (Energy)
H = 0

# 1. Assign each shift to the required number of staff (min_shifts_per_day)
for j in range(num_shifts):
    H += (sum(x[i, j] for i in range(num_staff)) - min_shifts_per_day)**2

# 2. Ensure staff are assigned only to available shifts
for i in range(num_staff):
    for j in range(num_shifts):
        H += (x[i, j] - availability[i][j])**2  # Penalize assignments outside availability

# 3. Ensure staff don't exceed the max shifts per week
for i in range(num_staff):
    H += (sum(x[i, j] for j in range(num_shifts)) - max_shifts_per_week)**2

# Compile the QUBO model
model = H.compile()
Q, offset = model.to_qubo(index_label=True)

# Submit the QUBO to Dynex for solving
try:
    print("📤 Submitting QUBO to Dynex...")
    samplesetq = dynex.sample_qubo(
        Q,
        offset,
        mainnet=True,  # Set to False for testnet
        description='Healthcare Staff Scheduling',
        num_reads=1000,
        annealing_time=200
    )

    # Display the results
    print('Optimal Schedule:')
    print(samplesetq)

except Exception as e:
    print(f"Error submitting QUBO to Dynex: {e}")
