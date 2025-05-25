🩺 Healthcare Staff Scheduling using Dynex

This mini project was developed as a part of our college mini-project to explore the application of quantum-inspired optimization in real-world scenarios. We have implemented a Healthcare Staff Scheduling solution using the Dynex decentralized neuromorphic computing platform.

🚀 Project Overview

In the healthcare sector, efficient staff scheduling is essential to ensure that patients receive timely and quality care. Our project focuses on assigning staff to different shifts while considering:

Staff availability

Shift coverage requirements

Maximum working limits for staff

To solve this optimization problem, we formulated it as a QUBO (Quadratic Unconstrained Binary Optimization) model and submitted it to the Dynex quantum-inspired computing platform.

🧠 Technologies Used

Python

PyQUBO – For building QUBO models

dimod – Binary Quadratic Model interface

Dynex – For solving the QUBO using a decentralized neuromorphic supercomputing approach

⚙️ Problem Setup

Parameters

num_staff = 5 – Total number of staff members

num_shifts = 3 – Morning, Evening, Night shifts

max_shifts_per_week = 3 – No staff works more than 3 shifts per week

min_shifts_per_day = 1 – At least one staff member per shift

availability – A matrix defining which shifts each staff member is available for

Constraints Applied:

✅ Each shift must be covered by at least the required number of staff

✅ A staff member can only be assigned to a shift if they're available for it

✅ No staff member should exceed the max shifts per week

📚 Learning Outcomes

Learned to model real-world constraints as a QUBO

Understood basics of quantum-inspired optimization

Hands-on with Dynex’s decentralized computational power

