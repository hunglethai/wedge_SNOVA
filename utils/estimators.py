from cryptographic_estimators.MQEstimator import *
from tqdm import tqdm
excluded_algorithms_XL = [Bjorklund, Crossbred, DinurFirst, DinurSecond, ExhaustiveSearch, F5, HybridF5, Lokshtanov]

# Raw data
raw_data ="""
108	74	68	 16 	34
99	75	72	 16 	24
116	96	80	 16 	20
162	112	100	 16 	50
180	147	99	 16 	33
180	148	128	 16 	32
145	120	125	 16 	25
216	150	132	 16 	66
243	198	135	 16 	45
280	240	160	 16 	40
175	145	150	 16 	30
				
 112 	68	44	 256 	44
 160 	96	64	 16 	64
 184 	112	72	 256 	72
 244 	148	96	 256 	96
				
86	78	78	 16 	8
81	64	64	 16 	17
118	108	108	 16 	10
154	142	142	 16 	12
				
70	52	54	 2,048,383 	18
84	74	100	 282,475,249 	10
75	55	60	 29,791 	20
67	60	70	 819,628,286,980,801 	7
102	76	78	 2,048,383 	26
124	110	140	 282,475,249 	14
111	82	87	 29,791 	29
99	89	100	 819,628,286,980,801 	10
137	102	105	 2,048,383 	35
168	149	190	 282,475,249 	19
146	108	114	 29,791 	38
124	112	120	 819,628,286,980,801 	12
"""

# Define the list of (n, m, q) parameter sets
parameter_sets = []
for line in raw_data.strip().splitlines():
    if line.strip():
        parts = line.strip().split()
        n, m, q, d = int(parts[0]), int(parts[2]), int(parts[3].replace(",", "")), int(parts[4])
        parameter_sets.append((n-d, m, q))

# Estimator
results = []
for n, m, q in tqdm(parameter_sets, ncols = 100):
    E = MQEstimator(n=n, m=m, q=q, excluded_algorithms=excluded_algorithms_XL)
    output = E.estimate()
    time_value = next(iter(output.values()))['estimate']['time']
    results.append(round(time_value))
    # print(f"\nEstimating for n={n}, m={m}, q={q}:", round(time_value))
print(results)