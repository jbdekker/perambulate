import perambulate as pr

df = pr.datasets.load_sinusoid()

# empty condition
A = pr.Condition()

# from other condition
B = pr.Condition(A)

# from mask
C = pr.Condition(df.sinusoid > 0.0)

# valuesearch
D = pr.ValueSearch()

# periodic
E = pr.Periodic()
