
from scipy.optimize import newton
import scipy

# SciPy GitHub Issue #6379
# https://github.com/scipy/scipy/issues/6379

def f(x):
    return x**4 - x**2 + 1

root, info = newton(
    f,
    x0=0.001,
    full_output=True,
    disp=False
)

print("SciPy version:", scipy.__version__)
print("Returned root:", root)
print("Function value:", f(root))
print("Converged:", info.converged)

# Check the returned answer
if abs(f(root)) > 1e-6:
    print("WARNING: Result is not close to a root")
else:
    print("Result satisfies residual tolerance")
