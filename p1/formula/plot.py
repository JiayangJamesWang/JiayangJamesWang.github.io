import numpy as np
import matplotlib.pyplot as plt

def f(x):
    return (np.sin((x - 0.5) * np.pi) / 2) + 0.5

x = np.linspace(0, 1, 500)
y = f(x)

plt.figure(figsize=(8, 6))
plt.plot(x, y, color='b')
plt.title('Plot of the function $p_{rescaled}=\\frac{\\sin\\left(\\left(\\frac{p_{original}-p_{min}}{p_{max}-p_{min}}-0.5\\right)\\cdot\\pi\\right)}{2}+0.5$')
plt.xlabel('$p_{original}$')
plt.ylabel('$p_{rescaled}$')
plt.grid(True)
plt.legend()
plt.show()
