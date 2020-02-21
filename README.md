# peakfit
Wrapper around Scipy curve_fit to fit peak shaped data



```python
import numpy as np
import matplotlib.pyplot as plt

from peakfit import *


# Generate data
x = np.linspace(-5, 5, 123)
y = 7 + 0.1*np.random.randn(*x.shape)
y += Gauss()(x, 0.5, 1, 1)

# Fit using automatic estimation of initial parameters:
results, fit = peakfit(x, y, Gauss())

for r in results:
    print(r)

# Graph
plt.plot(x, y, '.k', label='data');
plt.plot(x, fit(x), 'r-', label='fit');
plt.xlabel('x'); plt.ylabel('y'); plt.legend();
```


```
{'x0': 0.5245303649979836, 'fwhm': 1.0589350763612955, 'amplitude': 0.8981417133942248}
{'slope': 0.0011523066888828963, 'intercept': 6.993703785318166}
```

![example_fit](./example/example_fit.png)

