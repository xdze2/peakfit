# peakfit

Wrapper around scipy _curve_fit(*)_ to fit peak shaped data

- include basic peak functions (Gauss, Lorentzian, PseudoVoigt)
- method to automate estimation of the initial parameters 
- allows to sum functions, for instance to fit double peak or peak with a background (see [test notebook](test_peakfit.ipynb))
- returns parameters in a dictionary with meaningful key names

[*] [_curve_fit_ documentation](https://docs.scipy.org/doc/scipy-1.5.1/reference/generated/scipy.optimize.curve_fit.html#scipy.optimize.curve_fit), non-linear least square, usually Levenberg-Marquardt algorithm

_note:_ uses Jupyter notebook, and [jupytext](https://jupytext.readthedocs.io/en/latest/index.html) to pair notebooks to _light Script_ py format

## Simple example

```python
import numpy as np
import matplotlib.pyplot as plt
from peakfit import *

# Generate random test data
x = np.linspace(-5, 5, 123)
y = 7 + 0.1*np.random.randn(*x.shape)
y += Gauss()(x, 0.5, 1, 1)

# Fit using automatic estimation of initial parameters
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


## Install
    
Copy the `peakfit.py` file in your project folder...

## Next 
- package install
- error estimation
- include function name is results ? for Sum ?
- standard graph for verification

## Also
- https://fityk.nieto.pl/