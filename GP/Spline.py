import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import copy
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline

import input_data


def get_natural_cubic_spline_model(x, y, minval=None, maxval=None, n_knots=None, knots=None):
    """
    Get a natural cubic spline model for the data.

    For the knots, give (a) `knots` (as an array) or (b) minval, maxval and n_knots.

    If the knots are not directly specified, the resulting knots are equally
    space within the *interior* of (max, min).  That is, the endpoints are
    *not* included as knots.

    Parameters
    ----------
    x: np.array of float
        The input data
    y: np.array of float
        The outpur data
    minval: float 
        Minimum of interval containing the knots.
    maxval: float 
        Maximum of the interval containing the knots.
    n_knots: positive integer 
        The number of knots to create.
    knots: array or list of floats 
        The knots.

    Returns
    --------
    model: a model object
        The returned model will have following method:
        - predict(x):
            x is a numpy array. This will return the predicted y-values.
    """

    if knots:
        spline = NaturalCubicSpline(knots=knots)
    else:
        spline = NaturalCubicSpline(max=maxval, min=minval, n_knots=n_knots)

    p = Pipeline([
        ('nat_cubic', spline),
        ('regression', LinearRegression(fit_intercept=True))
    ])

    p.fit(x, y)

    return p, spline.knots


class AbstractSpline(BaseEstimator, TransformerMixin):
    """Base class for all spline basis expansions."""

    def __init__(self, max=None, min=None, n_knots=None, n_params=None, knots=None):
        if knots is None:
            if not n_knots:
                n_knots = self._compute_n_knots(n_params)
            knots = np.linspace(min, max, num=(n_knots + 2))[1:-1]
            max, min = np.max(knots), np.min(knots)
        self.knots = np.asarray(knots)

    @property
    def n_knots(self):
        return len(self.knots)

    def fit(self, *args, **kwargs):
        return self


class NaturalCubicSpline(AbstractSpline):
    """Apply a natural cubic basis expansion to an array.
    The features created with this basis expansion can be used to fit a
    piecewise cubic function under the constraint that the fitted curve is
    linear *outside* the range of the knots..  The fitted curve is continuously
    differentiable to the second order at all of the knots.
    This transformer can be created in two ways:
      - By specifying the maximum, minimum, and number of knots.
      - By specifying the cutpoints directly.  

    If the knots are not directly specified, the resulting knots are equally
    space within the *interior* of (max, min).  That is, the endpoints are
    *not* included as knots.
    Parameters
    ----------
    min: float 
        Minimum of interval containing the knots.
    max: float 
        Maximum of the interval containing the knots.
    n_knots: positive integer 
        The number of knots to create.
    knots: array or list of floats 
        The knots.
    """

    def _compute_n_knots(self, n_params):
        return n_params

    @property
    def n_params(self):
        return self.n_knots - 1

    def transform(self, X, **transform_params):
        X_spl = self._transform_array(X)
        if isinstance(X, pd.Series):
            col_names = self._make_names(X)
            X_spl = pd.DataFrame(X_spl, columns=col_names, index=X.index)
        return X_spl

    def _make_names(self, X):
        first_name = "{}_spline_linear".format(X.name)
        rest_names = ["{}_spline_{}".format(X.name, idx)
                      for idx in range(self.n_knots - 2)]
        return [first_name] + rest_names

    def _transform_array(self, X, **transform_params):
        X = X.squeeze()
        try:
            X_spl = np.zeros((X.shape[0], self.n_knots - 1))
        except IndexError: # For arrays with only one element
            X_spl = np.zeros((1, self.n_knots - 1))
        X_spl[:, 0] = X.squeeze()

        def d(knot_idx, x):
            def ppart(t): return np.maximum(0, t)

            def cube(t): return t*t*t
            numerator = (cube(ppart(x - self.knots[knot_idx]))
                         - cube(ppart(x - self.knots[self.n_knots - 1])))
            denominator = self.knots[self.n_knots - 1] - self.knots[knot_idx]
            return numerator / denominator

        for i in range(0, self.n_knots - 2):
            X_spl[:, i+1] = (d(i, X) - d(self.n_knots - 2, X)).squeeze()
        return X_spl


def func(x):
    return 1/(1+25*x**2)


def print_value(y):
    print('yi = [{}]'.format(y))


def evaluate(x, y, y_est, knots):
    error = np.abs(y_est - y)
    total = sum(error)
    while True:
        for i in range(len(knots)):
            if error[i] > error[i+1]:
                knot = knots[i]
                knots[i] += 40
                model, _ = get_natural_cubic_spline_model(x, y, minval=min(x), maxval=max(x), n_knots=num, knots=list(knots))
                new_error = np.abs(model.predict(x) - y)
                if sum(error) < sum(new_error):
                    knots[i] = knot
        new_total = sum(new_error)
        if total > new_total:
            total = new_total
        else:
            break
    
    return total, knots, model
        


# make sample data
x = input_data.dataset[:, 1]
y = input_data.dataset[:, 0]

# The number of knots canbe used to control the amount of smooothness
num = 8  # the number of knots
model_5, knots = get_natural_cubic_spline_model(x, y, minval=min(x), maxval=max(x), n_knots=num)
old_knots = copy.copy(knots)  # 初期の等間隔の節点
print("Sum of residual error: {}, knots: {}".format(sum(np.abs(y - model_5.predict(x))), knots))
y_est_5 = model_5.predict(x)
error, new_knots, new_model = evaluate(x, y, y_est_5, knots)
y_est = new_model.predict(x)
new_y_est = new_model.predict(new_knots)  # 更新後のknotsにおける関数の出力値
print('Sum of residual error: {}, new knots: {}'.format(error, new_knots))

plt.plot(x, y, ls='', marker='.', label='originals')
plt.plot(x, y_est_5, marker='.', label='model(n_knots= 6)')
plt.plot(x, y_est, marker='.', label='new model(n_knots=6)', alpha=0.3)
plt.scatter(old_knots, new_model.predict(old_knots), color='blue', alpha=0.3, label='init knot')
plt.scatter(new_knots, new_model.predict(new_knots), color='red', label='new knot')
# plt.plot(x, y_est_15, marker='.', label='n_knots = 15')
plt.legend(); plt.show()