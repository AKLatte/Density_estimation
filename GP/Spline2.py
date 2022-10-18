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


def print_value(y):
    print('yi = [{}]'.format(y))

def calc_error(y, yest):
    return np.abs(y - yest)


def calc_residual(x, y, residual, knots):
    each_residual = []
    save_x, save_residual = copy.copy(x), copy.copy(residual)
    q = 0  # 節点間隔における残差の絶対値の総和
    for i in knots:
        count = 0
        for j in x:
            if j < i:
                q += residual[count]
                count += 1
            else:
                each_residual.append(q)
                q = 0
                x, residual = np.delete(x, np.s_[:count]), np.delete(residual, np.s_[:count])
                break
    if q == 0:
        each_residual.append(sum(residual))
    else:
        each_residual.append(q)
        each_residual.append(0)
    x, residual = copy.copy(save_x), copy.copy(save_residual)  # x, residualのリセット

    return each_residual


def searchKnots(x, y, num, residual, old_knots): 
    before_result = sum(residual)  # 節点配置変更前の残差の絶対値の総和(逐次更新される)
    while True:
        for i in range(len(residual)-1):
            if residual[i] > residual[i+1]:   #Q_iとQ_(i+1)比較
                save_knot = old_knots[i]
                old_knots[i] += 5
                if old_knots[i] >= 82.5:  # knotが最後のknotに追いついたとき
                    old_knots[i] = save_knot
                    continue
                # knot更新時に次のknotの座標を超えてしまった場合の処理
                old_knots = np.sort(old_knots)
                # new_model: 節点配置変更後のスプライン関数
                new_model, _ = get_natural_cubic_spline_model(x, y, minval=min(x), maxval=max(x), n_knots=num, knots=list(old_knots))
                # update_residual: 節点配置変更後の残差Qのリスト
                update_residual = calc_residual(x, y, calc_error(y, new_model.predict(x)), old_knots)
                if sum(residual) > sum(update_residual):
                    residual = update_residual
                else:
                    old_knots[i] = save_knot
        if before_result - sum(residual) > 1e-9:
            before_result = sum(residual)
        else:
            break

    
    return old_knots



# make sample data
x = input_data.dataset[:, 1]
y = input_data.dataset[:, 0]
num = 5
model, knots = get_natural_cubic_spline_model(x, y, minval=min(x), maxval=max(x), n_knots=num)
old_knots = copy.copy(knots)
old_yknots = model.predict(old_knots)  # 節点の初期値(等間隔)
old_yest = model.predict(x)  # 節点の初期値の下での各データに対するスプライン関数の予測値
old_residual = calc_error(y, old_yest)  # 節点初期値の下でのスプライン関数S(xi)とデータyiの残差
print("Sum of residual error: {}, knots: {}".format(sum(old_residual), old_knots))
# 節点間隔の残差の絶対値を算出
new_residual = calc_residual(x, y, old_residual, old_knots)
# 節点配置の最適化
new_knots = searchKnots(x, y, num, new_residual, old_knots)

# 最適なスプライン関数
opt_spline, opt_knots = get_natural_cubic_spline_model(x, y, minval=min(x), maxval=max(x), n_knots=num, knots=list(new_knots))
opt_yest = opt_spline.predict(x)
opt_residual = sum(calc_error(y, opt_yest))
print('opt residual: {}'.format(opt_residual))
print(old_knots)

if __name__ == "__main__":
    plt.plot(x, y, ls='', marker='.', label='original')
    plt.plot(x, old_yest, label='init spline')
    # plt.plot(x, opt_spline.predict(x), label='opt spline')
    plt.scatter(opt_knots, opt_spline.predict(opt_knots), label='opt knot')
    plt.scatter(old_knots, model.predict(old_knots), label='old knot')
    plt.legend()
    plt.show()