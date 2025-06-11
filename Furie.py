import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import quad
from scipy.interpolate import CubicSpline


def main():
    X = np.array([0, 1, 2, 3, 4, 5, 6, 7])
    Y = np.array([0, 1, 2, 2, 5, 4, 3, 0])

    cspline = CubicSpline(X, Y)
    elem_count = int(input())

    L = X[-1] - X[0] 
    k = np.arange(0, elem_count + 1)

    def integrate_cos(x, k):
        return np.cos(2 * np.pi * x * k / L) * cspline(x)

    def integrate_sin(x, k):
        return np.sin(2 * np.pi * x * k / L) * cspline(x)

    a0 = (2 / L) * quad(cspline, 0, L)[0]
    ak = (2 / L) * np.array([quad(integrate_cos, 0, L, args=(ki,))[0] for ki in k])
    bk = (2 / L) * np.array([quad(integrate_sin, 0, L, args=(ki,))[0] for ki in k])

    def fourier_series(x):
        series = -a0/2 + np.sum(ak.reshape(-1, 1) * np.cos(2 * np.pi * x * k.reshape(-1, 1) / L) + bk.reshape(-1, 1) * np.sin(2 * np.pi * x * k.reshape(-1, 1) / L), axis=0)
        return series.flatten()

    x_values = np.linspace(X[0], X[-1], 1000)
    y_interp = cspline(x_values)
    y_fourier = fourier_series(x_values)
    # r = y_fourier[0] - y_interp[0]
    # print(r)
    # y_fourier -= r
    print(f"a_0 = {a0}" )
    print(f"a_k = {ak}" )
    print(f"b_k = {bk}" )

    plt.plot(x_values, y_interp, 'r', label='Интерполированная функция')
    plt.plot(x_values, y_fourier, 'b--', label=f'Сумма ряда Фурье (N={elem_count})')
    plt.scatter(X, Y, color='black', label='Табличные данные')
    plt.legend(loc='upper right')
    plt.xlabel( 'X')
    plt.ylabel('Y')
    plt.title('График интерполированной функции и ряда Фурье')
    plt.grid(True)
    plt.show()

    # x_range = np.linspace(-10,10,1000)
    # a = np.array([1,2,3])
    # # np.fft.

    # fig= plt.figure(figsize=(12,7))
    # plt.plot()

    # plt.grid()
    # plt.xlim(-10,10)
    # plt.ylim(-10,10)
    # plt.show()


    # t = np.arange(256)
    # sp = np.fft.fft(np.sin(t))
    # freq = np.fft.fftfreq(t.shape[-1])
    # plt.plot(freq, sp.real, freq, sp.imag)
    # plt.show()


if __name__ == "__main__":
    main()