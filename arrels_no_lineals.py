# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt

# 1. Evaluar f y df (funciones definidas arriba)
def f(x):
    return x**5 - 4*x**4 + 7*x**3 - 21*x**2 + 6*x +18

def df(x):
    return 5*x**4 - 16*x**3 + 21*x**2 - 42*x + 6


# 2. Graficar f en [-1,4]
def plot_f():
    xs = np.linspace(-1,4,400)
    plt.figure()
    plt.plot(xs, f(xs), label='f(x)')
    plt.axhline(0, color='k', lw=0.8)
    plt.xlabel('x'); plt.ylabel('f1(x)')
    plt.title('Gráfica de f1 en [-1,4]')
    plt.legend(); plt.show()

def find_real_roots(coef = [1, -4, 7, -21, 6, 18]):
    roots = np.roots(coef)
    return [r.real for r in roots if abs(r.imag) < 1e-8]

# 3. Newton iterativo para n iteraciones
def newton_iter(f, df, x0, n):
    xs = [x0]
    for k in range(n):
        xs.append(xs[-1] - f(xs[-1]) / df(xs[-1]))
    return np.array(xs)

# 4. Raíces con distintos x0
# Uso: roots = {x0: newton_iter(f1, df1, x0, n)[-1] for x0 in initials}

# 5. Cálculo de error aproximado r_k ≃ |x_k - x_{k+1}|/|x_{k+1}|
def errors_rel(xs):
    return np.abs((xs[:-1] - xs[1:]) / xs[1:])

# 6. Newton con criterios de parada y vector de residuos
def newton(f, df, x0, max_iter=50, comply_with_convg_criteria = True, tol_x=1e-8, tol_f=1e-8):
    xs = [x0]
    residuals = []
    for k in range(max_iter):
        xk = xs[-1]
        fx = f(xk)
        if abs(fx) < tol_f:
            break
        dfx = df(xk)
        x_next = xk - fx/dfx
        residuals.append(abs(fx))
        if comply_with_convg_criteria and abs(x_next - xk) <= tol_x * abs(x_next):
            xs.append(x_next)
            break
        xs.append(x_next)
    return np.array(xs), np.array(residuals)

# ----------------------
# Ejercicio adicional 1: Newton en g
# g(x) = x^5 - 2x^4 - 6x^3 + 12x^2 + 9x - 18
# ----------------------

def g(x):
    return x**5 - 2*x**4 - 6*x**3 + 12*x**2 + 9*x - 18

def dg(x):
    return 5*x**4 - 8*x**3 - 18*x**2 + 24*x + 9

# ----------------------
# Ejercicio adicional 2: Método de Whittaker
# x_{k+1} = x_k - f(x_k)/m
# ----------------------

def whittaker(f, x0, m, tol=1e-10, max_iter=100, max_val=1e6):
    xs = [x0]
    for k in range(max_iter):
        xk = xs[-1]
        x_next = xk - f(xk)/m
        
        # si diverge demasiado lo paramos
        if abs(x_next) > max_val:
            break

        xs.append(x_next)
        if abs(x_next - xk) < tol:
            break
    return np.array(xs)

# ----------------------
# Ejercicio adicional 3: Método de la secante
# x_{k+2} = x_{k+1} - f(x_{k+1})*(x_{k+1}-x_k)/(f(x_{k+1})-f(x_k))
# ----------------------

def secant(f, x0, x1, tol=1e-12, max_iter=50):
    xs = [x0, x1]

    for k in range(max_iter-1):
        xk, xk1 = xs[-2], xs[-1]
        fxk, fxk1 = f(xk), f(xk1)
        # fórmula de la secante
        x_next = xk1 - fxk1 * (xk1 - xk) / (fxk1 - fxk)
        
        if abs(x_next - xs[-1]) < tol:
            break

        xs.append(x_next)
    return np.array(xs)

# ----------------------
# Ejercicio adicional 4: Newton con derivada numérica
# f'(a) ≈ [f(a+h)-f(a)]/h
# ----------------------

def derivative_num(f, a, h=1e-6):
    return (f(a+h) - f(a)) / h

def newton_num_deriv(f, x0, h=1e-6, tol=1e-12, max_iter=50):
    xs = [x0]
    residuals = [abs(f(x0))]
    for k in range(max_iter):
        xk = xs[-1]
        fx = f(xk)
        df_approx = derivative_num(f, xk, h)
        if df_approx == 0:
            print("Derivada aproximada cero en iteración", k)
            break
        x_next = xk - fx / df_approx
        residuals.append(abs(f(x_next)))
        if abs(x_next - xk) < tol:
            print(x_next, xk)
            
            break
        xs.append(x_next)
    return np.array(xs), np.array(residuals)