import sys
import numpy as np
import matplotlib.pyplot as plt
from arrels_no_lineals import plot_f, find_real_roots, newton_iter, newton, f, df, g, dg, whittaker, secant, derivative_num, newton_num_deriv


def main_menu():

    niter = 10
    initial_approx = [-1, 2, 3, 2.5]

    while True:
        print("\n=== MENÚ BÚSQUEDA DE RAÍCES ===")
        print("1. Graficar f en [-1,4]")
        print("2. Newton fijo para f")
        print("3. Newton con aproximaciones iniciales fijas")
        print("4. Errores absoultos de las iteraciones de Newton fijo con aproximaciones iniciales fijas")
        print("5. Errores absolutos de las iteraciones de Newton con criterio de convergencia con aproximaciones iniciales fijas")
        print("6. Ejercicios adicionales")
        print("7. Salir")
        choice = input("Selecciona una opción [1-7]: ")

        match choice:
            case '1':
                plot_f()
                roots = find_real_roots()
                print("Soluciones (reales) de f: ")
                for r in sorted(roots):
                    abs_err = abs(f(r))
                    rel_err = abs_err / abs(r)
                    print(f"-> {r:.6f}, error relativo: {rel_err:.6f}, error absoluto: {abs_err:.6f}")

            case '2':
                try:
                    x0 = float(input("Valor inicial x0: "))
                    niter = int(input("Número de iteraciones: "))
                    xs = newton_iter(f, df, x0, niter)
                    print(f"Raíz aproximada tras {niter} iteraciones: {xs[-1]:.10f}")
                except ValueError:
                    print("Entrada no válida. Debe ser un número.")

            case '3':
                print("Soluciones (reales) de f con Newton fijo partiendo de x0:")
                for x0 in initial_approx:
                    xs = newton_iter(f, df, x0, niter)
                    print(f"x0 = {x0} -> {xs[-1]}")


            case '4':
                for x0 in initial_approx:
                    xs, res = newton(f, df, x0, niter, False)
                    print(f"Converge en {len(xs)-1} iteraciones a {xs[-1]:.10f}")
                    plt.figure()
                    plt.plot(range(len(res)), np.log10(res), marker='o')
                    plt.title(f'|f(x_k)| log10, x0={x0}')
                    plt.xlabel('Iteración k'); plt.ylabel('log10(abs_error)')
                    plt.show()
                
            case '5':
                for x0 in initial_approx:
                    xs, res = newton(f, df, x0, niter)
                    print(f"Converge en {len(xs)-1} iteraciones a {xs[-1]:.10f}")
                    plt.figure()
                    plt.plot(range(len(res)), np.log10(res), marker='o')
                    plt.title(f'|f(x_k)| log10, x0={x0}')
                    plt.xlabel('Iteración k'); plt.ylabel('log10(abs_error)')
                    plt.show()

            case '6':
                additional_exs_menu()

            case '7':
                print("Saliendo...")
                sys.exit(0)
            case _:
                print("Opción no válida. Intenta de nuevo.")

def additional_exs_menu():
    initial_approxs = [-1, 1, 3]
    niter = 10
    
    while True:
        print("\n=== MENÚ BÚSQUEDA EJERCICIOS ADICIONALES ===")
        print("1. Newton fijo para g")
        print("2. Whittaker de g con x0 = 1 y x0 = 2 y valores de m fijos")
        print("3. Metodo de la secante")
        print("4. Newton con derivada numerica")
        print("5. Salir")
        choice = input("Selecciona una opción [1-5]: ")

        match choice:
            case '1':
                for x0 in initial_approxs:
                    xs, res = newton(g, dg, x0, niter)
                    print(f"Converge en {len(xs)-1} iteraciones a {xs[-1]:.10f}")
                    plt.figure()
                    plt.plot(range(len(res)), np.log10(res), marker='o')
                    plt.title(f'|g(x_k)| log10, x0={x0}')
                    plt.xlabel('Iteración k'); plt.ylabel('log10(abs_error)')
                    plt.show()
                
                roots = find_real_roots([1, -2, -6, 12, 9, -18])
                print("Soluciones (reales) de g: ")
                for r in sorted(roots):
                    abs_err = abs(g(r))
                    rel_err = abs_err / abs(r)
                    print(f"-> {r:.6f}, error relativo: {rel_err:.6f}, error absoluto: {abs_err:.6f}")
                
                # como tenemos que las raíces no son simples el orden de convergencia pasa de cuádratico a lineal
            
            case '2':

                inital_x0 = [1, 2]
                mValues = [-26, -32, -20, 150]
                series = {}
                print("Resultados Whittaker:")

                for m in mValues:
                    print(f"==== m = {m} ====")
                    for x0 in inital_x0:
                        xs = whittaker(g, x0, m)
                        errors = np.abs(np.diff(xs))
                        series[x0] = (xs, errors)

                        if len(errors) == 0:
                            print(f"x0={x0} convergió en 0 pasos")
                            continue

                        print(f"x0 = {x0} -> solucion encontrada = {xs[-1]}, pasos = {len(xs)}, abs_error = {g(xs[-1])}, rel_error = {abs(g(xs[-1]))/abs(xs[-1])}")

                    max_steps = max(len(errs) for (_, errs) in series.values())
    
                    plt.figure()
                    for x0, (_, errs) in series.items():
                        padded = np.full(max_steps, np.nan)
                        padded[:len(errs)] = errs
                        plt.plot(
                            np.arange(1, max_steps+1),
                            np.log10(padded),
                            marker='o',
                            label=f'x0 = {x0}'
                        )
                    
                    plt.title(f'Whittaker: variación de x0 para m = {m} (log10|xₖ₊₁ - xₖ|)')
                    plt.xlabel('Iteración k')
                    plt.ylabel('log10(error)')
                    plt.legend()
                    plt.grid(True)
                    plt.show()

            case '3':
                inital_x0 = [0, 1, 3, 0]
                initial_x1 = [-1, 2, 4, 4]
                pairs = [(x, y) for x, y in zip(inital_x0, initial_x1)]

                print(f"{'x0':>6} {'x1':>6} {'   solución':>12} {' abs_error':>12} {' rel_error':>12} {' iteraciones' :>6}")
                print("-" * 66)
                
                for x0, x1 in pairs:
                    xs = secant(g, x0, x1)
                    errors = np.abs(np.diff(xs))
                    plt.plot(
                        np.arange(1, len(errors) + 1),
                        np.log10(errors + 1e-8),
                        marker='o',
                        label=f'x0={x0}, x1={x1}'
                    )

                    root = xs[-1]

                    # Calcular errores
                    abs_err = abs(g(root))
                    rel_err = abs_err / abs(root) if root != 0 else np.nan

                    # Imprimir formateado
                    print(f"{x0:6.2f} {x1:6.2f} {root:12.6f} {abs_err:12.2e} {rel_err:12.2e} {len(xs):6}")

                plt.title('Método de la secante: comparación de pares iniciales\n(log10|xₖ₊₁ − xₖ|)')
                plt.xlabel('Iteración k')
                plt.ylabel('log10(error local)')
                plt.legend()
                plt.grid(True)
                plt.show()

            case '4':
                x0 = 1.0
                h_values = [1e-1, 1e-3, 1e-6, 1e-9]

                plt.figure(figsize=(8, 6))
                for h in h_values:
                    xs, res = newton_num_deriv(g, x0, h)
                    ks = np.arange(len(res))
                    plt.plot(ks, np.log10(res + 1e-16), marker='o', label=f'h = {h}')

                plt.title('Convergencia de Newton con derivada numérica para distintos h')
                plt.xlabel('Iteración k')
                plt.ylabel('log10(|f(x_k)|)')
                plt.legend()
                plt.grid(True)
                plt.show()

            case '5':
                return
            case _:
                print("Opción no válida. Intenta de nuevo.")

if __name__ == '__main__':
    main_menu()