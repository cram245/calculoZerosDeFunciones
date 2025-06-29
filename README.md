# Nonlinear Root-Finding Methods

This repository contains Python implementations of several classical root-finding algorithms (Newton’s method, secant method, Whittaker’s method, and Newton with numerical derivative), along with scripts to visualize and compare their convergence behaviors on sample polynomials.

## Features

- **Newton’s Method**  
  - Exact derivative version  
  - Numerical derivative approximation  
- **Secant Method**  
- **Whittaker’s Fixed-Step Method**  
- Convergence plots and error analysis  
- Interactive menu script for quick experimentation  

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/cram245/calculoZerosDeFunciones.git
   cd calculoZerosDeFunciones
     cd nonlinear-root-finding
   ```

2. (Optional) Create and activate a virtual environment:
   ```bash
   python3 -m venv venv
   source venv/bin/activate      # macOS/Linux
   venv\Scripts\activate.bat     # Windows
   ```

3. Install dependencies:
   ```bash
   pip install numpy matplotlib sympy
   ```

## Usage

**Important:** To launch the interactive menu, run the `menu_raices.py` file:
```bash
python menu_raices.py
```

### 1. Interactive Menu

Launch the menu to select and run examples:
```bash
python menu_raices.py
```

### 2. Standalone Scripts

- **Plot Newton with numerical derivative**  
  ```bash
  python newton_num_deriv_compare.py
  ```
- **Compare Whittaker for different initial guesses**  
  ```bash
  python comparar_whittaker_x0.py
  ```
- **Compare Secant method**  
  ```bash
  python comparar_secante.py
  ```

### 3. Import as Module

You can also import functions in your own scripts:
```python
from arrels_no_lineals import newton, secant, whittaker, newton_num_deriv

# Example:
root, residuals = newton(lambda x: x**3 - 2*x - 5,
                        lambda x: 3*x**2 - 2,
                        x0=2.0)
print("Root:", root[-1])
```

## Code Structure

```
.
├── arrels_no_lineals.py      # Core implementations
├── menu_raices.py             # Interactive command-line menu (execute this file)
└── README.md                  # This file
```

## Polynomial Examples

- **Example 1**:  
  \(f_1(x) = x^5 - 4x^4 + 7x^3 - 21x^2 + 6x + 18\)
- **Example 2**:  
  \(f_2(x) = x^5 - 2x^4 - 6x^3 + 12x^2 + 9x - 18\)

## Convergence Analysis

- **Quadratic vs. Linear** convergence:  
  - Newton’s method on simple roots → quadratic  
  - Newton with numerical derivative or on multiple roots → linear convergence factor
- **Basins of attraction** visualized via interactive scripts.

## License

This project is released under the MIT License. See [LICENSE](LICENSE) for details.
