import math
import numpy as np

# =====================================================
# M06 - Demonstrasi Pemecahan Sistem Persamaan Nonlinear
# Nama  : Fayyadh Muhammad Habibibie
# NIM   : 21120123120040
# NIMx  : 40 mod 4 = 0 → Kombinasi g1A dan g2A
# =====================================================

TOLERANSI = 1e-6
MAKS_ITERASI = 50

# ----------------------------------------
# FUNGSI ITERASI (untuk NIMx = 0)
# ----------------------------------------

# Berdasarkan kombinasi g1A dan g2A dari modul M06
# g1A: x = sqrt((10 - y) / 4)
# g2A: y = sqrt((57 - 3x) / 5)

def g1A(x, y):
    val = (10 - y) / 4
    if val <= 0:
        val = abs(val) * 0.1
    return math.sqrt(val)

def g2A(x, y):
    val = (57 - 3 * x) / 5
    if val <= 0:
        val = abs(val) * 0.1
    return math.sqrt(val)

# ----------------------------------------
# Iterasi Jacobi
# ----------------------------------------
def jacobi(g1, g2, x0, y0, tol=TOLERANSI):
    print("\n=== METODE ITERASI TITIK TETAP (JACOBI) ===")
    for i in range(MAKS_ITERASI):
        x1 = g1(x0, y0)
        y1 = g2(x0, y0)
        dx, dy = abs(x1 - x0), abs(y1 - y0)
        print(f"Iterasi {i+1:2d}: x={x1:.6f}, y={y1:.6f}, Δx={dx:.6f}, Δy={dy:.6f}")
        if dx < tol and dy < tol:
            print(f"Konvergen pada iterasi ke-{i+1}\n")
            return x1, y1
        x0, y0 = x1, y1
    print("Tidak konvergen!\n")
    return x1, y1

# ----------------------------------------
# Iterasi Gauss-Seidel
# ----------------------------------------
def seidel(g1, g2, x0, y0, tol=TOLERANSI):
    print("\n=== METODE ITERASI TITIK TETAP (GAUSS-SEIDEL) ===")
    for i in range(MAKS_ITERASI):
        x1 = g1(x0, y0)
        y1 = g2(x1, y0)
        dx, dy = abs(x1 - x0), abs(y1 - y0)
        print(f"Iterasi {i+1:2d}: x={x1:.6f}, y={y1:.6f}, Δx={dx:.6f}, Δy={dy:.6f}")
        if dx < tol and dy < tol:
            print(f"Konvergen pada iterasi ke-{i+1}\n")
            return x1, y1
        x0, y0 = x1, y1
    print("Tidak konvergen!\n")
    return x1, y1

# ----------------------------------------
# Newton-Raphson (sistem 2 variabel)
# ----------------------------------------
def f1(x, y): return x**2 + x*y - 10
def f2(x, y): return y + 3*x*(y**2) - 57
def df1dx(x, y): return 2*x + y
def df1dy(x, y): return x
def df2dx(x, y): return 3*y**2
def df2dy(x, y): return 1 + 6*x*y

def newton_raphson(x0, y0, tol=TOLERANSI):
    print("\n=== METODE NEWTON-RAPHSON ===")
    for i in range(MAKS_ITERASI):
        J = np.array([[df1dx(x0, y0), df1dy(x0, y0)],
                      [df2dx(x0, y0), df2dy(x0, y0)]])
        F = np.array([-f1(x0, y0), -f2(x0, y0)])
        delta = np.linalg.solve(J, F)
        x1, y1 = x0 + delta[0], y0 + delta[1]
        dx, dy = abs(x1 - x0), abs(y1 - y0)
        print(f"Iterasi {i+1:2d}: x={x1:.6f}, y={y1:.6f}, Δx={dx:.6f}, Δy={dy:.6f}")
        if dx < tol and dy < tol:
            print(f"Konvergen pada iterasi ke-{i+1}\n")
            return x1, y1
        x0, y0 = x1, y1
    print("Tidak konvergen!\n")
    return x1, y1

# ----------------------------------------
# Metode Secant (fungsi 1 variabel)
# ----------------------------------------
def f(x): return x**3 - 2*x**2 + 3*x - 5

def secant(x0, x1, tol=TOLERANSI):
    print("\n=== METODE SECANT ===")
    for i in range(MAKS_ITERASI):
        f0, f1 = f(x0), f(x1)
        if f1 - f0 == 0:
            print("Error: Pembagi mendekati nol.")
            break
        x2 = x1 - f1 * (x1 - x0) / (f1 - f0)
        dx = abs(x2 - x1)
        print(f"Iterasi {i+1:2d}: x={x2:.6f}, Δx={dx:.6f}")
        if dx < tol:
            print(f"Konvergen pada iterasi ke-{i+1}\n")
            return x2
        x0, x1 = x1, x2
    print("Tidak konvergen!\n")
    return x2

# ----------------------------------------
# MAIN PROGRAM
# ----------------------------------------
if __name__ == "__main__":
    print("===============================================")
    print(" DEMONSTRASI METODE NUMERIK - M06 (NIMx = 0)  ")
    print("===============================================")

    # Tebakan awal
    x0, y0 = 1.0, 1.0

    # Jalankan semua metode
    x_j, y_j = jacobi(g1A, g2A, x0, y0)
    x_s, y_s = seidel(g1A, g2A, x0, y0)
    x_n, y_n = newton_raphson(1.0, 1.0)
    akar_secant = secant(1.0, 2.0)

    print("\n=== HASIL AKHIR ===")
    print(f"Jacobi       → x={x_j:.6f}, y={y_j:.6f}")
    print(f"Gauss-Seidel → x={x_s:.6f}, y={y_s:.6f}")
    print(f"Newton-Raphson → x={x_n:.6f}, y={y_n:.6f}")
    print(f"Secant       → akar = {akar_secant:.6f}")
