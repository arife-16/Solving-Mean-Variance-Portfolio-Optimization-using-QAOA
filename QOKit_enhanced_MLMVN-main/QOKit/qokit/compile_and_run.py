
import os
import subprocess
import sys

# Установка путей
base_path = "/content/drive/MyDrive/AizenbergSphere/qoMLMVN/3/JPMC_PO_Benchmark/QOKit/qokit"
cython_path = os.path.join(base_path, "Cpy")
sys.path.append(base_path)

# Установка Cython, если не установлен
subprocess.run([sys.executable, "-m", "pip", "install", "cython", "numpy"])

# Компиляция Cython
os.chdir(cython_path)
subprocess.run([sys.executable, "setup.py", "build_ext", "--inplace"], check=True)

# Проверка импорта
try:
    from Cpy.qaoa_maxcut_energy import compute_maxcut_energy_cython
    print("Cython module compiled and imported successfully!")
except ImportError as e:
    print(f"Failed to import Cython module: {e}")

# Пример запуска бенчмарка
from qokit.qaoa_objective_maxcut import get_qaoa_maxcut_objective
import networkx as nx

G = nx.random_regular_graph(3, 12)  # Пример графа
N, p = 12, 4
objective = get_qaoa_maxcut_objective(N, p, G, simulator="c")
print("QAOA objective initialized successfully!")
