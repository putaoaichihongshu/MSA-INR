# target_curves.py
import numpy as np

def sin_200x(x): return np.sin(200 * x)
def sin_500x(x): return np.sin(500 * x)
def sin_1000x(x): return np.sin(1000 * x)
def multi_scale(x):
    part1 = 0.5 * np.sin(2 * x)
    part2 = np.sin(100 * x + 1.5) * np.cos(5 * x - 0.8)
    part3 = np.cos(200 * x) * np.cos(x)
    part4 = np.sin(300 * x + 0.3) * np.sin(5 * x)
    return part1 + part2 + part3 + part4

TARGET_DICT = {
    'sin_200x': sin_200x,
    'sin_500x': sin_500x,
    'sin_1000x': sin_1000x,
    'multi_scale': multi_scale,
}
