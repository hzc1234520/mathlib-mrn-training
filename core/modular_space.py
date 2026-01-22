import numpy as np

class ModularSpace:
    """Modular space Z_m^n with vector operations"""
    
    def __init__(self, modulus: int, dimension: int):
        self.modulus = modulus
        self.dimension = dimension
        
    def discretize(self, vector: np.ndarray) -> np.ndarray:
        """Convert continuous vector to discrete modular representation
        x_v,i = round(e_v,i) mod m
        """
        rounded = np.round(vector).astype(int)
        return rounded % self.modulus
    
    def add(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Modular vector addition: (a + b) mod m"""
        return (a + b) % self.modulus
    
    def subtract(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Modular vector subtraction: (a - b) mod m"""
        return (a - b) % self.modulus
    
    def matrix_multiply(self, A: np.ndarray, v: np.ndarray) -> np.ndarray:
        """Matrix-vector multiplication in modular space
        (Av)_i = sum_k A_ik * v_k mod m
        """
        result = np.dot(A, v).astype(int)
        return result % self.modulus
    
    def verify_bounds(self, vector: np.ndarray) -> bool:
        """Verify that all components are in [0, m-1]"""
        return np.all(vector >= 0) and np.all(vector < self.modulus)
    
    def random_matrix(self, rows: int, cols: int) -> np.ndarray:
        """Generate random matrix with values in Z_m"""
        return np.random.randint(0, self.modulus, size=(rows, cols))
    
    def zero_vector(self) -> np.ndarray:
        """Return zero vector in Z_m^n"""
        return np.zeros(self.dimension, dtype=int)
    
    def distance(self, a: np.ndarray, b: np.ndarray) -> int:
        """Hamming distance between two vectors"""
        return np.sum(a != b)