import numpy as np
import time

class Poly:
    array_size = 359
    multiplicative_matrix = np.zeros((array_size, array_size), dtype=int)

    def __init__(self, stringx=None):
        if stringx is None:
            self.data = np.zeros(Poly.array_size, dtype=int)
        else:
            self.data = np.array(stringx, dtype=int)

    @staticmethod
    def zero():
        return Poly()

    @staticmethod
    def one():
        result = Poly()
        result.data.fill(1)
        return result

    def __str__(self):
        return ''.join(str(x) for x in reversed(self.data))

    @classmethod
    def from_string(self, s):
        data = [int(i) for i in reversed(s)]
        return self(data)

    @staticmethod
    def generate_random_binary_string():
        return ''.join(str(np.random.randint(0, 2)) for _ in range(Poly.array_size - 1))

    @classmethod
    def calculate_multiplicative_matrix(self):
        p = 2 * self.array_size + 1
        array = np.zeros(self.array_size, dtype=int)
        array[0] = 1
        for i in range(1, self.array_size):
            array[i] = (array[i - 1] * 2) % p

        for i in range(self.array_size):
            for j in range(self.array_size):
                a_i = array[i]
                a_j = array[j]
                conditions = [
                    (a_i + a_j) % p == 1,
                    (a_i - a_j + p) % p == 1,
                    (p - a_i + a_j) % p == 1,
                    (p - a_i - a_j + p) % p == 1
                ]
                self.multiplicative_matrix[i, j] = 1 if any(conditions) else 0

    def __add__(self, other):
        result = Poly()
        result.data = self.data ^ other.data
        return result

    def __lshift__(self, pos):
        result = Poly()
        result.data = np.roll(self.data, pos)
        return result

    def __rshift__(self, pos):
        result = Poly()
        result.data = np.roll(self.data, -pos)
        return result

    def mirror(self):
        result = Poly()
        result.data = self.data[::-1]
        return result

    def __mul__(self, other):
        result = Poly()
        u = self.mirror()
        v = other.mirror()

        for i in range(Poly.array_size):
            temp = np.zeros(Poly.array_size, dtype=int)
            for j in range(Poly.array_size):
                temp[j] = np.sum(u.data * Poly.multiplicative_matrix[j]) % 2
            ss = np.sum(temp * v.data) % 2
            u <<= 1
            v <<= 1
            result.data[i] = ss
        result = result.square()
        return result

    def square(self):
        result = Poly()
        result.data[:-1] = self.data[1:]
        result.data[-1] = self.data[0]
        return result

    def power(self, power):
        result = Poly.one()
        base = self
        for bit in power.data:
            if bit == 1:
                result *= base
            base = base.square()
        return result

    def inverse(self):
        result = self
        temp = self
        for _ in range(Poly.array_size - 2):
            temp = temp.square()
            result *= temp
        result = result.square()
        return result

    def trace(self):
        return Poly([np.bitwise_xor.reduce(self.data)])

def measure_time(func, *args):
    start_time = time.perf_counter()
    func(*args)
    end_time = time.perf_counter()
    return end_time - start_time

if __name__ == "__main__":

    Poly.calculate_multiplicative_matrix()

    A = Poly.from_string("01001010110111111111110011110011100000001010011100000111100011110001110010011001100100110100110110010011110010111110101010011110001111001100101100001011001101111111011010100101001010010010010101110001110001101001000001011110110101000011100000101110101010101010001101010010000110111111000011100011001101100010011111000111000000010000010011101100010100001010111")
    B = Poly.from_string("11111010100111110101101111010111110100101110111110010010001100001100111111101001101000011000100111100100101011111000000101100111000011111100000011111111111110101011100111001110001111110011111111011001101100101101100010011000000011111010000100000100101101100110010100111110001111111100011011111111100010010010111011000111111100110101100010100000011110100111010")
    C = Poly.from_string("10111110100111010011010010000101100011011011111111000110000010101010000001101101111101001101100101100000110110010011011001111101100100001101000110011001101110111100101011110111011001100101101111110110000010110101101101110101100011111001010011100100000111001000100101101010100001110001000100001101110100101010001010100111110111110100100000110001001010001000010")

    print("A:      ", A, "\nB:      ", B, "\nC:      ", C)

    print("A + B:  ", A+B)
    print("A * B:  ", A*B)
    print("A^2:    ", A.square())
    print("A^C:    ", A.power(C))
    print("Trace:  ", A.trace())
    print("A^-1:   ", A.inverse())

    # print("Rand:", Poly.generate_random_binary_string())
    cpu_clock_speed = 2.0

    operations = {
        "A + B": lambda: A + B,
        "A * B": lambda: A * B,
        "A^-1": lambda: A.inverse(),
        "B^-1": lambda: B.inverse(),
        "A^2": lambda: A.square(),
        "A^B": lambda: A.power(B),
        "Tr(A)": lambda: A.trace(),
    }
    for op_name, op_func in operations.items():
        elapsed_time = measure_time(op_func)
        cpu_cycles = elapsed_time * cpu_clock_speed * 10 ** 6
        print(f"{op_name}: Time = {elapsed_time:.6f} seconds, CPU Cycles = {cpu_cycles:.0f}")



