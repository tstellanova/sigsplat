import numpy as np

class MSeqGeneratorLSFR:
    """
    Class that provides m-sequence with a memory-efficient method,
    for a predefined set of sequence lengths N
    Example:
        ```
        n = 1023  # Change this to 127, 255, 511, or 1023
        lfsr = MSeqGeneratorLSFR(n)
        sequence = lfsr.generate(n)
        print(f"Sequence for n={n}: {len(sequence)}")
        ```
    """
    def __init__(self, n):
        self.n = n
        self.m = int(np.log2(n + 1))  # Number of stages (m)

        # Define taps and seeds for the given lengths
        self.taps_dict = {
            127: [6, 5, 4, 0],       # Polynomial x^7 + x^6 + x^5 + x^4 + 1
            255: [7, 5, 4, 3],       # Polynomial x^8 + x^6 + x^5 + x^4 + 1
            511: [8, 7, 4, 3],       # Polynomial x^9 + x^8 + x^5 + x^4 + 1
            1023: [10, 9, 8, 7]      # Polynomial x^10 + x^9 + x^7 + x^6 + 1
        }

        self.seed_dict = {
            127: np.array([1] * 7 + [0], dtype=int),
            255: np.array([1] * 8 + [0], dtype=int),
            511: np.array([1] * 9 + [0], dtype=int),
            1023: np.array([1] * 10 + [0], dtype=int)
        }

        # Get the taps and seed for the specified n
        if n not in self.taps_dict:
            raise ValueError("Unsupported value of n. Supported values are 127, 255, 511, and 1023.")

        self.taps = self.taps_dict[n]
        self.register = self.seed_dict[n]

    def _feedback(self):
        feedback = 0
        for tap in self.taps:
            feedback ^= self.register[tap]
        return feedback

    def next(self):
        """Generate the next value in the sequence."""
        # Output the current value (last bit of the register)
        output = self.register[-1]

        # Compute the feedback bit
        feedback = self._feedback()

        # Shift the register and insert the feedback bit at the start
        self.register = np.roll(self.register, shift=1)
        self.register[0] = feedback

        return output

    def generate(self, length):
        """Generate a sequence of given length using a generator."""
        if length != self.n:
            raise ValueError(f"Length must be {self.n}.")
        for _ in range(length):
            yield self.next()

# # Example usage:
# n = 127  # Change this to 127, 255, 511, or 1023
# lfsr = MSeqGeneratorLSFR(n)
# # Generate the sequence
# sequence = lfsr.generate(n)
# print(f"Sequence for n={n}: {len(sequence)}")
#
# n = 255  # Change this to 127, 255, 511, or 1023
# lfsr = MSeqGeneratorLSFR(n)
# sequence = lfsr.generate(n)
# print(f"Sequence for n={n}: {len(sequence)}")
#
# n = 511  # Change this to 127, 255, 511, or 1023
# lfsr = MSeqGeneratorLSFR(n)
# sequence = lfsr.generate(n)
# print(f"Sequence for n={n}: {len(sequence)}")
#
# n = 1023  # Change this to 127, 255, 511, or 1023
# lfsr = MSeqGeneratorLSFR(n)
# sequence = lfsr.generate(n)
# print(f"Sequence for n={n}: {len(sequence)}")
