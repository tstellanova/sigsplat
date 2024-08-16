
import numpy as np
from scipy.signal import chirp, correlate
import matplotlib.pyplot as plt

# Function to generate a chirp and perform cross-correlation
def search_chirp(iq_signal, fs, f0_range, f1_range, t_range, plot_result=False):
    """
    Searches for a chirp signal in a given IQ signal using cross-correlation.

    Parameters:
    - iq_signal: The baseband RF I/Q signal (complex-valued numpy array).
    - fs: Sampling frequency of the IQ signal.
    - f0_range: Tuple (min_f0, max_f0) specifying the range of possible start frequencies.
    - f1_range: Tuple (min_f1, max_f1) specifying the range of possible end frequencies.
    - t_range: Tuple (min_t, max_t) specifying the range of possible durations.
    - plot_result: If True, plot the correlation result.

    Returns:
    - best_params: Dictionary with the best found chirp parameters.
    """
    max_corr = 0
    best_params = {}

    for f0 in np.linspace(f0_range[0], f0_range[1], 10):  # Adjust step size for finer search
        for f1 in np.linspace(f1_range[0], f1_range[1], 10):
            for t in np.linspace(t_range[0], t_range[1], 10):
                t_samples = int(t * fs)
                time = np.linspace(0, t, t_samples)

                # Generate the reference chirp signal
                reference_chirp = chirp(time, f0=f0, f1=f1, t1=t, method='linear')

                # Cross-correlate with the I/Q signal
                correlation = correlate(iq_signal, reference_chirp, mode='valid')
                corr_max = np.max(np.abs(correlation))

                # Keep track of the best correlation
                if corr_max > max_corr:
                    max_corr = corr_max
                    best_params = {'f0': f0, 'f1': f1, 'duration': t, 'max_corr': max_corr}

                    if plot_result:
                        plt.figure()
                        plt.plot(np.abs(correlation))
                        plt.title(f"Cross-correlation (f0={f0}, f1={f1}, duration={t})")
                        plt.xlabel("Sample Index")
                        plt.ylabel("Correlation Magnitude")
                        plt.show()

    return best_params

# Example usage:

# Simulate a sample IQ signal with a chirp (replace with actual data)
fs = 1e6  # Sampling frequency in Hz
duration = 0.01  # Duration of the chirp in seconds
f0_true = 100e3  # Start frequency in Hz
f1_true = 200e3  # End frequency in Hz

# Generate the true chirp signal
time = np.linspace(0, duration, int(fs * duration))
true_chirp_signal = chirp(time, f0=f0_true, f1=f1_true, t1=duration, method='linear')

# Add the chirp signal to a noisy IQ signal
noise = 0.1 * np.random.randn(len(time))
iq_signal = true_chirp_signal + noise

# Define the search ranges
f0_range = (90e3, 110e3)  # Start frequency range in Hz
f1_range = (190e3, 210e3)  # End frequency range in Hz
t_range = (0.009, 0.011)   # Duration range in seconds

# Search for the chirp
best_chirp = search_chirp(iq_signal, fs, f0_range, f1_range, t_range, plot_result=True)

print("Best chirp parameters found:")
print(best_chirp)
