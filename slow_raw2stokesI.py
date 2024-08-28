import numpy as np

def calculate_stokes_I(samples, sampling_rate, fft_length, spectra_to_sum):
    """
    Calculate the Stokes I total power from RF I/Q samples.

    Parameters:
    - samples: A numpy array of complex I/Q samples.
    - sampling_rate: Sampling rate in samples per second (e.g., 20 Msps).
    - fft_length: The length of the FFT to use for calculating spectra.
    - spectra_to_sum: The number of spectra to sum together into a single integration point.

    Returns:
    - stokes_I: The total power calculated from the summed spectra.
    """
    # Number of samples to process for each integration point
    samples_per_integration = fft_length * spectra_to_sum

    # Calculate the total number of integration points
    num_integration_points = len(samples) // samples_per_integration

    stokes_I = np.zeros(num_integration_points)

    for i in range(num_integration_points):
        # Extract the samples for the current integration point
        start_idx = i * samples_per_integration
        end_idx = start_idx + samples_per_integration
        integration_samples = samples[start_idx:end_idx]

        # Calculate the power spectrum for each FFT segment
        power_spectra = np.zeros(fft_length)

        for j in range(spectra_to_sum):
            segment_start = j * fft_length
            segment_end = segment_start + fft_length

            segment = integration_samples[segment_start:segment_end]
            spectrum = np.fft.fft(segment, n=fft_length)
            power_spectra += np.abs(spectrum)**2

        # Average the power spectra to get the Stokes I value for this integration point
        stokes_I[i] = np.mean(power_spectra) / fft_length

    return stokes_I

# Example usage
sampling_rate = 20e6  # 20 Msps
fft_length = 1024  # Length of FFT
spectra_to_sum = 10  # Number of spectra to sum together

# Generate example I/Q samples (replace this with actual data)
samples = np.random.randn(100000) + 1j * np.random.randn(100000)

# Calculate the Stokes I total power
stokes_I = calculate_stokes_I(samples, sampling_rate, fft_length, spectra_to_sum)

print(stokes_I)
