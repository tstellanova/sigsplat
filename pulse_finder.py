import numpy as np
#!/usr/bin/env python
"""
Experiment with searching for Linear Frequency Modulated "chirps" within a signal file
"""
import argparse
import os
import numpy as np
from sigmf import sigmffile, SigMFFile
from scipy.signal import welch

def read_file_meta(sigfile_obj):
    '''
    Read some commonly-used meta information from sigmf file object
    :param sigfile_obj: SigMF object
    :return:
    '''
    sample_rate_hz = int(sigfile_obj.get_global_field(SigMFFile.SAMPLE_RATE_KEY))
    print(f'sample_rate_hz: {sample_rate_hz}')
    sample_size_bytes = sigfile_obj.get_sample_size()
    print(f'sample size (bytes): {sample_size_bytes}')

    center_freq_hz = sigfile_obj.get_capture_info(0)[SigMFFile.FREQUENCY_KEY]
    half_sampling_rate_hz = sample_rate_hz // 2
    freq_lower_edge_hz = center_freq_hz - half_sampling_rate_hz
    freq_upper_edge_hz = center_freq_hz + half_sampling_rate_hz

    total_samples_guess = sample_rate_hz

    focus_label = None
    first_sample_annotations = sigfile_obj.get_annotations(0)
    for annotation in first_sample_annotations:
        if annotation[SigMFFile.LENGTH_INDEX_KEY] is not None:
            total_samples_guess = int(annotation[SigMFFile.LENGTH_INDEX_KEY])
        if annotation[SigMFFile.FLO_KEY] is not None:
            freq_lower_edge_hz = int(annotation[SigMFFile.FLO_KEY])
        if annotation[SigMFFile.FHI_KEY] is not None:
            freq_upper_edge_hz = int(annotation[SigMFFile.FHI_KEY])
        if annotation[SigMFFile.LABEL_KEY] is not None:
            focus_label = annotation[SigMFFile.LABEL_KEY]

    return center_freq_hz, sample_rate_hz, freq_lower_edge_hz, freq_upper_edge_hz, total_samples_guess, focus_label

import numpy as np
from scipy.signal import welch

def measure_power_threshold(file_object, chunk_size=1024, total_chunks=3, percentile=1, method="fft"):
    total_powers = []
    total_samples_read = 0

    n_chunks_read = 0
    while n_chunks_read < total_chunks:
        # Read the next chunk of samples
        chunk = file_object.read_samples(total_samples_read, chunk_size)
        num_samples = len(chunk)

        if num_samples < chunk_size:
            break  # End of file reached
        total_samples_read += num_samples

        # Compute power using FFT or PSD
        if method == "fft":
            # FFT-based power computation
            fft_result = np.fft.fft(chunk)
            power_chunk = np.sum(np.abs(fft_result) ** 2) / num_samples
        elif method == "psd":
            # PSD-based power computation using Welch method
            freqs, psd = welch(chunk, nperseg=num_samples)
            power_chunk = np.sum(psd)
        else:
            raise ValueError("Unsupported method. Choose 'fft' or 'psd'.")

        n_chunks_read += 1
        total_powers.append(power_chunk)

    # Convert list of total powers to a numpy array
    all_powers = np.array(total_powers)

    # Determine the threshold as a specified percentile of the total power distribution
    all_powers_db = 10 * np.log10(all_powers)
    print(f"all powers min: {np.min(all_powers_db)} med: {np.median(all_powers_db)} max: {np.max(all_powers_db)}")
    power_threshold_db = 2 * np.min(all_powers_db)
    # power_threshold_db = np.percentile(all_powers_db, percentile)
    print(f"power_threshold_db: {power_threshold_db}")
    return power_threshold_db

def measure_pulse_durations(file_object, power_threshold, chunk_size=1024, total_chunks=3,  method="fft"):
    pulse_durations = []
    in_pulse = False
    pulse_start = None
    total_samples_read = 0

    n_chunks_read = 0
    while n_chunks_read < total_chunks:
        # Read the next chunk of samples
        chunk = file_object.read_samples(total_samples_read, chunk_size)
        num_samples = len(chunk)

        if num_samples < chunk_size:
            break  # End of file reached
        total_samples_read += num_samples

        # Compute power using FFT or PSD
        if method == "fft":
            fft_result = np.fft.fft(chunk)
            power_chunk = np.sum(np.abs(fft_result) ** 2) / num_samples
        elif method == "psd":
            freqs, psd = welch(chunk, nperseg=num_samples)
            power_chunk = np.sum(psd)
        else:
            raise ValueError("Unsupported method. Choose 'fft' or 'psd'.")

        if power_chunk <= 0:
            power_chunk = 1E-20
        power_chunk_db = 10 * np.log10(power_chunk)

        # Detect pulses based on the power threshold
        if power_chunk_db > power_threshold:
            if not in_pulse:
                # Detected the start of a pulse
                print(f"start power_chunk_db: {power_chunk_db}")
                in_pulse = True
                pulse_start = total_samples_read - num_samples
            else:
                print(f"continue pulse {power_chunk_db}")
        else:
            print(f"power_chunk_db: {power_chunk_db} < {power_threshold}")
            if in_pulse:
                # Detected the end of a pulse
                print(f"end power_chunk_db: {power_chunk_db}")
                in_pulse = False
                pulse_end = total_samples_read - num_samples
                pulse_duration = pulse_end - pulse_start
                pulse_durations.append(pulse_duration)

        n_chunks_read += 1

        # Handle the case where a pulse extends across chunk boundaries
        if in_pulse and n_chunks_read == total_chunks:
            # Assume the last pulse is still ongoing but we've reached the end of the file
            pulse_end = total_samples_read
            pulse_duration = pulse_end - pulse_start
            pulse_durations.append(pulse_duration)
            break  # End of file reached


    if pulse_durations:
        min_duration = min(pulse_durations)
        med_duration = np.median(pulse_duration)
        max_duration = max(pulse_durations)
    else:
        min_duration, med_duration, max_duration = None, None, None

    return min_duration, med_duration, max_duration



def main():
    parser = argparse.ArgumentParser(description='Analyze SIGMF file for linear frequency modulated (LFM) \"chirps\" ')
    parser.add_argument('src_meta',
                        help="Source sigmf file name eg 'test_sigmf_file.sigmf-meta' with the 'sigmf-meta' file extensions")

    # parser.add_argument('--pulse_rep_freq_min','-prf_min',dest='prf_min', type=float, default=1E3,
    #                     help="Minimum pulse repetition frequency (Hz)")
    # parser.add_argument('--pulse_rep_freq_max','-prf_max',dest='prf_max', type=float, default=7.6E3,
    #                     help="Maximum pulse repetition frequency (Hz)")
    # # parser.add_argument('--chirp_dur','-cd',dest='chirp_period_sec', type=float,default=73E-6,
    # #                     help="Estimate of the duration of each chirp, in seconds")
    # parser.add_argument("--min_search_freq","-minf",type=float,default=1.0,required=False,
    #                     help="Minimum chirp frequency to search, in Hz")
    # parser.add_argument("--max_search_freq","-maxf",type=float,default=1.0,required=False,
    #                     help="Maximum chirp frequency to search, in Hz")

    parser.add_argument("--sample_start_time","-st", dest='sample_start_time',type=float,default=0.0,required=False,
                        help="Start time for valid samples (seconds)")

    args = parser.parse_args()
    # prf_min_hz = args.prf_min
    # chirp_period_sec = 1/prf_min_hz
    # print(f"max chirp period: {chirp_period_sec}")

    # chirp_period_sec = args.chirp_period_sec
    base_data_name = args.src_meta
    print(f'loading file info: {base_data_name}')
    sigfile_obj = sigmffile.fromfile(base_data_name)

    center_freq_hz, sample_rate_hz, freq_lower_edge_hz, freq_upper_edge_hz, total_samples, focus_label \
        = read_file_meta(sigfile_obj)

    print(f" center_freq_hz, sample_rate_hz, freq_lower_edge_hz, freq_upper_edge_hz, total_samples:\n ",
          center_freq_hz, sample_rate_hz, freq_lower_edge_hz, freq_upper_edge_hz, total_samples)

    # 204 us / 73 us =

    power_chunk_size = 8192
    power_num_chunks = int(total_samples / power_chunk_size)
    print(f"power_num_chunks: {power_num_chunks}")

    power_thresh = measure_power_threshold(
        sigfile_obj, power_chunk_size,total_chunks=power_num_chunks)
    print(f"power_thresh: {power_thresh} dB")
    if power_thresh > -50.0:
        power_thresh = -50.0
        print(f"power_thresh: {power_thresh} dB")

    # pulse_check_chunk_size = int(power_chunk_size / 32 )
    pulse_check_chunk_size = power_chunk_size//32
    pulse_num_chunks = int(total_samples / pulse_check_chunk_size)
    print(f"pulse_num_chunks: {pulse_num_chunks}")
    min_duration, med_duration, max_duration = measure_pulse_durations(
        sigfile_obj, power_thresh, total_chunks=pulse_num_chunks, chunk_size=pulse_check_chunk_size, method="fft")

    print(f"min_duration: {min_duration} med_duration: {med_duration} max_duration: {max_duration}")

    pulse_duration = med_duration / sample_rate_hz
    print(f"pulse_duration: {pulse_duration}")



if __name__ == "__main__":
    main()
