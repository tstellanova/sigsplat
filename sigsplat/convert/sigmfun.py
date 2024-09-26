
from sigmf import sigmffile, SigMFFile

# def read_file_meta(sigfile_obj):
#     """
#         Read some commonly-used meta information from sigmf file object
#     :param sigfile_obj: SigMF object
#     :return:
#     """
#     sample_rate_hz = int(sigfile_obj.get_global_field(SigMFFile.SAMPLE_RATE_KEY))
#     print(f'sample_rate_hz: {sample_rate_hz}')
#     sample_size_bytes = sigfile_obj.get_sample_size()
#     print(f'sample size (bytes): {sample_size_bytes}')
#
#     center_freq_hz = sigfile_obj.get_capture_info(0)[SigMFFile.FREQUENCY_KEY]
#     half_sampling_rate_hz = sample_rate_hz // 2
#     freq_lower_edge_hz = center_freq_hz - half_sampling_rate_hz
#     freq_upper_edge_hz = center_freq_hz + half_sampling_rate_hz
#
#     total_samples_guess = sample_rate_hz
#
#     first_sample_annotations = sigfile_obj.get_annotations(0)
#     for annotation in first_sample_annotations:
#         if annotation[SigMFFile.LENGTH_INDEX_KEY] is not None:
#             total_samples_guess = int(annotation[SigMFFile.LENGTH_INDEX_KEY])
#         if annotation[SigMFFile.FLO_KEY] is not None:
#             freq_lower_edge_hz = int(annotation[SigMFFile.FLO_KEY])
#         if annotation[SigMFFile.FHI_KEY] is not None:
#             freq_upper_edge_hz = int(annotation[SigMFFile.FHI_KEY])
#
#     return center_freq_hz, sample_rate_hz, freq_lower_edge_hz, freq_upper_edge_hz, total_samples_guess


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
