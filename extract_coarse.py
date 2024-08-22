"""
From a GUPPI .raw file (or set of files),
extract a single coarse channel's worth of data
into two separate polarization SigMF file sets
including both sigmf-data and sigmf-meta files
for each polarization.

Based on prior raw2sigmf from https://github.com/greghell/extractor

first argument = filename of one '.raw' file of the data set

second argument = frequency within coarse channel to extract
"""
import glob
import os
import sys
import math
import numpy as np
import argparse
import re
import json
import sigmf
from sigmf import SigMFFile
import datetime
import dateutil.parser
from copy import deepcopy

def extract_guppi_header_key_val(string):
    match = re.search(r'([^=\s]+)\s*=\s*(\S.*)', string)
    if match:
        return match.group(1), match.group(2).strip()
    return None


parser = argparse.ArgumentParser(description=
                                 """ From a GUPPI .raw file (or set of files), 
                                 extract a single coarse channel's worth of data 
                                 into two separate polarization SigMF file sets 
                                 including both sigmf-data and sigmf-meta files 
                                 for each polarization. """)
parser.add_argument('in_fname',
                    help='GUPPI RAW file name. This will be generalized to the set of all files with similar names')
parser.add_argument("--ctr_freq", '-fc', dest='in_ctr_freq', type=float, default=8419.3,
                    help='Frequency within a coarse channel to extract')

args = parser.parse_args()
fileinit = args.in_fname
fFreq = args.in_ctr_freq

# Take input file name and generalize to all raw files with similar names
if (fileinit[-4:] == '.raw'):
    fileinit = fileinit.replace(fileinit[len(fileinit)-8:],"*.raw")
    # print("fileinit after: ", fileinit)
else:
    sys.exit('Missing raw file name')


all_filenames = sorted(glob.glob(fileinit))
print("all_filenames: ",  all_filenames)

telescope = 'Unknown Telescope'
frontend = ''
source = 'Unknown Source'
observer = 'Unknown Observer'

fname = all_filenames[0]
fread = open(fname,'rb')        # open first file of data set
currline = fread.read(80).decode('ascii')          # reads first line
line_key_val = extract_guppi_header_key_val(currline)
nHeaderLines = 1
headr = []
print("Processing header...")
while line_key_val :           # until reaching end of header
    print("currline ", currline )
    print("line_key_val ", line_key_val )
    match line_key_val[0]:
        case 'PKTSIZE': # read packet size
            nPktSize = float(line_key_val[1])
            print("nPktSize ", nPktSize)
        case 'OBSFREQ': # read cenral frequency
            cenfreq = float(line_key_val[1])
            print("cenfreq ", cenfreq)
        case 'OBSBW': # read bandwidth
            obsbw = float(line_key_val[1])
        case 'OBSNCHAN': # read number of coarse channels
            obsnchan = float(line_key_val[1])
        case 'DIRECTIO': # read directio flag
            ndirectio = float(line_key_val[1])
        case 'BLOCSIZE': # read block size
            nblocsize = int(line_key_val[1])
        case 'NBITS': # read bit-depth
            nbits = int(line_key_val[1])
        case 'DAQPULSE': # read time and date of observation
            # TODO validate against ISO 8601 or other time format?
            dt_str = str(line_key_val[1])
            # example  : 'Wed Dec 30 15:45:27 2015'
            best_datetime = dateutil.parser.parse(dt_str)
            timedate = best_datetime.isoformat()+'Z'
        #timedate = str(line_key_val[1])
        case 'SRC_NAME':        # read name of tracked source
            source = str(line_key_val[1])
        case 'TELESCOP': # read name of instrument
            telescope = str(line_key_val[1])
        case 'FRONTEND': # read RF frontend
            frontend = str(line_key_val[1])
        case 'BACKEND':
            backend = str(line_key_val[1])
        case 'OBSERVER': # read name of observer
            observer = str(line_key_val[1])
        case 'CHAN_BW':
            chan_bw = float(line_key_val[1])
        case 'TBIN':
            time_resolution_sec = float(line_key_val[1])
        case 'PROJID':
            unused_project_id = str(line_key_val[1])
        case 'POL_TYPE': # read polarization type
            # "Set to AA+BB for Fast4k mode,
            # IQUV for all other incoherent modes,
            # and AABBCRCI for all coherent modes."
            unused_pol_type = str(line_key_val[1])
        case 'NPOL':
            # "Number of polarization states coming from HW.
            # Should be 1 for Fast4k mode, and 4 for all other modes."
            # Note this is not the actual number of polarizations in the data.
            unused_npol = int(line_key_val[1])
        case 'NRCVR':
            # "Number of receiver channels (polarizations)"
            # We later verify that this matches our expectation of two polarizations
            nrcvr_chan = int(line_key_val[1])

    headr.append(currline)
    nHeaderLines = nHeaderLines + 1         # counts number of lines in header
    currline = fread.read(80).decode('ascii')  # read new header line
    line_key_val = extract_guppi_header_key_val(currline)

print("nHeaderLines: " , nHeaderLines)

fread.close()


if nrcvr_chan != 2:
    sys.exit('NRCVR must be 2 to process two polarizations')

nChanSize = int(nblocsize / obsnchan)   # size of 1 channel per block in bytes
nPadd = 0
if ndirectio == 1:
    nPadd = int((math.floor(80.*nHeaderLines/512.)+1)*512 - 80*nHeaderLines)    # calculate directIO padding
statinfo = os.stat(fname)
NumBlocs = int(round(statinfo.st_size / (nblocsize + nPadd + 80*nHeaderLines)))

# frequency coverage of dataset
fLow = cenfreq - obsbw/2.
fHigh = cenfreq + obsbw/2.
dChanBW = obsbw/obsnchan

# validate some values reported in the header
if chan_bw:
    if not math.isclose(dChanBW, chan_bw, rel_tol=1e-9):
        print(f'WARNING: calculated channel bandwidth: {dChanBW:0.9f} , reported: {chan_bw:0.9f}')
if time_resolution_sec:
    calc_freq_resolution = 1 / time_resolution_sec
    chan_bw_hz = chan_bw * 1E6
    print(f'time res: {time_resolution_sec:0.3E} , calc freq res: {calc_freq_resolution:0.3f}, chan_bw: {chan_bw_hz:0.3f}')


if fFreq is None:
    # if user doesn't care, extract the center channel
    fFreq = (fHigh + fLow) / 2

# make sure user asked for frequency covered by file
if fFreq < min(fLow,fHigh) or fFreq > max(fLow,fHigh):
    print("Frequency bandwidth = ["+str(min(fLow,fHigh))+","+str(max(fLow,fHigh))+"]")
    sys.exit('Unable to extract requested frequency from input file')

# calculate what channel number to extract
if obsbw > 0:
    nChanOI = int((fFreq-fLow)/dChanBW)
else:
    nChanOI = int((fLow-fFreq)/abs(dChanBW))

print(f"obsbw: {obsbw} fLow: {fLow} fHigh: {fHigh} fFreq: {fFreq} dChanBW: {dChanBW}")

BlocksPerFile = np.zeros(len(all_filenames))
idx = 0
for fname in all_filenames:
    statinfo = os.stat(fname)
    BlocksPerFile[idx] = int(statinfo.st_size / (nblocsize + nPadd + 80*nHeaderLines))
    idx = idx+1

NumBlockTotal = int(sum(BlocksPerFile))
NewTotBlocSize = int(NumBlockTotal * nChanSize)

# extracted channel frequency coverage
fLowChan = fLow + (nChanOI)*dChanBW
fHighChan = fLowChan + dChanBW
TotCenFreq = (fLowChan+fHighChan)/2.
print('extracting channel #' + str(nChanOI))
print('frequency coverage = ' + str(min(fLowChan,fHighChan))+ ' - ' + str(max(fLowChan,fHighChan)) + ' MHz')

fn_prefix = fname.split('\\')[-1][:-14]
fn_prefix_pol1 = f"{fn_prefix}_ch{nChanOI:0>3}_pol1"
ofname_meta1 = f"{fn_prefix_pol1}.sigmf-meta"
ofname_data1 = f"{fn_prefix_pol1}.sigmf-data"
fn_prefix_pol2 = f"{fn_prefix}_ch{nChanOI:0>3}_pol2"
ofname_meta2 = f"{fn_prefix_pol2}.sigmf-meta"
ofname_data2 = f"{fn_prefix_pol1}.sigmf-data"

# extract and write data to file
output_file1 = open(ofname_data1,'wb')
output_file2 = open(ofname_data2,'wb')
idx = -1
n_total_samples = 0
for fname in all_filenames:
    idx = idx+1
    fread = open(fname,'rb')
    print("extract data from "+fname.split('\\')[-1])
    for nblock in range(int(BlocksPerFile[idx])):
        fread.seek(int(nblock*(nHeaderLines*80+nPadd+nblocsize)+nHeaderLines*80+nPadd+nChanOI*nChanSize))
        tmpdata = np.fromfile(fread, dtype=np.int8, count=int(nChanSize))
        cur_data_len = int(nChanSize/2)
        tmpdata = np.reshape(tmpdata,(2,cur_data_len),order='F')
        # normalize the data ?
        data_range = tmpdata.max() - tmpdata.min()
        tmpdata = (tmpdata - tmpdata.min()) / data_range

        pol1_data = np.reshape(tmpdata[:,::2],(1,cur_data_len),order='F')
        pol1_med = np.median(pol1_data)
        # print(f"pol1_data :{pol1_data.shape}")
        pol1_data[0, int(pol1_data.shape[1]/2)] = pol1_med # DC ignore

        output_file1.write(pol1_data)    # write pol 1
        pol2_data = np.reshape(tmpdata[:,1::2],(1,cur_data_len),order='F')
        pol2_med = np.median(pol2_data)
        pol2_data[0, int(pol2_data.shape[1]/2)] = pol2_med # DC ignore
        output_file2.write(pol2_data)   # write pol 2
        n_total_samples += nChanSize
fread.close()
output_file1.close()
output_file2.close()

# build metadata files for two polarizations

if nbits != 8:
    dataset_format = "ci"+str(nbits)+"_be"
else:
    # endianness doesn't matter for single byte data
    dataset_format = "ci"+str(nbits)

print("dataset_format: ", dataset_format)

pol1_meta = {
    'global':
        {
            # SigMFFile.RECORDER_KEY: 'hackrf_transfer',
            SigMFFile.DATATYPE_KEY: dataset_format,
            SigMFFile.VERSION_KEY: f'{sigmf.__version__}',
            SigMFFile.SAMPLE_RATE_KEY: np.abs(dChanBW)*1e6,
            SigMFFile.HW_KEY:  telescope + ' ' + frontend,
            SigMFFile.AUTHOR_KEY:  observer,
            SigMFFile.DESCRIPTION_KEY: 'converted from GUPPI RAW - source observed : ' + source,
            'core:recorder': backend,
            'core:extensions' : {
                'antenna' : 'v0.9.0'
            },
            'antenna:low_frequency' : min(fLowChan,fHighChan),
            'antenna:high_frequency' : max(fLowChan,fHighChan),
            'antenna:cross_polar_discrimination' : 'polarization XX',
            'antenna:version' : 'v0.9.0'
        },
    'captures': [
        {
            SigMFFile.START_INDEX_KEY: 0,
            SigMFFile.FREQUENCY_KEY: TotCenFreq*1e6,
            SigMFFile.DATETIME_KEY: timedate
        }
    ],
    'annotations': [
        {
            SigMFFile.START_INDEX_KEY: 0,
            SigMFFile.LENGTH_INDEX_KEY: int(f'{n_total_samples}'),
            SigMFFile.FHI_KEY: float(8419.35E6),
            SigMFFile.FLO_KEY: int(8419.25E6),
            SigMFFile.LABEL_KEY: source,
        }
    ]
}

# the only difference between first and second polarization meta is the polarization field
pol2_meta = deepcopy(pol1_meta)
pol2_meta['global']['antenna:cross_polar_discrimination'] = 'polarization YY'


fopen1 = open(ofname_meta1,'w')
fopen1.write(json.dumps(pol1_meta,indent=2))
fopen1.close()
print(f"Wrote: {ofname_meta1}")

fopen2 = open(ofname_meta2,'w')
fopen2.write(json.dumps(pol2_meta,indent=2))
fopen2.close()
print(f"Wrote: {ofname_meta2}")


