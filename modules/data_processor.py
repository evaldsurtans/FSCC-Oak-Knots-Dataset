import os
import platform
import numpy as np
import sys
import json
import traceback
from sphfile import SPHFile # pip install sphfile
import multiprocessing
from multiprocessing import Process, Queue, Array, Value, Lock, Semaphore
import pyroomacoustics as pra   # pip install pyroomacoustics
import librosa
import matplotlib.pyplot as plt
import soundfile as sf
from loguru import logger
from datetime import datetime

#from librosa.feature import melspectrogram

from modules.file_utils import FileUtils
EPS = np.finfo(float).eps


class DataProcessor(Process):
    def __init__(self, data, args, process_idx):
        if sys.platform != 'darwin':
            multiprocessing.set_start_method('spawn', force=True)
        super().__init__()

        self.process_idx = process_idx
        self.args = args
        self.process_input_dir = data
        self.process_output_dir = None

    def init(self):
        filename = os.path.abspath('./logs/' + datetime.now().strftime('%y-%m-%d_auto') + '.log')
        logger.add(filename, rotation='00:00')
        if self.args.is_fixed_seed:
            np.random.seed(self.process_idx)

        if platform.system() == 'Windows':
            splited_dir = self.process_input_dir[0].split('\\')  # split path to get output dir
        else:
            splited_dir = self.process_input_dir[0].split('/')  # split path to get output dir
        self.process_output_dir = os.path.join(self.args.path_output_data, splited_dir[-2], splited_dir[-1])
        FileUtils.createDir(self.process_output_dir)    # create output dir

    def load_file(self, filename):
        try:
            if filename.endswith(('.json', '.csv', '.eaf', '.xml')):
                data = None
            elif filename.endswith('.sph'):
                sph = SPHFile(filename)
                data = librosa.util.buf_to_float(sph.content, n_bytes=sph.format['sample_n_bytes'])
                if data.shape[0] == 0:
                    logger.error(f'no data found inside: {filename}')
                    return None
                if sph.format['sample_rate'] != self.args.sample_rate:
                    sr = sph.format['sample_rate']
                    logger.info(f'Different sample rate {filename} {sr}')
                    data = librosa.core.resample(data, sr, self.args.sample_rate)
                    # for ted talks that are in sph format
                    # leave out beginning and end 15sec / 20sec
                    data = data[15 * self.args.sample_rate:(data.shape[0] - self.args.sample_rate * 20)]
            else:
                data = librosa.load(filename, sr=self.args.sample_rate)
                data = data[0]  # only data
            return data
        except Exception as e:
            logger.exception(str(e))
            exc_type, exc_value, exc_tb = sys.exc_info()
            logger.exception(traceback.format_exception(exc_type, exc_value, exc_tb))

    @staticmethod
    def is_clipped(data, clipping_threshold=0.99):
        return any(abs(data) > clipping_threshold)

    def active_rms(self, clean, noise, energy_thresh=-50):
        '''Returns the clean and noise RMS of the noise calculated only in the active portions'''
        window_size = 100 # in ms
        window_samples = int(self.args.sample_rate*window_size/1000)
        sample_start = 0
        noise_active_segs = []
        clean_active_segs = []

        while sample_start < len(noise):
            sample_end = min(sample_start + window_samples, len(noise))
            noise_win = noise[sample_start:sample_end]
            clean_win = clean[sample_start:sample_end]
            noise_seg_rms = 20*np.log10((noise_win**2).mean()+EPS)
            # Considering frames with energy
            if noise_seg_rms > energy_thresh:
                noise_active_segs = np.append(noise_active_segs, noise_win)
                clean_active_segs = np.append(clean_active_segs, clean_win)
            sample_start += window_samples

        if len(noise_active_segs)!=0:
            noise_rms = (noise_active_segs**2).mean()**0.5
        else:
            noise_rms = EPS

        if len(clean_active_segs)!=0:
            clean_rms = (clean_active_segs**2).mean()**0.5
        else:
            clean_rms = EPS
        return clean_rms, noise_rms

    @staticmethod
    def normalize_segmental_rms(audio, rms, target_level=-25):
        '''Normalize the signal to the target level
        based on segmental RMS'''
        scalar = 10 ** (target_level / 20) / (rms+EPS)
        audio = audio * scalar
        return audio

    def add_noise(self, clean_data, target_level=-25, clipping_threshold=0.99):
        with open(os.path.join(self.args.noise_file_path, 'noise.json'), 'r') as fp:
            noise_data_len = json.load(fp)['shape'][0]   # get data len
        noise_data = np.memmap(os.path.join(self.args.noise_file_path, 'noise.npy'), mode='r', dtype=np.float16, shape=(noise_data_len))

        snr = np.random.randint(self.args.snr_lower_limit, self.args.snr_upper_limit)

        if noise_data_len < len(clean_data):
            logger.warning(f'noise data - {noise_data_len} should be longer than speech data - {len(clean_data)}')
            return None, None, None

        noise_start_idx = np.random.randint(noise_data_len - len(clean_data))
        noise_data = noise_data[noise_start_idx:noise_start_idx+len(clean_data)]

        clean_data = clean_data/(max(abs(clean_data))+EPS)
        noise_data = noise_data/(max(abs(noise_data))+EPS)

        rmsclean, rmsnoise = self.active_rms(clean=clean_data, noise=noise_data)
        clean_data = self.normalize_segmental_rms(clean_data, rms=rmsclean, target_level=target_level)
        noise_data = self.normalize_segmental_rms(noise_data, rms=rmsnoise, target_level=target_level)

        # Set the noise level for a given SNR
        noisescalar = rmsclean / (10**(snr/20)) / (rmsnoise+EPS)
        noise_data = noise_data * noisescalar

        # Mix noise and clean speech
        noisy_data = clean_data + noise_data

        # Randomly select RMS value between -15 dBFS and -35 dBFS and normalize noisyspeech with that value
        # There is a chance of clipping that might happen with very less probability, which is not a major issue.
        noisy_rms_level = np.random.randint(self.args.target_level_lower, self.args.target_level_upper)
        rmsnoisy = (noisy_data**2).mean()**0.5
        scalarnoisy = 10 ** (noisy_rms_level / 20) / (rmsnoisy + EPS)
        noisy_data = noisy_data * scalarnoisy
        clean_data = clean_data * scalarnoisy

        # Final check to see if there are any amplitudes exceeding +/- 1. If so, normalize all the signals accordingly
        if self.is_clipped(noisy_data):
            noisyspeech_maxamplevel = max(abs(noisy_data)) / (clipping_threshold-EPS)
            noisy_data = noisy_data / noisyspeech_maxamplevel
            clean_data = clean_data / noisyspeech_maxamplevel
            noisy_rms_level = int(20*np.log10(scalarnoisy/noisyspeech_maxamplevel*(rmsnoisy+EPS)))

        return noisy_data, clean_data, noisy_rms_level

    def add_echo(self, data):
        chunk_size = int(len(data) / self.args.echo_parts)
        size_to_change = int(chunk_size * self.args.echo_percent_of_samples)  # size of part to mix with gaussian
        for chunk_idx in range(self.args.echo_parts):
            # generate room size
            room_x = np.random.uniform(low=self.args.room_size_x_min, high=self.args.room_size_x_max)
            room_y = np.random.uniform(low=self.args.room_size_y_min, high=self.args.room_size_y_max)
            corners = np.array([[0, 0], [0, room_y], [room_x, room_y], [room_x, 0]]).T  # [x,y]
            absorption = np.random.uniform(low=self.args.room_absorption_min, high=self.args.room_absorption_max)
            order = np.random.randint(low=0, high=self.args.room_max_order)
            room = pra.Room.from_corners(corners, fs=16000, absorption=absorption, max_order=order)
            room_z = np.random.uniform(low=self.args.room_ceiling_min, high=self.args.room_ceiling_min)
            room.extrude(room_z, absorption=absorption)
            # select data
            start_idx = int(chunk_size * (1 - self.args.echo_percent_of_samples))  # get max start idx for chunk
            start_idx = np.random.randint(start_idx) + (chunk_idx * chunk_size)
            end_idx = start_idx + size_to_change  # get end idx
            chunk_data = data[start_idx:end_idx]  # select data to simulate room
            # chunk_max = chunk_data.max()
            # chunk_min = chunk_data.min()
            # generate audio source
            source_x = np.random.uniform(low=0, high=room_x)
            source_y = np.random.uniform(low=0, high=room_y)
            source_z = np.random.uniform(low=self.args.room_speaker_height_min, high=self.args.room_speaker_height_max)
            room.add_source([source_x, source_y, source_z], signal=chunk_data) # [[x], [y], [z]]
            # generate microphone
            microphone_x = np.random.uniform(low=0, high=room_x)
            microphone_y = np.random.uniform(low=0, high=room_y)
            microphone_z = np.random.uniform(low=self.args.room_microphone_height_min, high=self.args.room_microphone_height_max)
            R = np.array([[microphone_x], [microphone_y], [microphone_z]])  # [[x], [y], [z]]
            room.add_microphone_array(pra.MicrophoneArray(R, room.fs))
            # simulate output audio
            room.simulate()
            chunk_data = room.mic_array.signals[0, :]
            chunk_clip_float = (chunk_data.shape[0] - (end_idx - start_idx)) / 2
            chunk_clip = int(chunk_clip_float)
            if chunk_clip_float != chunk_clip:
                chunk_data = chunk_data[chunk_clip:chunk_data.shape[0] - chunk_clip - 1]
            else:
                chunk_data = chunk_data[chunk_clip:chunk_data.shape[0] - chunk_clip]
            # chunk_data = self.normalize(chunk_data, chunk_min, chunk_max)
            data[start_idx:end_idx] = chunk_data[:]  # copy back data
        return data

    def run(self):
        try:
            self.init()

            included_samples = {}
            for file in self.process_input_dir[2]:
                input_filename = os.path.join(self.process_input_dir[0], file)
                clean_data = self.load_file(input_filename)
                noisy_data = None
                # skip non audio files
                if clean_data is None:
                    continue
                maximum_value = max(max(abs(clean_data)), 0.99)
                if 'noise' in self.process_input_dir[0].lower() or 'sound' in self.process_input_dir[0].lower():
                    data = clean_data[np.newaxis]
                    features = {'noise': 0}
                else:
                    # add echo from room acoustics library
                    if self.args.is_echo_added:
                        clean_data = self.add_echo(clean_data)
                    # add noise
                    if self.args.is_noises_added:
                        noisy_data, clean_data, noisy_rms_level = self.add_noise(clean_data)

                        if clean_data is None:
                            logger.warning(f'skipping sample {self.process_input_dir[0], file}. data should be shorter than noise data.')
                            continue

                        if self.is_clipped(clean_data, maximum_value) or self.is_clipped(noisy_data, maximum_value):
                            logger.warning(f'skipping sample {self.process_input_dir[0], file}. clipping after noise added')
                            continue

                    logger.info(f'{self.process_input_dir[0], file} finished augmentation.')
                    if noisy_data is not None:
                        data = np.stack((clean_data, noisy_data))
                        features = {
                            'clean': 0,
                            'noisy': 1
                        }
                    else:
                        data = clean_data[np.newaxis]
                        features = {'clean': 0}

                logger.info(f'{self.process_input_dir[0], file} added.')

                file_new = file.split('.')[0]
                included_samples[file_new] = {
                    'file': file_new + '.npy',
                    'shape': data.shape,
                    'features': features,
                    'dtype': 'float32'
                }
                file_path_new = os.path.join(self.process_output_dir, file_new + '.npy')
                data_memmap = np.memmap(filename=file_path_new, mode='w+', dtype='float32',
                                        shape=data.shape)
                data_memmap[:] = data[:]

            json_path = os.path.join(self.process_output_dir, 'info.json')
            with open(json_path, 'w') as outfile:
                json.dump(included_samples, outfile, indent=4)

        except Exception as e:
            logger.exception(str(e))
            exc_type, exc_value, exc_tb = sys.exc_info()
            logger.exception(traceback.format_exception(exc_type, exc_value, exc_tb))
