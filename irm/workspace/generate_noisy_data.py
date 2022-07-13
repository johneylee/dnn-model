"""
example
python generate_noisy_data.py train
"""
import os
import sys
import numpy as np
import scipy.io.wavfile as wav
import librosa
from glob import glob


def generate_noisy_wav(wav_speech, wav_noise, snr):
	len_speech = len(wav_speech)
	len_noise = len(wav_noise)

	'''cut noise'''
	st = np.random.randint(0, len_noise - len_speech)
	ed = st + len_speech
	wav_noise = wav_noise[st:ed] # cut noise same as speech length

	'''cut dc element'''
	dc_speech = np.mean(wav_speech)
	dc_noise = np.mean(wav_noise)
	pow_speech = np.mean(np.power(wav_speech - dc_speech, 2.0))
	pow_noise = np.mean(np.power(wav_noise - dc_noise, 2.0))
	alpha = np.sqrt(10.0 ** (float(-snr) / 10.0) * pow_speech / pow_noise) # parameter for snr

	noisy_wav = wav_speech + alpha * wav_noise # construct noisy wav
	return noisy_wav

mode = 'devel'
speech_dir = '../data/speech/'
noise_dir = '../data/noise/'
# noise_dir = '../data/noise/' # original

snr_set = [-5, 0, 5 ,10]
snr_set_string = ['m5dB', '0dB', '5dB', '10dB']
num_snrs = len(snr_set)
noise_subset = ['noisex92_babble', 'noisex92_factory', 'noisex92_military', 'noisex92_pink', 'noisex92_volvo', 'noisex92_white']

# Make speech file list
# ex) ../data/speech/train/clean/*.wav
list_speech_files = sorted(glob(speech_dir + mode + os.sep + 'clean' + os.sep + '*.wav'))

# Make directory
save_dir = speech_dir + mode
# ex) ../data/speech/train or +/noisy
if os.path.isdir(save_dir) is False:
	os.system('mkdir ' + save_dir)
if os.path.isdir(save_dir + '/noisy') is False:
	os.system('mkdir ' + save_dir + '/noisy')
# Make lists for three mode
log_file_name = '../data/log_generate_data_' + mode + '.txt'
f = open(log_file_name, 'w')

if mode in ['train', 'devel']:
	# Make noise file list train
	list_noise_files = []
	# ex) ../data/noise/*/*.wav   <- To find all data
	list_noise_files = sorted(glob(noise_dir + 'train' + os.sep + 'noise*' + os.sep + '*.wav'))
	# list_noise_files = sorted(glob(noise_dir + 'noise*' + os.sep + '*.wav')) # original

	# number of speech file: 3,696 items
	for addr_speech in list_speech_files:

		# Read speech
		speech_file_name = addr_speech.split('/')[-1]
		wav_speech, fs = librosa.load(addr_speech, 16000)

		# Select and read noise / list_noise_files: 14 items
		ridx_noise_file = np.random.randint(0, len(list_noise_files))
		addr_noise = list_noise_files[ridx_noise_file]
		# noise load at wav_noise
		wav_noise, fs = librosa.load(addr_noise, 16000)
		# wav_noise = wav_noise[:int(len(wav_noise)*0.8)] # 80% noise type from noise file each other  # origianl

		# Select SNR
		ridx_snr = np.random.randint(0, len(snr_set)) # mix snr type
		snr_in_db = snr_set[ridx_snr]

		# Mix
		wav_noisy = generate_noisy_wav(wav_speech, wav_noise, snr_in_db)

		# Write
		addr_noisy = speech_dir + mode + '/noisy/' + speech_file_name
		wav_noisy = wav_noisy * 32768
		wav_noisy = wav_noisy.astype(np.int16)
		wav.write(addr_noisy, 16000, wav_noisy)

		# Show process
		print('%s > %s' % (addr_speech, addr_noisy))
		f.write('%s\t%s\t%s\t%d dB\n' % (addr_noisy, addr_speech, addr_noise, snr_in_db))
	f.close()

elif mode == 'test':

	# Make noise file list for test
	# ex) ../data/noise/*/*.wav
	list_noise_files = []
	list_noise_files = sorted(glob(noise_dir + 'test' + os.sep + 'noise*' + os.sep + '*.wav'))
	# list_noise_files = sorted(glob(noise_dir + 'noise*' + os.sep + '*.wav')) # original

	for nidx in range(len(list_noise_files)):
		foldername = os.path.basename(list_noise_files[nidx].replace('.wav', ''))
		os.system('mkdir ' + speech_dir + mode + '/noisy/' + foldername) # ../data/speech/test/noisy/


		for didx in range(len(snr_set)):
			# ex) ../data/speech/test/noisy/babble/10dB
			os.system('mkdir ' + speech_dir + mode + '/noisy/' + foldername + os.sep + snr_set_string[didx])
			for addr_speech in list_speech_files:
				# Read speech
				speech_file_name = addr_speech.split('/')[-1]
				wav_speech, fs = librosa.load(addr_speech, 16000)

				# Select and read noise
				addr_noise = list_noise_files[nidx]
				wav_noise, fs = librosa.load(addr_noise, 16000)
				# noise length 의 뒤에서 20%만 가져다가 test set 으로 사용
				# wav_noise = wav_noise[int(len(wav_noise)*0.8):]

				# Mix
				wav_noisy = generate_noisy_wav(wav_speech, wav_noise, snr_set[didx])

				# Write
				addr_noisy = speech_dir + mode + '/noisy/' + foldername + os.sep + snr_set_string[didx] + os.sep + speech_file_name

				wav_noisy = wav_noisy * 32768
				wav_noisy = wav_noisy.astype(np.int16)
				wav.write(addr_noisy, 16000, wav_noisy)

				# Show process
				print('%s > %s' % (addr_speech, addr_noisy))
				f.write('%s\t%s\t%s\t%d dB\n' % (addr_noisy, addr_speech, addr_noise, snr_set[didx]))
	f.close()















