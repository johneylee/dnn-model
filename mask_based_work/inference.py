import os
import numpy as np
import tensorflow as tf
import librosa


def check_and_makedir(path):
    if not os.path.isdir(path):
        os.makedirs(path)


def stft(sig, framesize, overlapfac=0.5, window=np.hanning, nfft=512):
    hopsize = int(framesize * overlapfac)
    nframe = int(len(sig) / hopsize - 1)
    sig_padded = sig[0:(nframe + 1) * hopsize]
    zxx = np.array(
        [np.fft.fft(window(framesize) * sig_padded[n:n + framesize],
                    n=nfft) for n in range(0, len(sig) - framesize,
                                           hopsize)])
    return zxx[:, 0:int(nfft / 2) + 1]


def istft(zxx, framesize, overlapfac=0.5):
    hopsize = int(framesize * overlapfac)
    sig_reconstructed = np.zeros((zxx.shape[0] + 1) * hopsize)
    frm = np.real(np.fft.irfft(zxx))
    for n, i in enumerate(
            range(0, len(sig_reconstructed) - framesize, hopsize)):
        sig_reconstructed[i:i + framesize] += frm[n]
    return sig_reconstructed


class ModelInference:
    def __init__(self, args):
        self.arch = None
        self.in_type = None
        self.out_type = None
        self.save_dir = None
        self.save_path = None

    @staticmethod
    def check_and_makedir(path):
        if not os.path.isdir(path):
            os.makedirs(path)

    def estimate_IRM(self, sess, saver, model_path, foldername,
                     dataset, model, frm_len):
        # Load model
        self.arch = model.architecture['dnn']
        self.in_type = 'ctxt{0}logmag{1}'.format(dataset.context_size, 32)
        self.out_type = 'IRM32'
        dataset.load_test_list(self.in_type)
        dataset.load_distribution()
        saver.restore(sess, model_path)
        print('Loaded model from ' + model_path)

        self.distill_folder = os.path.join(foldername,
                                           model_path.split(os.sep)[-3])
        self.check_and_makedir(self.distill_folder)

        # Define file list
        file_list = sorted(dataset.test_list)

        for idx in range(len(file_list)):
            logmag, IRM = dataset.gen_batch_distill(file_list, idx,
                                                    infeature_type='.' + self.in_type,
                                                    outfeature_type='.' + self.out_type)
            train_output \
                = sess.run(model.fc_out,
                           feed_dict={model.X: logmag, model.Y: IRM,
                                      model.dropout: 0.0,
                                      model.phase: False})

            # plt.pcolormesh(np.abs(np.transpose(train_output)), cmap='jet')
            # plt.ylabel('frequency bin')
            # plt.xlabel('frame')
            # plt.show()
            # plt.savefig('target_teacher_-5dB_FDAW0_SI1406.png')
            file_info = file_list[idx].split(os.sep)
            filename = file_info[-1].split('.')[0]
            snr, noise_type = file_info[-2], file_info[-3]
            self.save_dir = os.path.join(self.distill_folder, noise_type, snr)
            self.check_and_makedir(self.save_dir)
            self.save_path = os.path.join(self.save_dir,
                                          '{0}.IRM{1:d}'.
                                          format(filename, int(frm_len * 1000)))
            train_output.astype('f').tofile(self.save_path)

            print("Saved file ", self.save_path)

        print("Architecture")
        print(model.architecture['dnn'])

    def waveform_reconstruction(self, save_dir, model_path, dataset, frm_len):
        """
        :param save_dir: directory to save enhanced waveform
        :param dataset: dataset
        :param IRM: estimated IRM
        :param frm_len: length of frame (sec)
        :return:
        """
        fs = 16000
        framesize = int(fs*frm_len)
        feature_dim = int(framesize / 2 + 1)
        self.in_type = 'wav'

        dataset.load_test_list(self.in_type)
        file_list = sorted(dataset.test_list)

        print(model_path)
        folder_dir = os.path.join(save_dir, model_path.split(os.sep)[-3])

        for file in file_list:
            noise_type, snr = file.split(os.sep)[-3], file.split(os.sep)[-2]
            IRM_filepath = os.path.join(folder_dir, noise_type, snr,
                                        file.split(os.sep)[-1])
            IRM = dataset._read_IRM(IRM_filepath.split('.')[0] + '.IRM32', feature_dim)
            noisy_speech, _ = librosa.load(file, sr=fs)
            noisy_stft = stft(noisy_speech,
                              framesize=framesize, overlapfac=0.5,
                              nfft=framesize)
            noisy_mag = np.abs(noisy_stft)
            noisy_phase = np.angle(noisy_stft)
            enhanced_stft = noisy_mag * IRM * np.exp(1j * noisy_phase)
            enhanced_speech = istft(enhanced_stft, framesize, overlapfac=0.5)
            save_dir = os.path.join(folder_dir, file.split(os.sep)[-3],
                                    file.split(os.sep)[-2])
            check_and_makedir(save_dir)
            save_path = os.path.join(save_dir, file.split(os.sep)[-1])
            librosa.output.write_wav(save_path, enhanced_speech, fs)
            print('Saved ', save_path)