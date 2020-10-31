import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf
import librosa
import librosa.display
import subprocess
import cv2
import warnings

MFCC_RATE = 50  # TODO: It's about 1/50 s, I'm not sure.


class AudioClass(object):
    ''' A wrapper around the audio data
        to provide easy access to common operations on audio data.
    '''

    def __init__(self, data=None, sample_rate=None, filename=None, n_mfcc=12):
        if filename:
            self.data, self.sample_rate = read_audio(
                filename, dst_sample_rate=None)
        elif (len(data) and sample_rate):
            self.data, self.sample_rate = data, sample_rate
        else:
            assert 0, "Invalid input. Use keyword to input either (1) filename, or (2) data and sample_rate"

        self.mfcc = None
        self.n_mfcc = n_mfcc  # feature dimension of mfcc
        self.mfcc_image = None
        self.mfcc_histogram = None

        # Record info of original file
        self.filename = filename
        self.original_length = len(self.data)

    def get_len_s(self):  # audio length in seconds
        return len(self.data) / self.sample_rate

    def _check_and_compute_mfcc(self):
        if self.mfcc is None:
            self.compute_mfcc()

    def resample(self, new_sample_rate):
        self.data = librosa.core.resample(self.data, self.sample_rate,
                                          new_sample_rate)
        self.sample_rate = new_sample_rate

    def compute_mfcc(self, n_mfcc=None):
        # https://librosa.github.io/librosa/generated/librosa.feature.mfcc.html

        # Check input
        if n_mfcc is None:
            n_mfcc = self.n_mfcc
        if self.n_mfcc is None:
            self.n_mfcc = n_mfcc

        # Compute
        self.mfcc = compute_mfcc(self.data, self.sample_rate,
                                 n_mfcc)

    def compute_mfcc_histogram(
            self,
            bins=10,
            binrange=(-50, 200),
            col_divides=5,
    ):
        ''' Function:
                Divide mfcc into $col_divides columns.
                For each column, find the histogram of each feature (each row),
                    i.e. how many times their appear in each bin.
            Return:
                features: shape=(feature_dims, bins*col_divides)
        '''
        self._check_and_compute_mfcc()
        self.mfcc_histogram = calc_histogram(
            self.mfcc, bins, binrange, col_divides)

        self.args_mfcc_histogram = (  # record parameters
            bins,
            binrange,
            col_divides,
        )

    def compute_mfcc_image(
            self,
            row=200,
            col=400,
            mfcc_min=-200,
            mfcc_max=200,
    ):
        ''' Convert mfcc to an image by converting it to [0, 255]'''
        self._check_and_compute_mfcc()
        self.mfcc_img = mfcc_to_image(self.mfcc, row, col,
                                      mfcc_min, mfcc_max)

    # It's difficult to set this threshold, better not use this funciton.
    def remove_silent_prefix(self, threshold=50, padding_s=0.5):
        ''' Remove the silence at the beginning of the audio data. '''

        l0 = len(self.data) / self.sample_rate

        func = remove_silent_prefix_by_freq_domain
        self.data, self.mfcc = func(self.data,
                                    self.sample_rate,
                                    self.n_mfcc,
                                    threshold,
                                    padding_s,
                                    return_mfcc=True)

        l1 = len(self.data) / self.sample_rate
        print(f"Audio after removing silence: {l0} s --> {l1} s")

    # --------------------------- Plotting ---------------------------
    def plot_audio(self, plt_show=False, ax=None):
        plot_audio(self.data, self.sample_rate, ax=ax)
        if plt_show: plt.show()

    def plot_mfcc(self, method='librosa', plt_show=False, ax=None):
        self._check_and_compute_mfcc()
        plot_mfcc(self.mfcc, self.sample_rate, method, ax=ax)
        if plt_show: plt.show()

    def plot_audio_and_mfcc(self, plt_show=False, figsize=(12, 5)):
        plt.figure(figsize=figsize)

        plt.subplot(121)
        plot_audio(self.data, self.sample_rate, ax=plt.gca())

        plt.subplot(122)
        self._check_and_compute_mfcc()
        plot_mfcc(self.mfcc,
                  self.sample_rate,
                  method='librosa',
                  ax=plt.gca())

        if plt_show: plt.show()

    def plot_mfcc_histogram(self, plt_show=False):
        if self.mfcc_histogram is None:
            self.compute_mfcc_histogram()

        plot_mfcc_histogram(self.mfcc_histogram,
                            *self.args_mfcc_histogram)
        if plt_show: plt.show()

    def plot_mfcc_image(self, plt_show=False):
        if self.mfcc_image is None:
            self.compute_mfcc_image()
        plt.show(self.mfcc_img)
        plt.title("mfcc image")
        if plt_show: plt.show()

    # --------------------------- Input / Output ---------------------------
    def write_to_file(self, filename):
        write_audio(filename, self.data, self.sample_rate)

    def play_audio(self):
        play_audio(data=self.data, sample_rate=self.sample_rate)


def read_audio(filename, dst_sample_rate=16000, is_print=False):
    if 0:  # This takes 0.4 seconds to read an audio of 1 second. But support for more format
        data, sample_rate = librosa.load(filename)
    else:  # This only takes 0.01 seconds
        data, sample_rate = sf.read(filename)

    assert len(data.shape) == 1, "This project only support 1 dim audio."

    if (dst_sample_rate is not None) and (dst_sample_rate != sample_rate):
        data = librosa.core.resample(data, sample_rate, dst_sample_rate)
        sample_rate = dst_sample_rate

    if is_print:
        print(
            "Read audio file: {}.\n Audio len = {:.2}s, sample rate = {}, num points = {}"
                .format(filename, data.size / sample_rate, sample_rate, data.size))
    return data, sample_rate


def compute_mfcc(data, sample_rate, n_mfcc=12):
    # Extract MFCC features
    # https://librosa.github.io/librosa/generated/librosa.feature.mfcc.html
    mfcc = librosa.feature.mfcc(
        y=data,
        sr=sample_rate,
        n_mfcc=n_mfcc,  # How many mfcc features to use? 12 at most.
        # https://dsp.stackexchange.com/questions/28898/mfcc-significance-of-number-of-features
    )
    return mfcc


def calc_histogram(mfcc, bins=10, binrange=(-50, 200), col_divides=5):
    ''' Function:
            Divide mfcc into $col_divides columns.
            For each column, find the histogram of each feature (each row),
                i.e. how many times their appear in each bin.
        Return:
            features: shape=(feature_dims, bins*col_divides)
    '''
    feature_dims, time_len = mfcc.shape
    cc = time_len // col_divides  # cols / num_hist = size of each hist

    def calc_hist(row, cl, cr):
        hist, bin_edges = np.histogram(mfcc[row, cl:cr], bins=bins, range=binrange)
        return hist / (cr - cl)

    features = []
    for j in range(col_divides):
        row_hists = [calc_hist(row, j * cc, (j + 1) * cc) for row in range(feature_dims)]
        row_hists = np.vstack(row_hists)  # shape=(feature_dims, bins)
        features.append(row_hists)
    features = np.hstack(features)  # shape=(feature_dims, bins*col_divides)
    return features

    # if 0: # deprecated code
    #     for j in range(col_divides):
    #         row_hists = [calc_hist(row, j*cc, (j+1)*cc) for row in range(feature_dims) ]
    #         row_hists = np.vstack(row_hists) # shape=(feature_dims, bins)
    #         features += [row_hists.reshape((1, -1))] # shape=(feature_dims * bins, 1)
    #     features = np.vstack(features).ravel() # shape=(feature_dims * bins * col_divides, )
    #     return features


def mfcc_to_image(mfcc, row=200, col=400,
                  mfcc_min=-200, mfcc_max=200):
    ''' Convert mfcc to an image by converting it to [0, 255]'''

    # Rescale
    mfcc_img = 256 * (mfcc - mfcc_min) / (mfcc_max - mfcc_min)

    # Cut off
    mfcc_img[mfcc_img > 255] = 255
    mfcc_img[mfcc_img < 0] = 0
    mfcc_img = mfcc_img.astype(np.uint8)

    # Resize to desired size
    img = cv2.resize(mfcc_img, (col, row))
    return img


def remove_silent_prefix_of_mfcc(mfcc, threshold, padding_s=0.2):
    '''
    threshold:  Audio is considered started at t0 if mfcc[t0] > threshold
    padding: pad data at left (by moving the interval to left.)
    '''

    # Set voice intensity
    voice_intensity = mfcc[1]
    if 1:
        voice_intensity += mfcc[0]
        threshold += -100

    # Threshold to find the starting index
    start_indices = np.nonzero(voice_intensity > threshold)[0]

    # Return sliced mfcc
    if len(start_indices) == 0:
        warnings.warn("No audio satisifies the given voice threshold.")
        warnings.warn("Original data is returned")
        return mfcc
    else:
        start_idx = start_indices[0]
        # Add padding
        start_idx = max(0, start_idx - int(padding_s * MFCC_RATE))
        return mfcc[:, start_idx:]


def remove_silent_prefix_by_freq_domain(
        data, sample_rate, n_mfcc, threshold, padding_s=0.2,
        return_mfcc=False):
    # Compute mfcc, and remove silent prefix
    mfcc_src = compute_mfcc(data, sample_rate, n_mfcc)
    mfcc_new = remove_silent_prefix_of_mfcc(mfcc_src, threshold, padding_s)

    # Project len(mfcc) to len(data)
    l0 = mfcc_src.shape[1]
    l1 = mfcc_new.shape[1]
    start_idx = int(data.size * (1 - l1 / l0))
    new_audio = data[start_idx:]

    # Return
    if return_mfcc:
        return new_audio, mfcc_new
    else:
        return new_audio


def plot_audio(data, sample_rate, yrange=(-1.1, 1.1), ax=None):
    if ax is None:
        plt.figure(figsize=(8, 5))
    t = np.arange(len(data)) / sample_rate
    plt.plot(t, data)
    plt.xlabel('time (s)')
    plt.ylabel('Intensity')
    plt.title('Audio with {} points, and a {} sample rate, '.format(
        (len(data)), (sample_rate)))
    plt.axis([None, None, yrange[0], yrange[1]])


def plot_mfcc(mfcc, sample_rate, method='librosa', ax=None):
    if ax is None:
        plt.figure(figsize=(8, 5))
    assert method in ['librosa', 'cv2']

    if method == 'librosa':
        librosa.display.specshow(mfcc, sr=sample_rate, x_axis='time')
        plt.colorbar()
        plt.title(f'MFCCs features, len = {mfcc.shape[1]}')
        plt.tight_layout()

    elif method == 'cv2':
        cv2.imshow("MFCCs features", mfcc)
        q = cv2.waitKey()
        cv2.destroyAllWindows()


def plot_mfcc_histogram(
        mfcc_histogram,
        bins,
        binrange,
        col_divides,
):
    plt.figure(figsize=(15, 5))
    # Plot by plt
    plt.imshow(mfcc_histogram)

    # Add notations
    plt.xlabel('{} bins, {} range, and {} columns(pieces)'.format(
        (bins), (binrange), (col_divides)))
    plt.ylabel("Each feature's percentage")
    plt.title("Histogram of MFCCs features")
    plt.colorbar()


def play_audio(filename=None, data=None, sample_rate=None):
    if filename:
        print("Play audio:", filename)
        subprocess.call(["cvlc", "--play-and-exit", filename])
    else:
        print("Play audio data")
        filename = '.tmp_audio_from_play_audio.wav'
        write_audio(filename, data, sample_rate)
        subprocess.call(["cvlc", "--play-and-exit", filename])


def write_audio(filename, data, sample_rate, dst_sample_rate=16000):
    if (dst_sample_rate is not None) and (dst_sample_rate != sample_rate):
        data = librosa.core.resample(data, sample_rate, dst_sample_rate)
        sample_rate = dst_sample_rate

    sf.write(filename, data, sample_rate)
    # librosa.output.write_wav(filename, data, sample_rate)
