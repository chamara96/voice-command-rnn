import arguments
import io_funcs
import audio_processing
import neural_network

import copy
import matplotlib.pyplot as plt
import librosa
from torch.utils.data import Dataset
import glob
import time
import numpy as np
import sklearn



if 1:
    import torch


def split_train_test(X,
                     Y,
                     test_size=0,
                     use_all_data_to_train=False,
                     dtype='numpy',
                     if_print=True):
    assert dtype in ['numpy', 'list']

    def _print(s):
        if if_print:
            print(s)

    _print("split_train_test:")

    if dtype == 'numpy':
        _print("\tData size = {}, feature dimension = {}".format(
            X.shape[0], X.shape[1]))
        if use_all_data_to_train:
            tr_X = np.copy(X)
            tr_Y = np.copy(Y)
            te_X = np.copy(X)
            te_Y = np.copy(Y)
        else:
            f = sklearn.model_selection.train_test_split
            tr_X, te_X, tr_Y, te_Y = f(
                X, Y, test_size=test_size, random_state=0)

    elif dtype == 'list':
        print(X)
        _print("\tData size = {}, feature dimension = {}".format(
            len(X), len(X[0])))

        if use_all_data_to_train:
            tr_X = X[:]
            tr_Y = Y[:]
            te_X = X[:]
            te_Y = Y[:]
        else:
            N = len(Y)
            train_size = int((1 - test_size) * N)
            randidx = np.random.permutation(N)
            n1, n2 = randidx[0:train_size], randidx[train_size:]

            def get(arr_vals, arr_idx):
                return [arr_vals[idx] for idx in arr_idx]

            tr_X = get(X, n1)[:]
            tr_Y = get(Y, n1)[:]
            te_X = get(X, n2)[:]
            te_Y = get(Y, n2)[:]
    _print("\tNum training: {}".format(len(tr_Y)))
    _print("\tNum evaluation: {}".format(len(te_Y)))
    return tr_X, tr_Y, te_X, te_Y


def split_train_eval_test(X, Y, ratios=[0.8, 0.1, 0.1], dtype='list'):
    X1, Y1, X2, Y2 = split_train_test(X,
                                      Y,
                                      1 - ratios[0],
                                      dtype=dtype,
                                      if_print=False)

    X2, Y2, X3, Y3 = split_train_test(X2,
                                      Y2,
                                      ratios[2] / (ratios[1] + ratios[2]),
                                      dtype=dtype,
                                      if_print=False)

    r1, r2, r3 = 100 * ratios[0], 100 * ratios[1], 100 * ratios[2]
    n1, n2, n3 = len(Y1), len(Y2), len(Y3)
    print("Split data into [Train={} ({}%), Eval={} ({}%),  Test={} ({}%)]".
          format((n1), (r1), (n2), (r2), (n3), (r3)))
    tr_X, tr_Y, ev_X, ev_Y, te_X, te_Y = X1, Y1, X2, Y2, X3, Y3
    return tr_X, tr_Y, ev_X, ev_Y, te_X, te_Y


class Timer(object):
    def __init__(self):
        self.t0 = time.time()

    def report_time(self, event="", prefix=""):
        print(prefix + "Time cost of '{}' is: {:.2f} seconds.".format(
            event, time.time() - self.t0))


class AudioClass(object):
    ''' A wrapper around the audio data
        to provide easy access to common operations on audio data.
    '''

    def __init__(self, data=None, sample_rate=None, filename=None, n_mfcc=12):
        if filename:
            self.data, self.sample_rate = io_funcs.read_audio(
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
        self.mfcc = audio_processing.compute_mfcc(self.data, self.sample_rate,
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
        self.mfcc_histogram = audio_processing.calc_histogram(
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
        self.mfcc_img = audio_processing.mfcc_to_image(self.mfcc, row, col,
                                                       mfcc_min, mfcc_max)

    # It's difficult to set this threshold, better not use this funciton.
    def remove_silent_prefix(self, threshold=50, padding_s=0.5):
        ''' Remove the silence at the beginning of the audio data. '''

        l0 = len(self.data) / self.sample_rate

        func = audio_processing.remove_silent_prefix_by_freq_domain
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
        audio_processing.plot_audio(self.data, self.sample_rate, ax=ax)
        if plt_show: plt.show()

    def plot_mfcc(self, method='librosa', plt_show=False, ax=None):
        self._check_and_compute_mfcc()
        audio_processing.plot_mfcc(self.mfcc, self.sample_rate, method, ax=ax)
        if plt_show: plt.show()

    def plot_audio_and_mfcc(self, plt_show=False, figsize=(12, 5)):
        plt.figure(figsize=figsize)

        plt.subplot(121)
        audio_processing.plot_audio(self.data, self.sample_rate, ax=plt.gca())

        plt.subplot(122)
        self._check_and_compute_mfcc()
        audio_processing.plot_mfcc(self.mfcc,
                                   self.sample_rate,
                                   method='librosa',
                                   ax=plt.gca())

        if plt_show: plt.show()

    def plot_mfcc_histogram(self, plt_show=False):
        if self.mfcc_histogram is None:
            self.compute_mfcc_histogram()

        audio_processing.plot_mfcc_histogram(self.mfcc_histogram,
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
        audio_processing.write_audio(filename, self.data, self.sample_rate)

    def play_audio(self):
        audio_processing.play_audio(data=self.data, sample_rate=self.sample_rate)


def get_filenames(folder, file_types=('*.wav',)):
    filenames = []

    if not isinstance(file_types, tuple):
        file_types = [file_types]

    for file_type in file_types:
        filenames.extend(glob.glob(folder + "/" + file_type))
    filenames.sort()
    return filenames


class AudioDataset(Dataset):
    ''' A dataset class for Pytorch to load data '''

    def __init__(
            self,
            data_folder="",
            classes_txt="",
            file_paths=[],
            file_labels=[],
            transform=None,
            is_cache_audio=False,
            is_cache_XY=True,  # cache features
    ):

        assert (data_folder and classes_txt) or (file_paths, file_labels)

        # Get all data's filename and label
        if not (file_paths and file_labels):
            file_paths, file_labels = AudioDataset.load_classes_and_data_filenames(
                classes_txt, data_folder)
        self._file_paths = file_paths
        self._file_labels = torch.tensor(file_labels, dtype=torch.int64)

        # Data augmentation methods are saved inside the `transform`
        self._transform = transform

        # Cache computed data
        self._IS_CACHE_AUDIO = is_cache_audio
        self._cached_audio = {}  # idx : audio
        self._IS_CACHE_XY = is_cache_XY
        self._cached_XY = {}  # idx : (X, Y). By default, features will be cached

    @staticmethod
    def load_classes_and_data_filenames(classes_txt, data_folder):
        '''
        Load classes names and all training data's file_paths.
        Arguments:
            classes_txt {str}: filepath of the classes.txt
            data_folder {str}: path to the data folder.
                The folder should contain subfolders named as the class name.
                Each subfolder contain many .wav files.
        '''
        # Load classes
        print(classes_txt)
        with open(classes_txt, 'r') as f:
            classes = [l.rstrip() for l in f.readlines()]

        # Based on classes, load all filenames from data_folder
        file_paths = []
        file_labels = []
        for i, label in enumerate(classes):
            folder = data_folder + "/" + label + "/"

            names = get_filenames(folder, file_types="*.wav")
            labels = [i] * len(names)

            file_paths.extend(names)
            file_labels.extend(labels)

        print("Load data from: ", data_folder)
        print("\tClasses: ", ", ".join(classes))
        return file_paths, file_labels

    def __len__(self):
        return len(self._file_paths)

    def get_audio(self, idx):
        ''' Load (idx)th audio, either from cached data, or from disk '''
        if idx in self.cached_audio:  # load from cached
            audio = copy.deepcopy(self.cached_audio[idx])  # copy from cache
        else:  # load from file
            filename = self._file_paths[idx]
            audio = AudioClass(filename=filename)
            # print(f"Load file: {filename}")
            self.cached_audio[idx] = copy.deepcopy(audio)  # cache a copy
        return audio

    def __getitem__(self, idx):

        timer = Timer()

        # -- Load audio
        if self._IS_CACHE_AUDIO:
            audio = self.get_audio(idx)
            print("{:<20}, len={}, file={}".format("Load audio from file",
                                                   audio.get_len_s(),
                                                   audio.filename))
        else:  # load audio from file
            if (idx in self._cached_XY) and (not self._transform):
                # if (1) audio has been processed, and (2) we don't need data augumentation,
                # then, we don't need audio data at all. Instead, we only need features from self._cached_XY
                pass
            else:
                filename = self._file_paths[idx]
                audio = AudioClass(filename=filename)

        # -- Compute features
        is_read_features_from_cache = (self._IS_CACHE_XY) and (
                idx in self._cached_XY) and (not self._transform)

        # Read features from cache:
        #   If already computed, and no augmentatation (transform), then read from cache
        if is_read_features_from_cache:
            X, Y = self._cached_XY[idx]

        # Compute features:
        #   if (1) not loaded, or (2) need new transform
        else:
            # Do transform (augmentation)
            if self._transform:
                audio = self._transform(audio)
                # self._transform(audio) # this is also good. Transform (Augment) is done in place.

            # Compute mfcc feature
            audio.compute_mfcc(n_mfcc=12)  # return mfcc

            # Compose X, Y
            X = torch.tensor(
                audio.mfcc.T,
                dtype=torch.float32)  # shape=(time_len, feature_dim)
            Y = self._file_labels[idx]

            # Cache
            if self._IS_CACHE_XY and (not self._transform):
                self._cached_XY[idx] = (X, Y)

        # print("{:>20}, len={:.3f}s, file={}".format("After transform", audio.get_len_s(), audio.filename))
        # timer.report_time(event="Load audio", prefix='\t')
        return (X, Y)


args = arguments.set_default_args()


def init_data_handling():
    args.num_epochs = 6
    args.learning_rate = 0.001
    args.train_eval_test_ratio = [0.7, 0.3, 0.0]
    args.do_data_augment = False
    args.data_folder = "training_data/"
    args.classes_txt = "classes/classes.names"
    args.load_weight_from = "model/my.ckpt"
    args.finetune_model = True  # If true, fix all parameters except the fc layer
    args.save_model_to = 'checkpoints/'  # Save model and log file

end_train=0

def start_model():
    global end_train

    init_data_handling()

    # Get data's filenames and labels
    file_paths, file_labels = AudioDataset.load_classes_and_data_filenames(args.classes_txt, args.data_folder)

    print(len(file_paths))
    print(len(file_labels))

    tr_X, tr_Y, ev_X, ev_Y, te_X, te_Y = split_train_eval_test(
        X=file_paths, Y=file_labels, ratios=args.train_eval_test_ratio, dtype='list')
    train_dataset = AudioDataset(file_paths=tr_X, file_labels=tr_Y, transform=None)
    eval_dataset = AudioDataset(file_paths=ev_X, file_labels=ev_Y, transform=None)

    print("Done")

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True)
    eval_loader = torch.utils.data.DataLoader(dataset=eval_dataset, batch_size=args.batch_size, shuffle=True)

    print("AAAAA === Create model and train")
    # Create model and train -------------------------------------------------
    model = neural_network.create_RNN_model(args, load_weight_from=args.load_weight_from)  # create model   ,load_weight_from=args.load_weight_from
    neural_network.train_model(model, args, train_loader, eval_loader)

    print("END")

    end_train=1

# def conn_with_windows():
#     training_window.vp_start_gui()
#
#     print("conneeww")