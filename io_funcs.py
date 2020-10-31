import soundfile as sf
import librosa
import librosa.display
import subprocess

def read_list(filename):
    with open(filename) as f:
        with open(filename, 'r') as f:
            data = [l.rstrip() for l in f.readlines()]
    return data



def play_audio(filename=None, data=None, sample_rate=None):
    if filename:
        print("Play audio:", filename)
        subprocess.call(["cvlc", "--play-and-exit", filename])
    else:
        print("Play audio data")
        filename = '.tmp_audio_from_play_audio.wav'
        write_audio(filename, data, sample_rate)
        subprocess.call(["cvlc", "--play-and-exit", filename])


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


def write_audio(filename, data, sample_rate, dst_sample_rate=16000):

    if (dst_sample_rate is not None) and (dst_sample_rate != sample_rate):
        data = librosa.core.resample(data, sample_rate, dst_sample_rate)
        sample_rate = dst_sample_rate

    sf.write(filename, data, sample_rate)
    # librosa.output.write_wav(filename, data, sample_rate)
