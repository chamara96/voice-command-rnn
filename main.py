from neural_network import *
from audio_processing import *

classes = 0
model = 0


def load_model(model_path):
    src_weight_path = model_path
    src_classes_path = 'classes/classes.names'
    global classes, model

    model, classes = init_model(src_weight_path, src_classes_path)


def voice_command(file):
    filename = file
    # filename = 'testing_data/' + file
    # data, sample_rate = read_audio(filename, dst_sample_rate=None)

    # print(data)
    # print(len(data))
    # print(sample_rate)

    audio = AudioClass(filename=filename)
    # audio.plot_audio(plt_show=True)
    # audio.plot_mfcc(plt_show=True)
    ## audio.plot_mfcc_image(plt_show=True)
    # audio.plot_audio_and_mfcc(plt_show=True)
    # audio.plot_mfcc_histogram(plt_show=True)
    # play_audio(audio)
    # print(audio.)

    # audio = lib_datasets.AudioClass(filename=recorder.filename)

    probs = model.predict_audio_label_probabilities(audio)
    print(probs,classes)
    predicted_idx = np.argmax(probs)
    predicted_label = classes[predicted_idx]
    max_prob = probs[predicted_idx]
    # print("\nAll word labels: {}".format(classes))
    # print("\nPredicted label: {}, probability: {}\n".format(
    #     predicted_label, max_prob))
    PROB_THRESHOLD = 0.8
    final_label = predicted_label if max_prob > PROB_THRESHOLD else "none"
    return final_label, max_prob


load_model('model/my.ckpt')
print("Model Loaded")

# from os import listdir
# from os.path import isfile, join
#
# mypath = 'testing_data'
# onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
#
# for i in onlyfiles:
#     pred, score = voice_command(i)
#     print("Original:", i, "Predicted:", pred, "Score:", score)

# def filelist():
#     from os import listdir
#     from os.path import isfile, join
#
#     mypath = 'testing_data'
#     onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
#     print(onlyfiles)
#
# filelist()

# ========================= Record Voice

DST_AUDIO_FOLDER = "temp_voice/"
# import time
#
# while True:
#     file_name = record_audio.record_audio_and_classifiy(DST_AUDIO_FOLDER)
#     pred, score = voice_command(file_name)
#     print("Original:", file_name, "Predicted:", pred, "Score:", score)
#     time.sleep(10)
