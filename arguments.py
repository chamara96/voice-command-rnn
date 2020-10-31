if 1:
    import torch

class SimpleNamespace:
    ''' This is the same as types.SimpleNamespace '''

    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    def __repr__(self):
        keys = sorted(self.__dict__)
        items = ("{}={!r}".format(k, self.__dict__[k]) for k in keys)
        return "{}({})".format(type(self).__name__, ", ".join(items))

    def __eq__(self, other):
        return self.__dict__ == other.__dict__


def set_default_args():
    args = SimpleNamespace()

    # model params
    args.input_size = 12  # == n_mfcc
    args.batch_size = 1
    args.hidden_size = 64
    args.num_layers = 3

    # training params
    args.num_epochs = 100
    args.learning_rate = 0.0001
    args.learning_rate_decay_interval = 5  # decay for every 5 epochs
    args.learning_rate_decay_rate = 0.5  # lr = lr * rate
    args.weight_decay = 0.00
    args.gradient_accumulations = 16  # number of gradient accums before step

    # training params2
    args.load_weight_from = None
    args.finetune_model = False  # If true, fix all parameters except the fc layer
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # data
    args.data_folder = "data/data_train/"
    args.train_eval_test_ratio = [0.9, 0.1, 0.0]
    args.do_data_augment = False

    # labels
    args.classes_txt = ""
    args.num_classes = None  # should be added with a value somewhere, like this:
    #                = len(lib_io.read_list(args.classes_txt))

    # log setting
    args.plot_accu = True  # if true, plot accuracy for every epoch
    args.show_plotted_accu = False  # if false, not calling plt.show(), so drawing figure in background
    args.save_model_to = 'checkpoints/'  # Save model and log file
    # e.g: model_001.ckpt, log.txt, log.jpg

    return args
