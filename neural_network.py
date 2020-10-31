import numpy as np
import torch.nn as nn
import os
import matplotlib.pyplot as plt

if 1:
    from io_funcs import *
    from arguments import *


# Recurrent neural network (many-to-one)
class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, device, classes=None):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size,
                            hidden_size,
                            num_layers,
                            batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)
        self.device = device
        self.classes = classes

    def forward(self, x):
        # Set initial hidden and cell states
        batch_size = x.size(0)
        h0 = torch.zeros(self.num_layers, batch_size,
                         self.hidden_size).to(self.device)
        c0 = torch.zeros(self.num_layers, batch_size,
                         self.hidden_size).to(self.device)

        # Forward propagate LSTM
        out, _ = self.lstm(
            x, (h0, c0))  # shape = (batch_size, seq_length, hidden_size)

        # Decode the hidden state of the last time step
        out = self.fc(out[:, -1, :])
        return out

    def predict_probabilities(self, x):
        ''' Given the feature x of an audio sample,
            compute the probability of classified as each class.
        Arguments:
            x {np.array}: features of a sample audio.
                Shape = L*N, where
                L is length of the audio sequence,
                N is feature dimension.
        Return:
            probs {np.array}: Probabilities.
        '''
        x = torch.tensor(x[np.newaxis, :], dtype=torch.float32)
        x = x.to(self.device)
        outputs = self.forward(x).data.cpu().numpy().flatten()
        outputs = np.exp(outputs - max(outputs))  # softmax
        probs = outputs / sum(outputs)
        return probs

    def predict(self, x):
        ''' Predict the label of the input feature of an audio.
        Arguments:
            x {np.array}: features of a sample audio.
                Shape = L*N, where
                L is length of the audio sequence,
                N is feature dimension.
        '''
        x = torch.tensor(x[np.newaxis, :], dtype=torch.float32)
        x = x.to(self.device)
        outputs = self.forward(x)
        _, predicted = torch.max(outputs.data, 1)
        predicted_index = predicted.item()
        return predicted_index

    def set_classes(self, classes):
        self.classes = classes

    def predict_audio_label(self, audio):
        idx = self.predict_audio_label_index(audio)
        if not self.classes:
            raise RuntimeError("Classes names are not set. Don't know what audio label is")
        label = self.classes[idx]
        return label

    def predict_audio_label_index(self, audio):
        audio.compute_mfcc()
        x = audio.mfcc.T  # (time_len, feature_dimension)
        idx = self.predict(x)
        return idx

    def predict_audio_label_probabilities(self, audio):
        audio.compute_mfcc()
        x = audio.mfcc.T  # (time_len, feature_dimension)
        probs = self.predict_probabilities(x)
        return probs


def load_weights(model, weights, is_print=False):
    # Load weights into model.
    # If param's name is different, raise error.
    # If param's size is different, skip this param.
    # see: https://discuss.pytorch.org/t/how-to-load-part-of-pre-trained-model/1113/2

    for i, (name, param) in enumerate(weights.items()):
        model_state = model.state_dict()

        if name not in model_state:
            print("-" * 80)
            print("weights name:", name)
            print("RNN states names:", model_state.keys())
            assert 0, "Wrong weights file"

        model_shape = model_state[name].shape
        if model_shape != param.shape:
            print(
                "\nWarning: Size of {} layer is different between model and weights. Not copy parameters."
                    .format((name)))
            print("\tModel shape = {}, weights' shape = {}.".format(
                (model_shape), (param.shape)))
        else:
            model_state[name].copy_(param)


def create_RNN_model(args, load_weight_from=None):
    ''' A wrapper for creating a 'class RNN' instance '''

    # Update some dependent args
    if hasattr(args, "classes"):
        classes = args.classes
    elif hasattr(args, "classes_txt"):
        classes = read_list(args.classes_txt)
    else:
        raise RuntimeError("The classes are no loaded into the RNN model.")
    args.num_classes = len(classes)
    args.save_log_to = args.save_model_to + "log.txt"
    args.save_fig_to = args.save_model_to + "fig.jpg"

    # Create model
    device = args.device
    model = RNN(args.input_size, args.hidden_size, args.num_layers,
                args.num_classes, device).to(device)
    model.set_classes(classes)

    # Load weights
    if load_weight_from:
        print(f"Load weights from: {load_weight_from}")
        weights = torch.load(load_weight_from, map_location=torch.device('cpu'))
        load_weights(model, weights)

    return model


def setup_default_RNN_model(weight_filepath, classes_txt):
    ''' Given filepath of the weight file and the classes,
        Initilize the RNN model with default parameters.
    '''
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_args = set_default_args()
    model_args.classes_txt = classes_txt
    model = create_RNN_model(model_args, weight_filepath)
    classes = model.classes
    if 0:  # Test with random data
        label_index = model.predict(np.random.random((66, 12)))
        print("Label index of a random feature: ", label_index)
        exit("Complete test.")
    return model, classes


def predict_audio_label_probabilities(self, audio):
    audio.compute_mfcc()
    x = audio.mfcc.T  # (time_len, feature_dimension)
    probs = predict_probabilities(x)
    return probs


def predict_probabilities(self, x):
    ''' Given the feature x of an audio sample,
        compute the probability of classified as each class.
    Arguments:
        x {np.array}: features of a sample audio.
            Shape = L*N, where
            L is length of the audio sequence,
            N is feature dimension.
    Return:
        probs {np.array}: Probabilities.
    '''
    x = torch.tensor(x[np.newaxis, :], dtype=torch.float32)
    x = x.to(self.device)
    outputs = self.forward(x).data.cpu().numpy().flatten()
    outputs = np.exp(outputs - max(outputs))  # softmax
    probs = outputs / sum(outputs)
    return probs


# classes = 0
# model = 0


def init_model(src_weight_path, src_classes_path):
    # Init model
    # global classes, model
    model, classes = setup_default_RNN_model(src_weight_path, src_classes_path)
    print("Number of classes = {}, classes: {}".format(len(classes), classes))
    model.set_classes(classes)
    return model,classes

def fix_weights_except_fc(model):
    not_fix = "fc"
    for name, param in model.state_dict().items():
        if not_fix in name:
            continue
        else:
            print(f"Fix {name} layer", end='. ')
            param.requires_grad = False
    print("")

def evaluate_model(model, eval_loader, num_to_eval=-1):
    ''' Eval model on a dataset '''
    device = model.device
    correct = 0
    total = 0
    for i, (featuress, labels) in enumerate(eval_loader):

        featuress = featuress.to(device)  # (batch, seq_len, input_size)
        labels = labels.to(device)

        # Predict
        outputs = model(featuress)

        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        # stop
        if i + 1 == num_to_eval:
            break
    eval_accu = correct / total
    print('  Evaluate on eval or test dataset with {} samples: Accuracy = {}%'.
          format(i + 1, 100 * eval_accu))
    return eval_accu


class TrainingLog(object):
    def __init__(
            self,
            training_args=None,  # arguments in training
            #  MAX_EPOCH = 1000,
    ):

        if not isinstance(training_args, dict):
            training_args = training_args.__dict__
        self.training_args = training_args

        self.epochs = []
        self.accus_train = []
        self.accus_eval = []
        self.accus_test = []

    def store_accuracy(self, epoch, train=-0.1, eval=-0.1, test=-0.1):
        self.epochs.append(epoch)
        self.accus_train.append(train)
        self.accus_eval.append(eval)
        self.accus_test.append(test)
        # self.accu_table[epoch] = self.AccuItems(train, eval, test)

    def plot_train_eval_accuracy(self):
        plt.cla()
        t = self.epochs
        plt.plot(t, self.accus_train, 'r.-', label="train")
        plt.plot(t, self.accus_eval, 'b.-', label="eval")
        plt.title("Accuracy on train/eval dataset")
        plt.xlabel("epoch")
        plt.ylabel("accuracy")

        # lim
        # plt.ylim([0.2, 1.05])
        plt.legend(loc='upper left')

    def save_log(self, filename):
        with open(filename, 'w') as f:

            # -- Args
            f.write("Args:" + "\n")
            for key, val in self.training_args.items():
                s = "\t{:<20}: {}".format(key, val)
                f.write(s + "\n")
            f.write("\n")
            # -- Accuracies
            f.write("Accuracies:" + "\n")
            f.write("\t{:<10}{:<10}{:<10}{:<10}\n".format(
                "Epoch", "Train", "Eval", "Test"))

            for i in range(len(self.epochs)):
                epoch = self.epochs[i]
                train = self.accus_train[i]
                eval = self.accus_eval[i]
                test = self.accus_test[i]
                f.write("\t{:<10}{:<10.3f}{:<10.3f}{:<10.3f}\n".format(
                    epoch, train, eval, test))

epoch_gb=0
total_epoch_gb=10
def check_curr_epoch():
    return epoch_gb,total_epoch_gb

def train_model(model, args, train_loader, eval_loader):
    global epoch_gb,total_epoch_gb
    total_epoch_gb=args.num_epochs

    device = model.device
    logger = TrainingLog(training_args=args)
    if args.finetune_model:
        fix_weights_except_fc(model)

    # -- create folder for saving model
    if args.save_model_to:
        if not os.path.exists(args.save_model_to):
            os.makedirs(args.save_model_to)

    # -- Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=args.learning_rate,
                                 weight_decay=args.weight_decay)
    optimizer.zero_grad()

    # -- For updating learning rate
    def update_lr(optimizer, lr):
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    # -- Train the model
    total_step = len(train_loader)
    curr_lr = args.learning_rate
    cnt_batches = 0

    for epoch in range(1, 1 + args.num_epochs):
        epoch_gb=epoch
        cnt_correct, cnt_total = 0, 0
        for i, (featuress, labels) in enumerate(train_loader):
            cnt_batches += 1
            ''' original code of pytorch-tutorial:
            images = images.reshape(-1, sequence_length, input_size).to(device)
            labels = labels.to(device)
            # we can see that the shape of images should be: 
            #    (batch_size, sequence_length, input_size)
            '''
            featuress = featuress.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(featuress)
            loss = criterion(outputs, labels)

            # Backward and optimize
            loss.backward()  # error
            if cnt_batches % args.gradient_accumulations == 0:
                # Accumulates gradient before each step
                optimizer.step()
                optimizer.zero_grad()

            # Record result
            _, argmax = torch.max(outputs, 1)
            cnt_correct += (labels == argmax.squeeze()).sum().item()
            cnt_total += labels.size(0)

            # Print accuracy
            train_accu = cnt_correct / cnt_total
            if (i + 1) % 50 == 0 or (i + 1) == len(train_loader):
                print(
                    'Epoch [{}/{}], Step [{}/{}], Loss = {:.4f}, Train accuracy = {:.2f}'
                    .format(epoch, args.num_epochs, i + 1, total_step,
                            loss.item(), 100 * train_accu))
            continue
        print(f"Epoch {epoch} completes")

        # -- Decay learning rate
        if (epoch) % args.learning_rate_decay_interval == 0:
            curr_lr *= args.learning_rate_decay_rate  # lr = lr * rate
            update_lr(optimizer, curr_lr)

        # -- Evaluate and save model
        if (epoch) % 1 == 0 or (epoch) == args.num_epochs:
            eval_accu = evaluate_model(model, eval_loader, num_to_eval=-1)
            if args.save_model_to:
                name_to_save = args.save_model_to + "/" + "{:03d}".format(
                    epoch) + ".ckpt"
                torch.save(model.state_dict(), name_to_save)
                print("Save model to: ", name_to_save)

            # logger record
            logger.store_accuracy(epoch, train=train_accu, eval=eval_accu)
            logger.save_log(args.save_log_to)

            # logger Plot
            if args.plot_accu and epoch == 1:
                plt.figure(figsize=(10, 8))
                plt.ion()
                if args.show_plotted_accu:
                    plt.show()
            if (epoch == args.num_epochs) or (args.plot_accu and epoch > 1):
                logger.plot_train_eval_accuracy()
                if args.show_plotted_accu:
                    plt.pause(0.01)
                plt.savefig(fname=args.save_fig_to)

        # An epoch end
        print("-" * 80 + "\n")

    # Training end
    return
