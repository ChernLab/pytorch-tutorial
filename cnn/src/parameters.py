class Params:
    def __init__(self):
        self.num_epochs = 1000
        self.early_stop = 10
        self.batch_size = 128
        self.learn_rate = 0.0005

    def print_options(self):
        print("Training options:")
        print("Number of epochs: {}".format(self.num_epochs))
        print("Early stop:       {}".format(self.early_stop))
        print("Batch size:       {}".format(self.batch_size))
        print("Learning rate:    {}".format(self.learn_rate))
        print()  # line break
