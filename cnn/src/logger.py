class Logger:
    def __init__(self, num_epochs):
        self.train_loss_list = []
        self.valid_loss_list = []
        self.train_acc_list = []
        self.valid_acc_list = []
        self.early_stop = num_epochs

    def write_loss(self, train_loss, valid_loss):
        self.train_loss_list.append(train_loss)
        self.valid_loss_list.append(valid_loss)

    def write_acc(self, train_acc, valid_acc):
        self.train_acc_list.append(train_acc)
        self.valid_acc_list.append(valid_acc)
