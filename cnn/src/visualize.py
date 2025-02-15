import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")


def plot_loss_accuracy(logger, file_path):
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].plot(logger.train_loss_list, color="#2980B9", label="train loss")
    ax[0].plot(logger.valid_loss_list, color="#17A589", label="valid loss")
    ax[0].axvline(
        x=logger.early_stop, color="#E74C3C", linestyle="--", label="early stop"
    )
    ax[0].set_ylabel("loss")
    ax[0].legend(loc="upper left")
    ax[1].plot(logger.train_acc_list, color="#2980B9", label="train acc")
    ax[1].plot(logger.valid_acc_list, color="#17A589", label="valid acc")
    ax[1].set_ylabel("accuracy")
    ax[1].axvline(
        x=logger.early_stop, color="#E74C3C", linestyle="--", label="early stop"
    )
    ax[1].legend(loc="lower left")
    fig.tight_layout()
    fig.savefig(file_path)
