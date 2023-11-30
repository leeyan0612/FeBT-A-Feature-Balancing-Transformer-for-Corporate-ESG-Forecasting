import numpy as np
import matplotlib.pyplot as plt
import matplotlib

def RSE(pred, true):
    return np.sqrt(np.sum((true - pred) ** 2)) / np.sqrt(np.sum((true - true.mean()) ** 2))


def CORR(pred, true):
    u = ((true - true.mean(0)) * (pred - pred.mean(0))).sum(0)
    d = np.sqrt(((true - true.mean(0)) ** 2 * (pred - pred.mean(0)) ** 2).sum(0))
    return (u / d).mean(-1)


def MAE(pred, true):
    return np.mean(np.abs(pred - true))


def MSE(pred, true):
    return np.mean((pred - true) ** 2)


def RMSE(pred, true):
    return np.sqrt(MSE(pred, true))


def MAPE(pred, true):
    return np.mean(np.abs((pred - true) / true))


def MSPE(pred, true):
    return np.mean(np.square((pred - true) / true))


def metric(pred, true):
    mae = MAE(pred, true)
    mse = MSE(pred, true)
    rmse = RMSE(pred, true)
    mape = MAPE(pred, true)
    mspe = MSPE(pred, true)

    return mae, mse, rmse, mape, mspe


def accuarcy(preds, trues):
    correct = (preds == trues).sum()
    total = len(preds)
    acc = correct / total
    return acc


def plot_accs(accs):
    # 创建一个新的图像
    plt.figure(figsize=(10, 6))

    # 绘制准确率
    plt.plot(accs, label="Accuracy", color='blue')

    # 添加标题和标签
    plt.title("Accuracy over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()

    # 显示图像
    plt.show()


def plot_rec(accs):
    # 创建一个新的图像
    plt.figure(figsize=(10, 6))

    # 绘制准确率
    plt.plot(accs, label="Recall", color='blue')

    # 添加标题和标签
    plt.title("Recall over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Recall")
    plt.legend()

    # 显示图像
    plt.show()


def plot_mape(mapes):
    # 创建一个新的图像
    plt.figure(figsize=(10, 6))

    # 绘制准确率
    plt.plot(mapes, label="mape", color='blue')

    # 添加标题和标签
    plt.title("mape over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("MAPE")
    plt.legend()

    # 显示图像
    plt.show()


def plot_att(transposed_lists):
    # 绘制图像
    for i, data in enumerate(transposed_lists):
        plt.plot(data, label=f'year {i + 1}')

    plt.legend()  # 显示图例
    plt.show()  # 显示图像

    # 显示图像
    plt.show()


def plot_pfi(pfi_lists, cols):
    matplotlib.rcParams['font.sans-serif'] = ['SimHei']
    last_pfi_list = pfi_lists[-1]
    labels = [cols[i + 4] if i != 39 else cols[0] for i in range(len(last_pfi_list))]
    plt.bar(range(len(last_pfi_list)), last_pfi_list)
    plt.xticks(range(len(last_pfi_list)), labels, rotation='vertical')
    plt.tight_layout()
    ax = plt.gca()  # Get the current axis
    for label in ax.get_yticklabels():
        label.set_fontname('DejaVu Sans')
    plt.show()
