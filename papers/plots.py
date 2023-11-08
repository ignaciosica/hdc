import numpy as np
import matplotlib.pyplot as plt


def bars(data, tlabels, elabels):
    colormap = plt.get_cmap("tab10", len(elabels))

    y = np.arange(len(tlabels))

    bar_width = 0.2
    _, ax = plt.subplots()

    for i, encoder in enumerate(elabels):
        try:
            means = [np.mean(arr) for arr in data[i, :, :]]
            std_devs = [np.std(arr) for arr in data[i, :, :]]
        except:
            continue
        #         try:
        #   print(x)
        # except:
        #   print("An exception occurred")

        for ii, value in enumerate(means):
            plt.text(
                0.01,
                y[ii] - (i * bar_width),
                f"{value:.4f} +- {std_devs[ii]:.4f}",
                va="center",
            )

        ax.barh(
            y - (i * bar_width),
            means,
            xerr=std_devs,
            capsize=1.5,
            height=bar_width,
            label=encoder,
            color=colormap(i),
        )

    y = y - bar_width * len(elabels) * 1.5 / 4

    ax.set_yticks(y)
    ax.set_yticklabels(tlabels)
    ax.legend(title="Encoders", loc="upper right")

    plt.show()


def bars_map(data, tlabels, elabels):
    colormap = plt.get_cmap("tab10", len(elabels))

    y = np.arange(len(tlabels))

    bar_width = 0.2
    _, ax = plt.subplots()

    for i, encoder in enumerate(elabels):
        try:
            means = [np.mean(arr) for arr in data[i, :, :]]
            std_devs = [np.std(arr) for arr in data[i, :, :]]
        except:
            continue

        for ii, value in enumerate(means):
            plt.text(
                0.01,
                y[ii] - (i * bar_width),
                f"{value:.4f} +- {std_devs[ii]:.4f}",
                va="center",
            )

        ax.barh(
            y - (i * bar_width),
            means,
            xerr=std_devs,
            capsize=1.5,
            height=bar_width,
            label=encoder,
            color=colormap(i),
        )

    y = y - bar_width * len(elabels) * 1.5 / 4

    ax.set_yticks(y)
    ax.set_yticklabels(tlabels)
    ax.legend(title="Encoders", loc="upper right")

    plt.show()


def bars_simple(data, labels):
    means = [np.mean(arr) for arr in data]
    std_devs = [np.std(arr) for arr in data]

    y = np.arange(len(means))

    plt.barh(y, means, xerr=std_devs, capsize=5, align="center")

    plt.yticks([i for i in range(len(means))], labels=labels)

    for i, value in enumerate(means):
        plt.text(0.5, y[i], f"{value:.2f}", va="center")

    # plt.tight_layout()
    plt.show()


# data = [np.random.normal(80, 5, 10) for _ in range(10)]
# data_labels = [f"Array {i+1}" for i in range(10)]
# bars(data, data_labels)
