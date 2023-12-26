import numpy as np
from matplotlib import pyplot as plt


def sci1(x):
    # x -> [n, k]
    k = x.shape[1]
    l1 = np.linalg.norm(x, ord=1, axis=-1)
    return (k * np.max(np.abs(x), axis=-1) / l1 - 1) / (k - 1)


def sci5(x):
    # x -> [n, k]
    k = x.shape[1]
    l1 = np.linalg.norm(x, ord=1, axis=-1)
    ids = np.argsort(np.abs(x), axis=-1)
    xt = np.take_along_axis(np.abs(x), ids, axis=-1)
    xt = xt[:, -5:].sum(axis=-1)
    return (k * xt / l1 - 1) / (k - 1)


CLASS_LABELS = ["Pedestrian", "Two Wheeler"]


def show_topk(x, D, Dl, y, yl, s, i, k):
    xi = x[i].numpy()
    ids = np.argsort(np.abs(xi))[::-1][:k]
    title = CLASS_LABELS[yl[i]]
    r = 2
    c = k // 2
    fig, ax = plt.subplots(r, c + 1, figsize=(c * 3, 9))
    ax[0][0].imshow(y[i].reshape(*D.shape[-2:]))
    ax[0][0].axis("off")
    ax[0][0].set_title(title + "\nSCI = {:.3f}".format(s[i]))
    xhat = np.dot(D[ids].transpose(1, 2, 0), xi[ids])
    ax[1][0].imshow(xhat)
    ax[1][0].axis("off")
    ax[1][0].set_title("Reconstructed\nTop {}".format(k))
    for j, ii in enumerate(ids):
        title = CLASS_LABELS[Dl[ii]]
        ax[j // c][j % c + 1].imshow(D[ii].reshape(*D.shape[-2:]))
        ax[j // c][j % c + 1].axis("off")
        ax[j // c][j % c + 1].set_title(title + "\nCoef = {:.3f}".format(xi[ii]))


def show_topd(dict_, x, y, yl, s, i, k):
    xi = x[i].numpy()
    D = dict_.dict.weight.detach().cpu().numpy().reshape(-1, *y.shape[-2:])
    ids = np.argsort(np.abs(xi))[::-1][:k]
    print(ids)
    title = CLASS_LABELS[yl[i]]
    r = 2
    c = k // 2
    fig, ax = plt.subplots(r, c + 1, figsize=(c * 3, 9))
    ax[0][0].imshow(y[i])
    ax[0][0].axis("off")
    ax[0][0].set_title(title + "\nSCI = {:.3f}".format(s[i]))
    xhat = np.dot(D[ids].transpose(1, 2, 0), xi[ids])
    ax[1][0].imshow(xhat)
    ax[1][0].axis("off")
    ax[1][0].set_title("Reconstructed\nTop {}".format(k))
    for j, ii in enumerate(ids):
        title = "Dictionary {}".format(ii)
        ax[j // c][j % c + 1].imshow(D[ii].reshape(*D.shape[-2:]))
        ax[j // c][j % c + 1].axis("off")
        ax[j // c][j % c + 1].set_title(title + "\nCoef = {:.3f}".format(xi[ii]))


def show_sparsity(x, eps=1e-3):
    xt = np.where(x < eps, 0, x)
    l0 = np.linalg.norm(xt, ord=0, axis=1)
    l1 = np.linalg.norm(xt, ord=1, axis=1)
    s1 = sci1(x)
    s5 = sci5(x)

    fig, ax = plt.subplots(2, 2, figsize=(15, 10))

    ax[0][0].plot(l0)
    ax[0][0].set_xlabel("Test Samples")
    ax[0][0].set_ylabel("L0 Norm of LASSO Coefs")
    ax[0][1].plot(l1)
    ax[0][1].set_xlabel("Test Samples")
    ax[0][1].set_ylabel("L1 Norm of LASSO Coefs")
    ax[1][0].plot(s1)
    ax[1][0].set_xlabel("Test Samples")
    ax[1][0].set_ylabel("SCI1")
    ax[1][1].plot(s5)
    ax[1][1].set_xlabel("Test Samples")
    ax[1][1].set_ylabel("SCI5")

    return s1, s5


TYPES = ["Pedestrian", "Cyclist", "Pets"]
CARRY_ONS = ["backpack", "shoulder-bag", "hand-bag", "luggage", "umbrella", "misc"]


def idof(y):
    return y[0]


def typeof(y):
    return TYPES[y[1]]


def is_occluded(y):
    return y[2] == 1


def carry_ons(y):
    c = y[3:]
    return ", ".join([CARRY_ONS[i] for i in range(len(c)) if c[i] == 1])


def show_topk2(x, D, Dl, y, yl, s, i, k):
    xi = x[i].numpy()
    ids = np.argsort(np.abs(xi))[::-1][:k]
    title = "{}{} ({})\n{}".format(
        typeof(yl[i]),
        idof(yl[i]),
        "OCC" if is_occluded(yl[i]) else "NOC",
        "carry-on: " + ("yes" if carry_ons(yl[i]) else "no"),
    )
    r = 2
    c = k // 2
    fig, ax = plt.subplots(r, c + 1, figsize=(c * 3, 9))
    ax[0][0].imshow(y[i].reshape(*D.shape[-2:]))
    ax[0][0].axis("off")
    ax[0][0].set_title(title + "\nSCI = {:.3f}".format(s[i]))
    xhat = np.dot(D[ids].transpose(1, 2, 0), xi[ids])
    ax[1][0].imshow(xhat)
    ax[1][0].axis("off")
    ax[1][0].set_title("Reconstructed\nTop {}".format(k))
    for j, ii in enumerate(ids):
        title = "{}{} ({})\n{}".format(
            typeof(Dl[ii]),
            idof(Dl[ii]),
            "OCC" if is_occluded(Dl[ii]) else "NOC",
            "carry-on: " + ("yes" if carry_ons(Dl[ii]) else "no"),
        )
        ax[j // c][j % c + 1].imshow(D[ii].reshape(*D.shape[-2:]))
        ax[j // c][j % c + 1].axis("off")
        ax[j // c][j % c + 1].set_title(title + "\nCoef = {:.3f}".format(xi[ii]))


def show_topd2(dict_, x, y, yl, s, i, k):
    xi = x[i].numpy()
    D = dict_.dict.weight.detach().cpu().numpy().reshape(-1, *y.shape[-2:])
    ids = np.argsort(np.abs(xi))[::-1][:k]
    title = "{}{} ({})\n{}".format(
        typeof(yl[i]),
        idof(yl[i]),
        "OCC" if is_occluded(yl[i]) else "NOC",
        "carry-on: " + ("yes" if carry_ons(yl[i]) else "no"),
    )
    r = 2
    c = k // 2
    fig, ax = plt.subplots(r, c + 1, figsize=(c * 3, 9))
    ax[0][0].imshow(y[i])
    ax[0][0].axis("off")
    ax[0][0].set_title(title + "\nSCI = {:.3f}".format(s[i]))
    xhat = np.dot(D[ids].transpose(1, 2, 0), xi[ids])
    ax[1][0].imshow(xhat)
    ax[1][0].axis("off")
    ax[1][0].set_title("Reconstructed\nTop {}".format(k))
    for j, ii in enumerate(ids):
        title = "Dictionary {}".format(ii)
        ax[j // c][j % c + 1].imshow(D[ii].reshape(*D.shape[-2:]))
        ax[j // c][j % c + 1].axis("off")
        ax[j // c][j % c + 1].set_title(title + "\nCoef = {:.3f}".format(xi[ii]))
