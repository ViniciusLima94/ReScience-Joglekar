import matplotlib
import numpy as np
import matplotlib.pyplot as plt


Nareas = 29
area_names = [
    "V1",
    "V2",
    "V4",
    "DP",
    "MT",
    "8m",
    "5",
    "8l",
    "TEO",
    "2",
    "F1",
    "STPc",
    "7A",
    "46d",
    "10",
    "9/46v",
    "9/46d",
    "F5",
    "TEpd",
    "PBr",
    "7m",
    "7B",
    "F2",
    "STPi",
    "PROm",
    "F7",
    "8B",
    "STPr",
    "24c",
]


def fig2b(t, Rweak, Rstrong):
    plt.plot(t / 1000, Rweak, "blue", lw=3, ls="--", label="weak LBA")
    plt.plot(t / 1000, Rstrong, "black", lw=3, label="strong LBA")
    plt.xlim([0, 0.6])
    plt.ylim([0, 6.0])
    plt.xlabel("Time (s)", fontsize=15)
    plt.ylabel("Excitatory rate (Hz)", fontsize=15)
    plt.legend(fontsize=15)
    plt.savefig("figures/fig2_b.pdf", dpi=600)
    plt.close()


def fig2c(Fmax, extent):
    Fmax[Fmax > 8.5] = np.nan
    cmap = matplotlib.cm.hot
    cmap.set_bad("lightgray", 1.0)
    plt.figure()
    plt.imshow(
        Fmax, aspect="auto", cmap=cmap, origin="lower", vmin=0, vmax=8.5,
        extent=extent
    )
    plt.colorbar()
    plt.ylabel("Local I to E coupling", fontsize=15)
    plt.xlabel("Local E to E coupling", fontsize=15)

    Wee = 6.00
    Wei = 6.70
    plt.plot(Wee, Wei, "black", marker="o", ms=10)
    Wee = 4.45
    Wei = 4.70
    plt.plot(Wee, Wei, "blue", marker="o", ms=10)

    plt.savefig("figures/fig2_c.pdf", dpi=600)
    plt.close()


def fig3b_d(tidx, r_w, max_freq_w, r_s, max_freq_s):
    area_plot = ["V1", "V4", "8m", "8l", "TEO", "7A", "9/46d", "TEpd", "24c"]
    area_indx = [0, 2, 5, 7, 8, 12, 16, 18, 28]
    sub_areas = np.zeros(29)
    for idx in area_indx:
        sub_areas[idx] = 1
    sub_areas = sub_areas.astype(int)
    sub_areas = sub_areas.astype(bool)

    Npop = 2
    Npop_total = Npop * Nareas

    ##########################################################################
    # Plotting rates
    ##########################################################################

    i_list = []
    count = 0
    size = Nareas
    scale = [i * size for i in range(size)]
    for i in range(Npop_total):
        if i % Npop == 0:
            i_list.append(i)
            count += 1

    from matplotlib.ticker import FormatStrFormatter

    plt.figure(figsize=(12, 6))
    use = np.unique(np.array(i_list) * sub_areas)
    size = len(use)
    scale = [i * size for i in range(size)]
    count = 0
    count2 = 0
    for i in use:
        ax = plt.subplot(len(use), 2, count + 1)
        plt.plot(tidx, r_w[i] - 10.0, "g")
        min_x = 0 - 0.1 * np.max(r_w[i] - 10.0)
        max_x = np.max(r_w[i] - 10.0) + 0.1 * np.max(r_w[i] - 10.0)
        plt.ylim([min_x, max_x])
        plt.yticks([min_x, max_x], [None, np.round(max_x, 3)])
        # ax.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
        ax.spines["bottom"].set_visible(False)
        if count + 1 == 1:
            plt.title("Weak GBA", fontsize=15)
            plt.plot([2000.0, 2250.0], [max_x + 5, max_x + 5], "k", lw="3")
            plt.text(2000, max_x + 12, "250 ms")
        plt.xticks([])
        plt.xlim([1750, 5000])
        count += 1
        ax = plt.subplot(len(use), 2, count + 1)
        plt.plot(tidx, r_w[i] - 10.0, "g")
        plt.plot(tidx, r_s[i] - 10.0, "m")
        min_x = 0 - 0.1 * np.max(r_s[i] - 10.0)
        max_x = np.max(r_s[i] - 10.0) + 0.1 * np.max(r_s[i] - 10.0)
        plt.ylim([min_x, max_x])
        plt.yticks([min_x, max_x], [None, np.round(max_x, 3)])
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
        ax.spines["bottom"].set_visible(False)
        ax.yaxis.set_label_position("right")
        plt.ylabel(area_plot[count2], rotation=0, fontsize=15)
        plt.xlim([1750, 5000])
        if count + 1 == 2:
            plt.title("Strong GBA", fontsize=15)
            plt.plot([2000.0, 2250.0], [max_x + 5, max_x + 5], "k", lw="3")
            plt.text(2000, max_x + 12, "250 ms")
        plt.xticks([])
        count += 1
        count2 += 1
    # plt.yticks(scale, area_plot)
    plt.tight_layout()
    # plt.show()
    plt.savefig("figures/fig3_c_d.pdf", dpi=600)
    plt.close()

    ax = plt.figure(figsize=(6, 3))
    plt.semilogy(range(Nareas), max_freq_w - 10, marker="o", c="g",
                 label="Weak GBA")
    plt.semilogy(range(Nareas), max_freq_s - 10, marker="o", c="m",
                 label="Strong GBA")
    plt.legend(fontsize=15)
    plt.xlim([-0.5, 30])
    # plt.ylim([1e-4, 1.5e2])
    plt.xticks(range(Nareas), area_names, rotation=90)
    plt.ylabel("Maximum firing rate [Hz]", fontsize=15)
    plt.xlabel("Areas", fontsize=15)
    plt.tight_layout()
    # plt.show()
    plt.savefig("figures/fig3_e.pdf", dpi=600)
    plt.close()


def fig3_f(muee_vec, max_r_24c_t, max_r_24c_f):
    c = ["b", "k"]
    l = ["Weak GBA", "Strong GBA"]
    l1 = [", Eq. (2) + (27)", ", Eq. (2)"]
    for uu in [0, 1]:
        plt.semilogy(muee_vec, max_r_24c_t[:, uu], "o-", c=c[uu])
        plt.semilogy(muee_vec, max_r_24c_f[:, uu], "--", c=c[uu])
    plt.xlabel("Global E to E coupling", fontsize=15)
    plt.ylabel("Maximum rate in 24c [Hz]", fontsize=15)
    plt.legend([l[0] + l1[0], l[0] + l1[1], l[1] + l1[0], l[1] + l1[1]],
               fontsize=15)
    plt.xlim([20, 52])
    plt.ylim([1e-8, 1e4])
    plt.savefig("figures/fig3_f.pdf", dpi=600)
    plt.close()


def plot_raster(times_ex, index_ex, reg, simtime, filename, save=False):
    NE = 1600  # Number of excitatory neurons in one population
    NI = 400  # Number of inhibitory neurons in one population

    # raster plot of spiking activitysimtime
    ind_ex = times_ex >= 0.0

    indexes = [2000] * 29
    indexes = np.cumsum(indexes)

    # plt.figure(figsize=(5,7))
    plt.plot(times_ex[ind_ex], index_ex[ind_ex], "b.", markersize=0.7,
             label="exc")

    for i in range(indexes.shape[0] - 1):
        plt.hlines(indexes[i] - NI + 1, 0, simtime, "k")

    # plt.ylabel('Neuron Index')
    plt.xlabel("Time [ms]", fontsize=15)
    #  plt.title('Raster Plot')
    plt.yticks(indexes - NI - 700, area_names)
    if reg == "async":
        plt.xlim(495, 730)
        plt.xticks([500, 700], [0, 200])
    elif reg == "sync":
        plt.xlim(500, 560)
        plt.xticks([500, 550], [0, 50])
    elif reg == None:
        plt.xlim(500, 800)
    plt.ylim([0, (NE + NI) * 29 - NI])
    if save == True:
        plt.savefig(filename, dpi=600)
        plt.close()


def fig4_5_6(t1, i1, t2, i2, mr1, mr2, reg, simtime, filename):
    plt.figure(figsize=(6, 8))
    plt.subplot2grid((6, 2), (0, 0), rowspan=4)
    plot_raster(t1, i1, reg, simtime, filename)
    plt.subplot2grid((6, 2), (0, 1), rowspan=4)
    plot_raster(t2, i2, reg, simtime, filename)
    plt.yticks([])
    plt.subplot2grid((6, 2), (4, 0), rowspan=2, colspan=2)
    plt.semilogy(range(Nareas), mr1, "o-", color="green", label="Weak GBA")
    plt.semilogy(range(Nareas), mr2, "o-", color="purple", label="Strong GBA")
    plt.ylabel("Maximum firing rate [Hz]", fontsize=15)
    plt.xticks(range(29), area_names, rotation=90)
    plt.tight_layout()
    plt.savefig(filename, dpi=600)
    plt.close()
