import numpy as np

from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvas
from scipy.optimize import curve_fit
from tqdm import tqdm


def f(x, a, b):
    return a * x + b


def propagate_sim(n_tracks=10000, input_res=1., beam_width=1200, slope_width=0.25):
    # all values in um, slope in mRad

    # number of planes
    n_planes = 8

    # res is pixel resolution
    res = np.array([input_res]*n_planes)
    res[-2:] *= 0.4

    z = [5200., 30000.0, 57000.0, 84000, 111000, 138000, 1e6, 4.5e6]
    # z = [5200., 105200.0, 205200.0, 305200, 405200, 505200, 605200, 705200000]

    # intialize MC
    x_hit = np.zeros(shape=(n_tracks, n_planes))
    tracks_x_seed = np.random.normal(0, beam_width, n_tracks)
    tracks_x_slope = np.random.normal(0, slope_width * 1e-3, n_tracks)
    reco_sigma = np.random.normal(0,res[-3]/np.sqrt(12)*1000, n_tracks)

    # prepare track storage
    tracks_x = np.zeros(shape=(n_tracks, n_planes))
    residuals_second = []
    residuals_last = []

    # draw smeared hit in all planes, also fit track seed for each plane
    for i in tqdm(range(n_tracks)):
        for plane in range(n_planes):
            # create discretized hits following the track
            # x_hit[i, plane] = np.random.normal(f(z[plane], tracks_x_slope[i], tracks_x_seed[i]), res[plane] / np.sqrt(12)) // res[plane]
            x_hit[i, plane] = f(z[plane], tracks_x_slope[i], tracks_x_seed[i])# // res[plane]
            x_hit[i, plane] = np.around(x_hit[i, plane] - 0.5) + 1
            x_hit[i, plane] *= res[plane]
            
            # propagate seed x and x_slope to planes 1 - 6
            if plane < (n_planes-2):
                tracks_x[i, plane] = f(z[plane], tracks_x_slope[i], tracks_x_seed[i])
            else:
                x_hit[i, plane] = f(z[plane]*1.02, tracks_x_slope[i], tracks_x_seed[i]) + np.random.normal(0,res[plane]) #// res[plane]
        # reconstruct track fit straight line, use first 4 planes
        popt, pcov = curve_fit(f, z[:-2], tracks_x[i, :-2], sigma=np.ones_like(tracks_x[i, :-2]) * reco_sigma[i], absolute_sigma=True)
        # only project track for last 2 planes, use fit from first 6 planes
        tracks_x[i, -2] = f(z[-2], *popt)
        tracks_x[i, -1] = f(z[-1], *popt)
        # calculate residuals
        residuals_second.append(tracks_x[i, -2] - x_hit[i, -2])
        residuals_last.append(tracks_x[i, -1] - x_hit[i, -1])

    plt.hist(residuals_last, bins = 50)
    plt.xlabel("x residuum")
    plt.title("residuals on last plane - %.f $\mu m$ pixel pitch" % res[-1])
    plt.savefig("./%.f_um_residuals.png" % input_res)
    plt.cla()

    plt.hist(tracks_x[:,-3], bins = 50)
    plt.xlabel("track x")
    plt.title("tracks on last fitted plane - %.f $\mu m$ pixel pitch" % res[-3])
    plt.savefig("./%.f_um_tracks_fitted.png" % input_res)
    plt.cla()

    plt.hist(tracks_x[:,-1], bins = 50)
    plt.xlabel("track x")
    plt.title("tracks on last plane - %.f $\mu m$ pixel pitch" % res[-1])
    plt.savefig("./%.f_um_tracks.png" % input_res)
    plt.cla()

    plt.hist(x_hit[:,-1], bins = 50)
    plt.xlabel("x")
    plt.title("hits on last plane - %.f $\mu m$ pixel pitch" % res[-1])
    plt.savefig("./%.f_um_hits.png" % input_res)
    plt.cla()

    plt.plot(tracks_x_slope*1e3, np.array(residuals_last), linestyle="None", marker="o", label = "%.f $\mu m$" % res[-1])
    plt.legend()
    plt.xlabel("x slope [mRad]")
    plt.ylabel("residuals [$\mu m$]")
    plt.savefig("./%.f_um_residuals_2d.png" % input_res)
    plt.cla()

    return tracks_x_slope * 1e3, np.array(residuals_second), np.array(residuals_last)


if __name__ == '__main__':

    fig = Figure()
    _ = FigureCanvas(fig)
    ax = fig.add_subplot(111)

    for resolution in [250., 50., 10., 2., 1.]:
        result = propagate_sim(n_tracks=10000, input_res=resolution)
        ax.plot(result[0], result[2], linestyle="None", marker="o", label = "%.f $\mu m$" % resolution)
        ax.legend()
        ax.set_title("residuals last plane")
        ax.set_xlabel("x slope [mRad]")
        ax.set_ylabel("residuals [$\mu m$]")

    fig.savefig("./all_residuals_2d.png")
