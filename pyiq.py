import collections.abc

import numpy as np
import scipy


def _promote_to_list(x: float | tuple[float], length: int) -> list:
    try:
        # check if is aready iterable
        _ = (_ for _ in x)
    except TypeError:
        # it is a single number
        x = [x for _ in range(length)]

    return x


def window_image(
    image: np.ndarray,
    window: collections.abc.Callable = scipy.signal.windows.hann,
    axis=(-2, -1),
    *args,
    **kwargs
) -> np.ndarray:
    """
    window is a callable that the takes the number of points as the first argument.
    args and kwargs are passed to window
    """
    res = image.copy()

    for i in axis:
        w = window(image.shape[i], *args, **kwargs)
        dshape = np.ones(len(image.shape), dtype=int)
        dshape[i] = -1
        w = w.reshape(dshape)

        res *= w

    return res


def powerspectrum(
    image: np.ndarray,
    pxsize: float | tuple[float] = (1, 1),
    window: collections.abc.Callable = scipy.signal.windows.hann,
    axis=(-2, -1),
):
    # ref. https://aapm.onlinelibrary.wiley.com/doi/pdf/10.1120/jacmp.v17i3.5841

    # remove the mean
    im = image - np.nanmean(image, axis=axis, keepdims=True)

    # window the image
    if window is not None:
        im = window_image(im, window, axis=axis)

    # power spectrum
    f = np.abs(np.fft.fftn(im, axes=axis)) ** 2

    # normalisation
    pxsize = _promote_to_list(pxsize, len(axis))
    for i, s in zip(axis, pxsize):
        f *= s / image.shape[i]

    # put zero frequency in the centre
    ps = np.fft.fftshift(f, axes=axis)

    return ps


def psfreq(
    shape: tuple[int],
    pxsize: float | tuple[float] = (1, 1),
    axis=(-2, -1),
) -> list[np.ndarray]:
    pxsize = _promote_to_list(pxsize, len(axis))
    freq = [np.fft.fftshift(np.fft.fftfreq(shape[i], s)) for i, s in zip(axis, pxsize)]

    return freq


def azimuthal_average(
    image: np.ndarray,
    coords: np.ndarray = None,
    binspacing: float = None,
    rmax: float = None,
    rbins: np.ndarray = None,
    center: np.ndarray = None,
):
    if coords is None:
        coords = np.indices(image.shape)
        if center is None:
            center = (np.array(image.shape) - 1) / 2
        if binspacing is None:
            binspacing = 1
    else:
        coords = np.array(np.meshgrid(*coords))
        if center is None:
            center = np.array([(c[-1] - c[0]) / 2 for c in coords])
        if binspacing is None:
            binspacing = np.max(np.array([c[1] - c[0] for c in coords]))

    coords = coords - np.reshape(center, (-1,) + image.ndim * (1,))
    r = np.sqrt(np.sum(coords**2, axis=0))

    if rbins is None:
        if rmax is None:
            rmax = np.min(
                [np.max(np.abs(coords[i, ...])) for i in range(coords.shape[0])]
            )
            # rmax = np.min(np.r_[center, np.array(image.shape) - 1 - center])
        rbins = np.arange(0, rmax, binspacing)

    rbinned, bin_edges, _ = scipy.stats.binned_statistic(
        r.ravel(), image.ravel(), bins=rbins
    )

    return bin_edges[:-1], rbinned


def task_disc(
    shape: tuple[int],
    radius: float,
    value: float = 1,
):
    indices = np.indices(shape)
    center = (np.array(shape) - 1) / 2
    coords = indices - np.reshape(center, (-1,) + len(shape) * (1,))
    r = np.sqrt(np.sum(coords**2, axis=0))
    selection = r <= radius
    w = np.zeros(shape)
    w[selection] = value

    return w


def dprime(taskps: np.ndarray, nps: np.ndarray, ttf: np.ndarray = 1, axis=None):
    a = taskps * ttf**2
    dp = np.sqrt(np.sum(a, axis=axis) ** 2 / np.sum(a * nps, axis=axis))

    return dp
