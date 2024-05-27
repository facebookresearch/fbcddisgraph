#!/usr/bin/env python3

"""
Copyright (c) Facebook, Inc. and its affiliates.

Plots of the differences between two subpopulations with disjoint scores

*
This implementation considers only responses r that are restricted to taking
values 0 or 1 (Bernoulli variates).
*

Functions
---------
cumulative
    Cumulative difference between observations from two disjoint subpopulations
equiscores
    Reliability diagram with roughly equispaced average scores over bins
equierrs
    Reliability diagram with similar ratio L2-norm / L1-norm of weights by bin
exactplot
    Reliability diagram with exact values plotted

This source code is licensed under the MIT license found in the LICENSE file in
the root directory of this source tree.
"""


import math
import os
import subprocess
import random
import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedFormatter


def cumulative(r, s, majorticks, minorticks, probs=False,
               filename='cumulative.pdf',
               title='subpop. deviation is the slope as a function of $A_k$',
               fraction=1, weights=None):
    """
    Cumulative difference between observations from two disjoint subpops.

    Saves a plot of the difference between the normalized cumulative sums of r
    for one subpopulation and the normalized cumulative sums of r for a second
    subpopulation whose scores are all distinct from the scores for the first,
    with majorticks major ticks and minorticks minor ticks on the lower axis,
    labeling the major ticks with the corresponding values from s.

    Parameters
    ----------
    r : list
        list of array_like values of random outcomes
    s : list
        list of array_like scores
        (each array must be in strictly increasing order)
    majorticks : int
        number of major ticks on the lower axis
    minorticks : int
        number of minor ticks on the lower axis
    probs : bool, optional
        set to True if the scores are the probabilities of success
        for Bernoulli variates; set to False (the default) to bound
        the variance of a Bernoulli variate by p(1-p) <= 1/4, as appropriate
        when p is unknown
    filename : string, optional
        name of the file in which to save the plot
    title : string, optional
        title of the plot
    fraction : float, optional
        proportion of the full horizontal axis to display
    weights : list
        list of array_like weights of the observations
        (the default None results in equal weighting)

    Returns
    -------
    float
        Kuiper statistic
    float
        Kolmogorov-Smirnov statistic
    float
        quarter of the full height of the isosceles triangle
        at the origin in the plot
    int
        length of the cumulative sequence
    """

    def mergesorted(a, b):
        # Combines into a sorted array c the sorted arrays a and b,
        # with tags in an array d as to which of arrays a and b
        # the corresponding entries in c originated (0 for a and 1 for b), and
        # with the corresponding index to either a or b stored in an array e.
        c = np.zeros((a.size + b.size))
        d = np.zeros((a.size + b.size), dtype=np.int32)
        e = np.zeros((a.size + b.size), dtype=np.int32)
        ia = 0
        ib = 0
        for k in range(c.size):
            if ia == a.size:
                c[k] = b[ib]
                d[k] = 1
                e[k] = ib
                ib += 1
            elif ib == b.size:
                c[k] = a[ia]
                d[k] = 0
                e[k] = ia
                ia += 1
            elif a[ia] < b[ib]:
                c[k] = a[ia]
                d[k] = 0
                e[k] = ia
                ia += 1
            else:
                c[k] = b[ib]
                d[k] = 1
                e[k] = ib
                ib += 1
        return c, d, e

    def binvalues(a, d, e, key, w):
        # Bins into an array b the weighted average of values in the array a,
        # based on the tags in array d which match key,
        # with a[e[k]] corresponding to d[k] for k in range(d.size);
        # each bin corresponds to a continguous block of values in d that
        # match key. The weights come from w (specifically, w[key][e[k]]
        # corresponds to d[k] for k in range(d.size); key=0 or key=1).
        winbin = 0
        b = [0]
        for k in range(d.size):
            if d[k] == key:
                winbin += w[key][e[k]]
                b[-1] += a[e[k]] * w[key][e[k]]
            elif winbin > 0:
                b[-1] /= winbin
                winbin = 0
                b.append(0)
        if d[-1] == key:
            b[-1] /= winbin
        else:
            del b[-1]
        return np.array(b)

    def aggvalues(a, d, e, key):
        # Bins into an array b the average of values in the array a,
        # based on the tags in array d which match key,
        # with a[e[k]] corresponding to d[k] for k in range(d.size);
        # each bin corresponds to a continguous block of values in d that
        # match key.
        inbin = 0
        b = [0]
        for k in range(d.size):
            if d[k] == key:
                inbin += 1
                b[-1] += a[e[k]]
            elif inbin > 0:
                b[-1] /= inbin
                inbin = 0
                b.append(0)
        if d[-1] == key:
            b[-1] /= inbin
        else:
            del b[-1]
        return np.array(b)

    def histcounts(nbins, x):
        # Counts the number of entries of x
        # falling into each of nbins equispaced bins.
        j = 0
        nbin = np.zeros(nbins, dtype=np.int64)
        for k in range(len(x)):
            if x[k] > x[0] + (x[-1] - x[0]) * (j + 1) / nbins:
                j += 1
            if j == nbins:
                break
            nbin[j] += 1
        return nbin

    # Check that the arrays in s are both sorted.
    for j in range(2):
        assert all(s[j][k] < s[j][k + 1] for k in range(len(s[j]) - 1))
    # Check that the arrays in s are disjoint.
    s01 = np.concatenate((s[0], s[1]))
    assert np.unique(s01).size == s[0].size + s[1].size
    # Determine the weighting scheme.
    if weights is None:
        w = []
        for j in range(2):
            w.append(np.ones((n[j])))
    else:
        w = weights.copy()
    for j in range(2):
        assert np.all(w[j] > 0)
    wtot = 0
    for j in range(2):
        wtot += w[j].sum()
    for j in range(2):
        w[j] /= wtot
    # Create the figure.
    plt.figure()
    ax = plt.axes()
    # Merge the scores from the subpopulations.
    _, d, e = mergesorted(s[0], s[1])
    # Bin the responses according to the merged scores.
    b = []
    for j in range(2):
        b.append(binvalues(r[j], d, e, j, w))
    # Bin the scores according to the merged scores.
    t = []
    for j in range(2):
        t.append(binvalues(s[j], d, e, j, w))
    # Stagger the scores from the two subpopulations into t01.
    t01 = np.zeros((t[0].size + t[1].size))
    for k in range(t01.size):
        if k % 2 == 0:
            t01[k] = t[d[0]][k // 2]
        else:
            t01[k] = t[1 - d[0]][k // 2]
    assert all(t01[k] < t01[k + 1] for k in range(len(t01) - 1))
    # Aggregate the weights according to the merged scores.
    w2 = []
    for j in range(2):
        w2.append(aggvalues(w[j], d, e, j))
    # Accumulate and average the associated aggregations of differences.
    la0 = min(len(b[d[0]]) - 1, len(b[1 - d[0]]))
    la1 = min(len(b[d[0]]) - 1, len(b[1 - d[0]]) - 1)
    a0 = np.zeros((la0))
    w0 = np.zeros((la0))
    for k in range(len(a0)):
        a0[k] = b[d[0]][k] - b[1 - d[0]][k]
        a0[k] += b[d[0]][k + 1] - b[1 - d[0]][k]
        a0[k] /= 2
        w0[k] = w2[d[0]][k] + 2 * w2[1 - d[0]][k] + w2[d[0]][k + 1]
        w0[k] /= 4
    a1 = np.zeros((la1))
    w1 = np.zeros((la1))
    for k in range(len(a1)):
        a1[k] = b[d[0]][k + 1] - b[1 - d[0]][k]
        a1[k] += b[d[0]][k + 1] - b[1 - d[0]][k + 1]
        a1[k] /= 2
        w1[k] = w2[1 - d[0]][k] + 2 * w2[d[0]][k + 1] + w2[1 - d[0]][k + 1]
        w1[k] /= 4
    a = np.zeros((la0 + la1))
    for k in range(len(a)):
        if k % 2 == 0:
            a[k] = a0[k // 2]
        else:
            a[k] = a1[k // 2]
    # Ensure that subpopulation 1 gets subtracted from subpopulation 0,
    # rather than the reverse.
    if d[0] == 1:
        a = -a
    # Stagger the weights from the two subpopulations into w01.
    w01 = np.zeros((la0 + la1))
    for k in range(len(w01)):
        if k % 2 == 0:
            w01[k] = w0[k // 2]
        else:
            w01[k] = w1[k // 2]
    w01sub = w01[:int(len(w01) * fraction)]
    w01sub /= w01[:int(len(w01) * fraction)].sum()
    aa = np.cumsum(a[:int(len(a) * fraction)] * w01sub)
    # Accumulate the weights.
    abscissae = np.cumsum(w01sub)
    # Plot the cumulative differences.
    plt.plot(abscissae, aa, 'k')
    # Make sure the plot includes the origin.
    plt.plot(0, 'k')
    # Add an indicator of the scale of 1/sqrt(n) to the vertical axis.
    t01sub = t01[:int(len(w01) * fraction)]
    if probs:
        lenscale = np.sqrt(np.sum(w01sub**2 * t01sub * (1 - t01sub)))
    else:
        lenscale = np.sqrt(np.sum(w01sub**2 / 4))
    # Adjust lenscale for the dependence between even and odd entries of t01.
    lenscale *= math.sqrt(2)
    # Adjust lenscale for taking the difference of 2 independent distributions
    # (one for each subpopulation).
    lenscale *= math.sqrt(2)
    plt.plot(2 * lenscale, 'k')
    plt.plot(-2 * lenscale, 'k')
    kwargs = {
        'head_length': 2 * lenscale, 'head_width': fraction / 20,
        'width': 0, 'linewidth': 0, 'length_includes_head': True, 'color': 'k'}
    plt.arrow(.1e-100, -2 * lenscale, 0, 4 * lenscale, shape='left', **kwargs)
    plt.arrow(.1e-100, 2 * lenscale, 0, -4 * lenscale, shape='right', **kwargs)
    plt.margins(x=0, y=.1)
    # Label the major ticks of the lower axis with the values of t01sub.
    sl = [
        '{:.2f}'.format(x)
        for x in t01sub[::(len(t01sub) // majorticks)].tolist()]
    plt.xticks(abscissae[:len(abscissae):(len(abscissae) // majorticks)][
        :majorticks], sl[:majorticks])
    if len(t01sub) >= 300 and minorticks >= 50:
        # Indicate the distribution of t01 via unlabeled minor ticks.
        plt.minorticks_on()
        ax.tick_params(which='minor', axis='x')
        ax.tick_params(which='minor', axis='y', left=False)
        ax.set_xticks(np.insert(abscissae, 0, [0])[
            np.cumsum(histcounts(minorticks, t01sub))], minor=True)
    # Label the axes.
    plt.xlabel('score ($S^0_{(k-1)/2}$ or $S^1_{(k-2)/2}$)')
    plt.ylabel('$C_k$')
    ax2 = plt.twiny()
    plt.xlabel(
        '$k/n$ (together with minor ticks at equispaced values of $A_k$)')
    ax2.tick_params(which='minor', axis='x', top=True, direction='in', pad=-17)
    ax2.set_xticks(np.arange(1 / majorticks, 1, 1 / majorticks), minor=True)
    ks = ['{:.2f}'.format(x) for x in
          np.arange(0, 1 + 1 / majorticks, 1 / majorticks).tolist()]
    alist = np.arange(0, 1 + 1 / majorticks, 1 / majorticks)
    alist *= len(abscissae) - 1
    alist = alist.tolist()
    # Jitter minor ticks that overlap with major ticks lest Pyplot omit them.
    alabs = []
    for x in alist:
        multiple = abscissae[int(x)] * majorticks
        if abs(multiple - round(multiple)) > 1e-3:
            alabs.append(abscissae[int(x)])
        else:
            alabs.append(abscissae[int(x)] * (1 + 1e-3))
    plt.xticks(alabs, ks)
    ax2.xaxis.set_minor_formatter(FixedFormatter(
        [r'$A_k\!=\!{:.2f}$'.format(1 / majorticks)]
        + [r'${:.2f}$'.format(k / majorticks) for k in range(2, majorticks)]))
    # Title the plot.
    plt.title(title)
    # Clean up the whitespace in the plot.
    plt.tight_layout()
    # Save the plot.
    plt.savefig(filename, bbox_inches='tight')
    plt.close()
    # Calculate summary statistics.
    aa0 = np.insert(aa, 0, [0])
    kuiper = np.max(aa0) - np.min(aa0)
    kolmogorov_smirnov = np.max(np.abs(aa))
    return kuiper, kolmogorov_smirnov, lenscale, len(t01sub)


def equiscores(r, s, nbins, filename='equiscores.pdf', weights=None,
               top=None, left=None, right=None):
    """
    Reliability diagram with roughly equispaced average scores over bins

    Plots a reliability diagram with roughly equispaced weighted average scores
    for the bins, for the two subpopulations whose responses are in r and
    whose scores are in s.

    Parameters
    ----------
    r : list
        list of array_like values of random outcomes
    s : list
        list of array_like scores (each array must be in non-decreasing order)
    nbins : int
        number of bins
    filename : string, optional
        name of the file in which to save the plot
    weights : list
        list of array_like weights of the observations
        (the default None results in equal weighting)
    top : float, optional
        top of the range of the vertical axis (the default None is adaptive)
    left : float, optional
        leftmost value of the horizontal axis (the default None is adaptive)
    right : float, optional
        rightmost value of the horizontal axis (the default None is adaptive)

    Returns
    -------
    None
    """

    def bintwo(nbins, a, b, q, qmax, w):
        # Determines the total weight of entries of q falling into each
        # of nbins equispaced bins, and calculates the weighted average per bin
        # of the arrays a and b, returning np.nan as the "average"
        # for any bin that is empty.
        j = 0
        bina = np.zeros(nbins)
        binb = np.zeros(nbins)
        wbin = np.zeros(nbins)
        for k in range(len(q)):
            if q[k] > qmax * (j + 1) / nbins:
                j += 1
            if j == nbins:
                break
            bina[j] += w[k] * a[k]
            binb[j] += w[k] * b[k]
            wbin[j] += w[k]
        # Normalize the sum for each bin to compute the weighted average.
        bina = np.divide(bina, wbin, where=wbin != 0)
        bina[np.where(wbin == 0)] = np.nan
        binb = np.divide(binb, wbin, where=wbin != 0)
        binb[np.where(wbin == 0)] = np.nan
        return wbin, bina, binb

    # Check that the arrays in s are both sorted.
    for j in range(2):
        assert all(s[j][k] <= s[j][k + 1] for k in range(len(s[j]) - 1))
    # Determine the weighting scheme.
    if weights is None:
        w = []
        for j in range(2):
            w.append(np.ones((n[j])))
    else:
        w = weights.copy()
    for j in range(2):
        assert np.all(w[j] > 0)
    wtot = 0
    for j in range(2):
        wtot += w[j].sum()
    for j in range(2):
        w[j] /= wtot
    # Create the figure.
    plt.figure()
    colors = ['black', 'gray']
    smin = 1e20
    smax = -1e20
    for j in range(1, -1, -1):
        _, binr, bins = bintwo(nbins, r[j], s[j], s[j], s[j][-1], w[j])
        smin = min(smin, s[j][0])
        smax = max(smax, s[j][-1])
        plt.plot(bins, binr, '*:', color=colors[j])
    xmin = smin if left is None else left
    xmax = smax if right is None else right
    plt.xlim((xmin, xmax))
    plt.ylim(bottom=0)
    plt.ylim(top=top)
    plt.xlabel('weighted average of scores in the bin')
    plt.ylabel('weighted average of responses in the bin')
    plt.title('reliability diagram')
    plt.tight_layout()
    plt.savefig(filename, bbox_inches='tight')
    plt.close()


def equierrs(r, s, nbins, filename='equierrs.pdf', weights=None,
             top=None, left=None, right=None):
    """
    Reliability diagram with similar ratio L2-norm / L1-norm of weights by bin

    Plots a reliability diagram with the ratio of the L2 norm of the weights
    to the L1 norm of the weights being roughly the same for every bin.
    The L2 norm is the square root of the sum of the squares, while the L1 norm
    is the sum of the absolute values (of course, weights are positive,
    so the absolute value does nothing special). The plot includes a graph
    for each of the two subpopulations whose responses are in r and
    whose scores are in s.

    Parameters
    ----------
    r : list
        list of array_like values of random outcomes
    s : list
        list of array_like scores (each array must be in non-decreasing order)
    nbins : int
        number of bins
    filename : string, optional
        name of the file in which to save the plot
    weights : list
        list of array_like weights of the observations
        (the default None results in equal weighting)
    top : float, optional
        top of the range of the vertical axis (the default None is adaptive)
    left : float, optional
        leftmost value of the horizontal axis (the default None is adaptive)
    right : float, optional
        rightmost value of the horizontal axis (the default None is adaptive)

    Returns
    -------
    list of int
        number of bins constructed for each subpopulation
    """

    def inbintwo(a, b, inbin, w):
        # Determines the total weight falling into the bins given by inbin,
        # and calculates the weighted average per bin of the arrays a and b,
        # returning np.nan as the "average" for any bin that is empty.
        wbin = [w[inbin[k]:inbin[k + 1]].sum() for k in range(len(inbin) - 1)]
        bina = [(w[inbin[k]:inbin[k + 1]] * a[inbin[k]:inbin[k + 1]]).sum()
                for k in range(len(inbin) - 1)]
        binb = [(w[inbin[k]:inbin[k + 1]] * b[inbin[k]:inbin[k + 1]]).sum()
                for k in range(len(inbin) - 1)]
        # Normalize the sum for each bin to compute the weighted average.
        bina = np.divide(bina, wbin, where=wbin != 0)
        bina[np.where(wbin == 0)] = np.nan
        binb = np.divide(binb, wbin, where=wbin != 0)
        binb[np.where(wbin == 0)] = np.nan
        return wbin, bina, binb

    def binbounds(nbins, w):
        # Partitions w into around nbins bins, each with roughly equal ratio
        # of the L2 norm of w in the bin to the L1 norm of w in the bin,
        # returning the indices defining the bins in the list inbin.
        proxy = len(w) // nbins
        v = w[np.sort(np.random.permutation(len(w))[:proxy])]
        # t is a heuristic threshold.
        t = np.square(v).sum() / v.sum()**2
        inbin = []
        k = 0
        while k < len(w) - 1:
            inbin.append(k)
            k += 1
            s = w[k]
            ss = w[k]**2
            while ss / s**2 > t and k < len(w) - 1:
                k += 1
                s += w[k]
                ss += w[k]**2
        if len(w) - inbin[-1] < (inbin[-1] - inbin[-2]) / 2:
            inbin[-1] = len(w)
        else:
            inbin.append(len(w))
        return inbin

    # Check that the arrays in s are both sorted.
    for j in range(2):
        assert all(s[j][k] <= s[j][k + 1] for k in range(len(s[j]) - 1))
    # Determine the weighting scheme.
    if weights is None:
        w = []
        for j in range(2):
            w.append(np.ones((n[j])))
    else:
        w = weights.copy()
    for j in range(2):
        assert np.all(w[j] > 0)
    wtot = 0
    for j in range(2):
        wtot += w[j].sum()
    for j in range(2):
        w[j] /= wtot
    # Create the figure.
    plt.figure()
    colors = ['black', 'gray']
    inbin = []
    for j in range(2):
        inbin.append(binbounds(nbins, w[j]))
    binsmin = 1e20
    binsmax = -1e20
    for j in range(1, -1, -1):
        _, binr, bins = inbintwo(r[j], s[j], inbin[j], w[j])
        binsmax = max(binsmax, np.max(bins))
        binsmin = min(binsmin, np.min(bins))
        plt.plot(bins, binr, '*:', color=colors[j])
    xmin = binsmin if left is None else left
    xmax = binsmax if right is None else right
    plt.xlim((xmin, xmax))
    plt.ylim(bottom=0)
    plt.ylim(top=top)
    plt.xlabel('weighted average of scores in the bin')
    plt.ylabel('weighted average of responses in the bin')
    title = r'reliability diagram'
    title += r' ($\Vert W \Vert_2 / \Vert W \Vert_1$ is similar for every bin)'
    plt.title(title)
    plt.tight_layout()
    plt.savefig(filename, bbox_inches='tight')
    plt.close()

    return [len(inbin[j]) - 1 for j in range(2)]


def exactplot(r, s, filename='exact.pdf', title='exact expectations',
              top=None, left=None, right=None):
    """
    Reliability diagram with exact values plotted

    Plots a reliability diagram at full resolution with fractional numbers,
    for the two subpopulations whose responses are in r and whose scores
    are in s. The entries of the members of r should be the expected values
    of outcomes, even if the outcomes are integer counts or just 0s and 1s.

    Parameters
    ----------
    r : list
        list of array_like expected values of class labels
    s : list
        list of array_like scores (each array must be in non-decreasing order)
    filename : string, optional
        name of the file in which to save the plot
    title : string, optional
        title of the plot
    top : float, optional
        top of the range of the vertical axis (the default None is adaptive)
    left : float, optional
        leftmost value of the horizontal axis (the default None is adaptive)
    right : float, optional
        rightmost value of the horizontal axis (the default None is adaptive)

    Returns
    -------
    None
    """
    # Check that the arrays in s are both sorted.
    for j in range(2):
        assert all(s[j][k] <= s[j][k + 1] for k in range(len(s[j]) - 1))
    # Create the figure.
    plt.figure()
    colors = ['black', 'gray']
    for j in range(1, -1, -1):
        plt.plot(s[j], r[j], '*', color=colors[j])
    plt.xlim((left, right))
    plt.ylim(bottom=0)
    plt.ylim(top=top)
    plt.xlabel('score')
    plt.ylabel('expected value of the response')
    plt.title(title)
    plt.tight_layout()
    plt.savefig(filename, bbox_inches='tight')
    plt.close()


if __name__ == '__main__':
    #
    # Generate directories with plots as specified via the code below,
    # with each directory named weighted/n[0]_n[1]_nbins_iex
    # (where n, nbins, and iex are defined in the code below).
    #
    # Set parameters.
    # minorticks is the number of minor ticks on the lower axis.
    minorticks = 100
    # majorticks is the number of major ticks on the lower axis.
    majorticks = 10
    # n is a list of the numbers of members for the subpopulations.
    n = [10000, 7000]
    # Consider an example; iex is the number of the example.
    for iex in range(4):
        # nbins is the number of bins for the reliability diagrams.
        for nbins in [10, 50]:
            # nbins must divide evenly every entry of n to pass the following.
            for m in n:
                assert m % nbins == 0

            if iex == 0:
                # Construct scores for the subpopulations.
                np.random.seed(987654321)
                s = []
                for j in range(2):
                    s.append(np.random.uniform(size=(n[j])))
                    if j == 1:
                        s[j] = (1 + (s[j] - .5)**3 / .5**3) / 2
                    else:
                        s[j] = (1 + (s[j] - .5) / .5) / 2
                    # The scores must be in increasing order.
                    s[j] = np.sort(s[j])
                # Construct the exact sampling probabilities.
                exact = []
                a = [.2, 1]
                for j in range(2):
                    exact.append(
                        a[j] * (s[j] - .5) - .7 * (s[j] - .75)**(j + 2) + .5)
                # Swap some outcomes in a "range" from start - width
                # to start + width.
                start = .9
                width = .06
                for k in range(len(s[1])):
                    if start - s[1][k] < width:
                        k0 = k
                        break
                for k in range(k0, len(s[1])):
                    if s[1][k] - start > width:
                        break
                    else:
                        ind = k - k0 + 9 * len(exact[0]) // 10
                        # Swap exact[1][k] and exact[0][ind].
                        t = exact[1][k]
                        exact[1][k] = exact[0][ind]
                        exact[0][ind] = t
                # Construct weights.
                weights = []
                for j in range(2):
                    weights.append(4 - np.cos(9 * np.arange(n[j]) / n[j]))

            if iex == 1:
                # Construct scores for the subpopulations.
                np.random.seed(987654321)
                s = []
                for j in range(2):
                    s.append(np.random.uniform(size=(n[j])))
                    if j == 0:
                        s[j] = s[j] ** 5
                    # The scores must be in increasing order.
                    s[j] = np.sort(s[j])
                # Construct the exact sampling probabilities.
                exact = []
                for j in range(2):
                    a = math.sqrt(1 / 2)
                    b = np.arange(-a, a - .1e-10, 2 * a / n[j]) - a / n[j]
                    ex = 1 + np.round(np.sin(5.5 * np.arange((n[j])) / n[j]))
                    ex /= 2
                    ex *= np.square(b) - a**2
                    ex += s[j]
                    ex = np.abs(ex)
                    exact.append(ex)
                # Construct weights.
                weights = []
                for j in range(2):
                    weights.append(4 - np.cos(9 * np.arange(n[j]) / n[j]))

            if iex == 2:
                # Construct scores for the subpopulations.
                np.random.seed(987654321)
                s = []
                for j in range(2):
                    s.append(np.random.uniform(size=(n[j])))
                    if j == 1:
                        s[j] = 1 + np.cbrt(s[j] - .5) / np.cbrt(.5)
                        s[j] /= 2
                    else:
                        s[j] = (1 + (s[j] - .5) / .5) / 2
                    # The scores must be in increasing order.
                    s[j] = np.sort(s[j])
                # Construct the exact sampling probabilities.
                exact = [s[0] * (1 + np.cos(16 * math.pi * s[0])) / 2]
                exact.append(np.random.uniform(size=(n[1])))
                # Construct weights.
                weights = []
                for j in range(2):
                    weights.append(4 - np.cos(9 * np.arange(n[j]) / n[j]))

            if iex == 3:
                # Construct scores for the subpopulations.
                np.random.seed(987654321)
                s = []
                for j in range(2):
                    s.append(np.random.uniform(size=(n[j])))
                    if j == 1:
                        s[j] = (1 + (s[j] - .5)**3 / .5**3) / 2
                    else:
                        s[j] = (1 + (s[j] - .5) / .5) / 2
                    # The scores must be in increasing order.
                    s[j] = np.sort(s[j])
                # Construct the exact sampling probabilities.
                exact = []
                for j in range(2):
                    exact.append(s[j])
                # Construct weights.
                weights = []
                for j in range(2):
                    weights.append(4 - np.cos(9 * np.arange(n[j]) / n[j]))

            # Set a unique directory for each collection of experiments
            # (creating the directory if necessary).
            dir = 'weighted'
            try:
                os.mkdir(dir)
            except FileExistsError:
                pass
            dir = dir + '/'
            for j in range(2):
                dir = dir + str(n[j]) + '_'
            dir = dir + str(nbins) + '_'
            dir = dir + str(iex)
            try:
                os.mkdir(dir)
            except FileExistsError:
                pass
            dir = dir + '/'
            print(f'./{dir} is under construction....')

            # Generate a sample of classifications into two classes,
            # correct (class 1) and incorrect (class 0),
            # avoiding numpy's random number generators
            # that are based on random bits --
            # they yield strange results for many seeds.
            random.seed(987654321)
            r = []
            for j in range(2):
                uniform = np.asarray([random.random() for _ in range(n[j])])
                r.append((uniform <= exact[j]).astype(float))

            # Generate five plots and a text file reporting metrics.
            filename = dir + 'cumulative.pdf'
            kuiper, kolmogorov_smirnov, lenscale, lencums = cumulative(
                r, s, majorticks, minorticks, False, filename, weights=weights)
            filename = dir + 'metrics.txt'
            with open(filename, 'w') as f:
                f.write('n:\n')
                f.write(f'{lencums}\n')
                f.write('n[0]:\n')
                f.write(f'{n[0]}\n')
                f.write('n[1]:\n')
                f.write(f'{n[1]}\n')
                f.write('lenscale:\n')
                f.write(f'{lenscale}\n')
                f.write('Kuiper:\n')
                f.write(f'{kuiper:.4}\n')
                f.write('Kolmogorov-Smirnov:\n')
                f.write(f'{kolmogorov_smirnov:.4}\n')
                f.write('Kuiper / lenscale:\n')
                f.write(f'{(kuiper / lenscale):.4}\n')
                f.write('Kolmogorov-Smirnov / lenscale:\n')
                f.write(f'{(kolmogorov_smirnov / lenscale):.4}\n')
            filename = dir + 'cumulative_exact.pdf'
            _, _, _, _ = cumulative(
                exact, s, majorticks, minorticks, False, filename,
                title='exact expectations', weights=weights)
            filename = dir + 'equiscores.pdf'
            equiscores(r, s, nbins, filename, weights=weights)
            filename = dir + 'equierrs.pdf'
            equierrs(r, s, nbins, filename, weights=weights)
            filename = dir + 'exact.pdf'
            exactplot(exact, s, filename)
