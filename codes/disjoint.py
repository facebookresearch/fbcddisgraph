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
equiscore
    Reliability diagram with roughly equispaced average scores over bins
equisamps
    Reliability diagram with an equal number of observations per bin
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


def cumulative(r, s, majorticks, minorticks, probs=False,
               filename='cumulative.pdf',
               title='subpop. deviation is the slope as a function of $k/n$',
               fraction=1):
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
        list of array_like values of class labels (0 for incorrect and
        1 for correct classification)
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

    def binvalues(a, d, e, key):
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
    # Create the figure.
    plt.figure()
    ax = plt.axes()
    # Merge the scores from the subpopulations.
    _, d, e = mergesorted(s[0], s[1])
    # Bin the responses according to the merged scores.
    b = []
    for j in range(2):
        b.append(binvalues(r[j], d, e, j))
    # Bin the scores according to the merged scores.
    t = []
    for j in range(2):
        t.append(binvalues(s[j], d, e, j))
    # Stagger the scores from the two subpopulations into t01.
    t01 = np.zeros((t[0].size + t[1].size))
    for k in range(t01.size):
        if k % 2 == 0:
            t01[k] = t[d[0]][k // 2]
        else:
            t01[k] = t[1 - d[0]][k // 2]
    assert all(t01[k] < t01[k + 1] for k in range(len(t01) - 1))
    # Accumulate and average the associated aggregations of differences.
    la0 = min(len(b[d[0]]) - 1, len(b[1 - d[0]]))
    la1 = min(len(b[d[0]]) - 1, len(b[1 - d[0]]) - 1)
    a0 = np.zeros((la0))
    for k in range(len(a0)):
        a0[k] = b[d[0]][k] - b[1 - d[0]][k]
        a0[k] += b[d[0]][k + 1] - b[1 - d[0]][k]
        a0[k] /= 2
    a1 = np.zeros((la1))
    for k in range(len(a1)):
        a1[k] = b[d[0]][k + 1] - b[1 - d[0]][k]
        a1[k] += b[d[0]][k + 1] - b[1 - d[0]][k + 1]
        a1[k] /= 2
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
    aa = np.cumsum(a[:int(len(a) * fraction)]) / int(len(a) * fraction)
    # Plot the cumulative differences.
    plt.plot(aa, 'k')
    # Make sure the plot includes the origin.
    plt.plot(0, 'k')
    # Add an indicator of the scale of 1/sqrt(n) to the vertical axis.
    t01sub = t01[:int(len(a) * fraction)]
    if probs:
        lenscale = np.sqrt(
            np.sum(t01sub * (1 - t01sub))) / int(len(a) * fraction)
    else:
        lenscale = np.sqrt(len(t01sub) / 4) / int(len(a) * fraction)
    # Adjust lenscale for the dependence between even and odd entries of t01.
    lenscale *= math.sqrt(2)
    # Adjust lenscale for taking the difference of 2 independent distributions
    # (one for each subpopulation).
    lenscale *= math.sqrt(2)
    plt.plot(2 * lenscale, 'k')
    plt.plot(-2 * lenscale, 'k')
    kwargs = {
        'head_length': 2 * lenscale, 'head_width': len(t01sub) / 20,
        'width': 0, 'linewidth': 0, 'length_includes_head': True, 'color': 'k'}
    plt.arrow(.1e-100, -2 * lenscale, 0, 4 * lenscale, shape='left', **kwargs)
    plt.arrow(.1e-100, 2 * lenscale, 0, -4 * lenscale, shape='right', **kwargs)
    plt.margins(x=0, y=0)
    # Label the major ticks of the lower axis with the values of t01.
    sl = [
        '{:.2f}'.format(x)
        for x in t01sub[::(len(t01sub) // majorticks)].tolist()]
    plt.xticks(
        np.arange(majorticks) * len(t01sub) // majorticks, sl[:majorticks])
    if len(t01sub) >= 300 and minorticks >= 50:
        # Indicate the distribution of t01 via unlabeled minor ticks.
        plt.minorticks_on()
        ax.tick_params(which='minor', axis='x')
        ax.tick_params(which='minor', axis='y', left=False)
        ax.set_xticks(np.cumsum(histcounts(minorticks, t01sub)), minor=True)
    # Label the axes.
    plt.xlabel('score ($S^0_{(k-1)/2}$ or $S^1_{(k-2)/2}$)')
    plt.ylabel('$C_k$')
    plt.twiny()
    plt.xlabel('$k/n$')
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


def equiscore(r, s, nbins, filename='equiscore.pdf'):
    """
    Reliability diagram with roughly equispaced average scores over bins

    Plots a reliability diagram with roughly equispaced average scores
    for the bins, for the two subpopulations whose responses are in r and
    whose scores are in s.

    Parameters
    ----------
    r : list
        list of array_like values of class labels (0 for incorrect and
        1 for correct classification)
    s : list
        list of array_like scores (each array must be in non-decreasing order)
    nbins : int
        number of bins
    filename : string, optional
        name of the file in which to save the plot

    Returns
    -------
    None
    """

    def bintwo(nbins, a, b, q, qmax):
        # Counts the number of entries of q falling into each of nbins bins,
        # and calculates the averages per bin of the arrays a and b,
        # returning np.nan as the "average" for any bin that is empty.
        j = 0
        bina = np.zeros(nbins)
        binb = np.zeros(nbins)
        nbin = np.zeros(nbins)
        for k in range(len(q)):
            if q[k] > qmax * (j + 1) / nbins:
                j += 1
            if j == nbins:
                break
            bina[j] += a[k]
            binb[j] += b[k]
            nbin[j] += 1
        # Normalize the sum for each bin to compute the arithmetic average.
        bina = np.divide(bina, nbin, where=nbin != 0)
        bina[np.where(nbin == 0)] = np.nan
        binb = np.divide(binb, nbin, where=nbin != 0)
        binb[np.where(nbin == 0)] = np.nan
        return nbin, bina, binb

    # Check that the arrays in s are both sorted.
    for j in range(2):
        assert all(s[j][k] <= s[j][k + 1] for k in range(len(s[j]) - 1))
    # Create the figure.
    plt.figure()
    colors = ['black', 'gray']
    smax = -1e20
    for j in range(1, -1, -1):
        _, binr, bins = bintwo(nbins, r[j], s[j], s[j], s[j][-1])
        smax = max(smax, s[j][-1])
        plt.plot(bins, binr, '*:', color=colors[j])
    plt.xlim((0, smax))
    plt.ylim((0, 1))
    plt.xlabel('average of scores in the bin')
    plt.ylabel('average of responses in the bin')
    plt.title('reliability diagram')
    plt.tight_layout()
    plt.savefig(filename, bbox_inches='tight')
    plt.close()


def equisamps(
        r, s, nbins, filename='equisamps.pdf',
        title='reliability diagram (equal number of subpop. scores per bin)'):
    """
    Reliability diagram with an equal number of observations per bin

    Plots a reliability diagram with an equal number of observations per bin,
    for the two subpopulations whose responses are in r and whose scores are
    in s.

    Parameters
    ----------
    r : list
        list of array_like values of class labels (0 for incorrect and
        1 for correct classification)
    s : list
        list of array_like scores (each array must be in non-decreasing order)
    nbins : int
        number of bins
    filename : string, optional
        name of the file in which to save the plot
    title : string, optional
        title of the plot

    Returns
    -------
    None
    """

    def hist(a, nbins):
        # Calculates the average of a in nbins bins,
        # each containing len(a) // nbins entries of a
        ns = len(a) // nbins
        return np.sum(np.reshape(a[:nbins * ns], (nbins, ns)), axis=1) / ns

    # Check that the arrays in s are both sorted.
    for j in range(2):
        assert all(s[j][k] <= s[j][k + 1] for k in range(len(s[j]) - 1))
    # Create the figure.
    plt.figure()
    colors = ['black', 'gray']
    binsmax = -1e20
    for j in range(1, -1, -1):
        binr = hist(r[j], nbins)
        bins = hist(s[j], nbins)
        binsmax = max(binsmax, np.max(bins))
        plt.plot(bins, binr, '*:', color=colors[j])
    plt.xlim((0, binsmax))
    plt.ylim((0, 1))
    plt.xlabel('average of scores in the bin')
    plt.ylabel('average of responses in the bin')
    plt.title(title)
    plt.tight_layout()
    plt.savefig(filename, bbox_inches='tight')
    plt.close()


def exactplot(r, s, filename='exact.pdf', title='exact expectations'):
    """
    Reliability diagram with exact values plotted

    Plots a reliability diagram at full resolution with fractional numbers,
    for the two subpopulations whose responses are in r and whose scores
    are in s. The entries of the members of r should be the expected values
    of class labels, not necessarily just 0s and 1s.

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
    plt.xlim((0, 1))
    plt.ylim((0, 1))
    plt.xlabel('score')
    plt.ylabel('expected value of the response')
    plt.title(title)
    plt.tight_layout()
    plt.savefig(filename, bbox_inches='tight')
    plt.close()


if __name__ == '__main__':
    #
    # Generate directories with plots as specified via the code below,
    # with each directory named unweighted/n[0]_n[1]_nbins_iex
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

            # Set a unique directory for each collection of experiments
            # (creating the directory if necessary).
            dir = 'unweighted'
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
                r, s, majorticks, minorticks, False, filename)
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
                title='exact expectations')
            filename = dir + 'equiscore.pdf'
            equiscore(r, s, nbins, filename)
            filename = dir + 'equisamps.pdf'
            equisamps(r, s, nbins, filename)
            filename = dir + 'exact.pdf'
            exactplot(exact, s, filename)
