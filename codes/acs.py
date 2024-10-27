#!/usr/bin/env python3

"""
Plot the subpopulation deviations for the American Community Survey of USCB.

Copyright (c) Facebook, Inc. and its affiliates.

This script creates a directory, "weighted," in the working directory if the
directory does not already exist, then creates subdirectories there for each
pair of the counties in California specified by the list "counties" defined
below, and fills each subdirectory with eight files:
1. metrics.txt -- metrics about the plots
2. cumulative.pdf -- plot of cumulative differences between the counties
3. equiscores10.pdf -- reliability diagram of the counties with 10 bins
                       (equispaced in scores)
4. equiscores20.pdf -- reliability diagram of the counties with 20 bins
                       (equispaced in scores)
5. equiscores100.pdf -- reliability diagram of the counties with 100 bins
                        (equispaced in scores)
6. equierrs10.pdf -- reliability diagram of the counties with about 10 bins
                     (the error bar is about the same for every bin)
7. equierrs20.pdf -- reliability diagram of the counties with about 20 bins
                     (the error bar is about the same for every bin)
8. equierrs100.pdf -- reliability diagram of the counties with about 100 bins
                      (the error bar is about the same for every bin)
The data comes from the American Community Survey of the U.S. Census Bureau,
specifically the household data from the counties in the state of California.
The scores are log_10 of the adjusted household personal incomes.
The results/responses are given by the variates specified in the list "varval"
defined below (together with the value of the variate to be considered
"success" in the sense of Bernoulli trials, or else the nonnegative integer
count for the variate, counting people, for instance).

This source code is licensed under the MIT license found in the LICENSE file in
the root directory of this source tree.
"""

import math
import numpy as np
import os
import shutil
from numpy.random import default_rng

from disjoint_weighted import equiscores, equierrs, cumulative


# Specify the name of the file of comma-separated values
# for the household data in the American Community Survey.
filename = 'psam_h06.csv'

# Count the number of lines in the file for filename.
lines = 0
with open(filename, 'r') as f:
    for line in f:
        lines += 1
print(f'reading and filtering all {lines} lines from {filename}....')

# Determine the number of columns in the file for filename.
with open(filename, 'r') as f:
    line = f.readline()
    num_cols = line.count(',') + 1

# Read and store all but the first two columns in the file for filename.
raw = np.zeros((lines, num_cols - 2))
with open(filename, 'r') as f:
    for line_num, line in enumerate(f):
        parsed = line.split(',')[2:]
        if line_num == 0:
            # The initial line is a header ... save its column labels.
            header = parsed.copy()
            # Eliminate the newline character at the end of the line.
            header[-1] = header[-1][:-1]
        else:
            # All but the initial line consist of data ... extract the ints.
            raw[line_num - 1, :] = np.array(
                [int(s if s != '' else -1) for s in parsed])

# Filter out undesirable observations -- keep only strictly positive weights,
# strictly positive household personal incomes, and strictly positive factors
# for adjusting the income.
keep = np.logical_and.reduce([
    raw[:, header.index('WGTP')] > 0,
    raw[:, header.index('HINCP')] > 0,
    raw[:, header.index('ADJINC')] > 0])
raw = raw[keep, :]
print(f'm = raw.shape[0] = {raw.shape[0]}')

# Form a dictionary of the lower- and upper-bounds on the ranges of numbers
# of the public-use microdata areas (PUMAs) for the counties in California.
puma = {
    'Alameda': (101, 110),
    'Alpine, Amador, Calaveras, Inyo, Mariposa, Mono and Tuolumne': (300, 300),
    'Butte': (701, 702),
    'Colusa, Glenn, Tehama and Trinity': (1100, 1100),
    'Contra Costa': (1301, 1309),
    'Del Norte, Lassen, Modoc, Plumas and Siskiyou': (1500, 1500),
    'El Dorado': (1700, 1700),
    'Fresno': (1901, 1907),
    'Humboldt': (2300, 2300),
    'Imperial': (2500, 2500),
    'Kern': (2901, 2905),
    'Kings': (3100, 3100),
    'Lake and Mendocino': (3300, 3300),
    'Los Angeles': (3701, 3769),
    'Madera': (3900, 3900),
    'Marin': (4101, 4102),
    'Merced': (4701, 4702),
    'Monterey': (5301, 5303),
    'Napa': (5500, 5500),
    'Nevada and Sierra': (5700, 5700),
    'Orange': (5901, 5918),
    'Placer': (6101, 6103),
    'Riverside': (6501, 6515),
    'Sacramento': (6701, 6712),
    'San Bernardino': (7101, 7115),
    'San Diego': (7301, 7322),
    'San Francisco': (7501, 7507),
    'San Joaquin': (7701, 7704),
    'San Luis Obispo': (7901, 7902),
    'San Mateo': (8101, 8106),
    'Santa Barbara': (8301, 8303),
    'Santa Clara': (8501, 8514),
    'Santa Cruz': (8701, 8702),
    'Shasta': (8900, 8900),
    'Solano': (9501, 9503),
    'Sonoma': (9701, 9703),
    'Stanislaus': (9901, 9904),
    'Sutter and Yuba': (10100, 10100),
    'Tulare': (10701, 10703),
    'Ventura': (11101, 11106),
    'Yolo': (11300, 11300),
}

# Specify which counties and variates to process, as well as the coded value
# of interest for each variate (or None if the values of interest are
# nonnegative integer counts).
varval = [('LNGI', 2)]
counties = [
    'Alameda', 'Butte', 'Contra Costa', 'Kern', 'Placer', 'Riverside',
    'San Francisco', 'San Joaquin', 'San Mateo']
exs = [{'county0': county0, 'county1': county1, 'var': var, 'val': val}
       for county0 in counties for county1 in counties
       if county0 != county1 for var, val in varval]

# Process the examples.
for ex in exs:

    # Form the scores, results, and weights.
    np.random.seed(seed=3820497)
    # Adjust the household personal income by the relevant factor.
    s = raw[:, header.index('HINCP')] * raw[:, header.index('ADJINC')] / 1e6
    # Convert the adjusted incomes to a log (base-10) scale.
    s = np.log(s) / math.log(10)
    # Dither in order to ensure the uniqueness of the scores.
    s = s * (np.ones(s.shape) + np.random.normal(size=s.shape) * 1e-8)
    # Read the result (raw integer count if the specified value is None,
    # Bernoulli indicator of success otherwise).
    if ex['val'] is None:
        r = raw[:, header.index(ex['var'])]
    else:
        r = raw[:, header.index(ex['var'])] == ex['val']
    # Read the weight.
    w = raw[:, header.index('WGTP')]
    # Sort the scores.
    perm = np.argsort(s)
    s = s[perm]
    r = r[perm]
    w = w[perm]

    # Set a directory for the counties (creating the directory if necessary).
    dir = 'weighted'
    try:
        os.mkdir(dir)
    except FileExistsError:
        pass
    dir = 'weighted/County_of_'
    dir += ex['county0'].replace(' ', '_').replace(',', '')
    dir += '_vs_'
    dir += ex['county1'].replace(' ', '_').replace(',', '')
    dir += '-'
    dir += ex['var']
    try:
        os.mkdir(dir)
    except FileExistsError:
        pass
    dir += '/'
    print(f'./{dir} is under construction....')

    # Identify the indices of the subsets corresponding to the counties.
    inds = []
    for j in range(2):
        slice = raw[perm, header.index('PUMA')]
        inds.append(
            slice >= (puma[ex['county' + str(j)]][0] * np.ones(raw.shape[0])))
        inds[j] = inds[j] & (
            slice <= (puma[ex['county' + str(j)]][1] * np.ones(raw.shape[0])))
        inds[j] = np.nonzero(inds[j])[0]
        inds[j] = np.unique(inds[j])

    # List the results, scores, and weights for the counties.
    r01 = [r[inds[j]] for j in range(2)]
    s01 = [s[inds[j]] for j in range(2)]
    w01 = [w[inds[j]] for j in range(2)]

    # Plot reliability diagrams and the cumulative graph.
    nin = [10, 20, 100]
    nout = {}
    for nbins in nin:
        filename = dir + 'equiscores' + str(nbins) + '.pdf'
        equiscores(r01, s01, nbins, filename, weights=w01, left=0)
        filename = dir + 'equierrs' + str(nbins) + '.pdf'
        rng = default_rng(seed=987654321)
        nout[str(nbins)] = equierrs(
            r01, s01, nbins, rng, filename, weights=w01)
    majorticks = 10
    minorticks = 300
    filename = dir + 'cumulative.pdf'
    kuiper, kolmogorov_smirnov, lenscale, lencums = cumulative(
        r01, s01, majorticks, minorticks, False,
        filename=filename, weights=w01)
    # Save metrics in a text file.
    filename = dir + 'metrics.txt'
    with open(filename, 'w') as f:
        f.write('m:\n')
        f.write(f'{len(s)}\n')
        f.write('n:\n')
        f.write(f'{lencums}\n')
        f.write('n[0]:\n')
        f.write(f'{len(inds[0])}\n')
        f.write('n[1]:\n')
        f.write(f'{len(inds[1])}\n')
        f.write('lenscale:\n')
        f.write(f'{lenscale}\n')
        for nbins in nin:
            f.write("nout['" + str(nbins) + "']:\n")
            f.write(f'{nout[str(nbins)][0]}\n')
            f.write(f'{nout[str(nbins)][1]}\n')
        f.write('Kuiper:\n')
        f.write(f'{kuiper:.4}\n')
        f.write('Kolmogorov-Smirnov:\n')
        f.write(f'{kolmogorov_smirnov:.4}\n')
        f.write('Kuiper / lenscale:\n')
        f.write(f'{(kuiper / lenscale):.4}\n')
        f.write('Kolmogorov-Smirnov / lenscale:\n')
        f.write(f'{(kolmogorov_smirnov / lenscale):.4}\n')
    if kolmogorov_smirnov / lenscale < 1:
        shutil.rmtree(dir[:-1])
