#!/usr/bin/env python3

"""
Plot difference of subpops. of ImageNet processed under a pretrained ResNet-18.

Copyright (c) Facebook, Inc. and its affiliates.

This script creates a directory, "unweighted," in the working directory if the
directory does not already exist, and then saves eight files in there for each
pair of classes for each of the types of scores (the types being probabilities
or negative log-likelihoods) for each of the fractions of the full range of
scores being considered:
1. metrics written by the function, "write_metrics," defined below
2. graph of cumulative differences between the pair of classes considered
3. reliability diagram with both classes with 10 bins (equispaced in scores)
4. reliability diagram with both classes with 30 bins (equispaced in scores)
5. reliability diagram with both classes with 50 bins (equispaced in scores)
6. reliability diagram with both classes with 10 bins
   (equal number of subpopulation scores per bin)
7. reliability diagram with both classes with 30 bins
   (equal number of subpopulation scores per bin)
8. reliability diagram with both classes with 50 bins
   (equal number of subpopulation scores per bin)

The naming convention for the files is
1. [prob or nll]-[fraction]-[number]-[name]_[number]-[name].txt
2. [prob or nll]-[fraction]-[number]-[name]_[number]-[name].pdf
3. [prob or nll]-[fraction]-[number]-[name]_[number]-[name]equiscore10.pdf
4. [prob or nll]-[fraction]-[number]-[name]_[number]-[name]equiscore30.pdf
5. [prob or nll]-[fraction]-[number]-[name]_[number]-[name]equiscore50.pdf
6. [prob or nll]-[fraction]-[number]-[name]_[number]-[name]equisamps10.pdf
7. [prob or nll]-[fraction]-[number]-[name]_[number]-[name]equisamps30.pdf
8. [prob or nll]-[fraction]-[number]-[name]_[number]-[name]equisamps50.pdf
where "prob" indicates that the scores are probabilities,
"nll" indicates that the scores are negative log-likelihoods,
"fraction" is the fraction of the full range of scores considered,
the first "number" is the number of the first class from among the 1,000,
the first "name" is a textual label for the first class in the pair,
the second "number" is the number of the second class from among the 1,000, and
the second "name" is a textual label for the second class in the pair.

This source code is licensed under the MIT license found in the LICENSE file in
the root directory of this source tree.
"""

import math
import numpy as np
import os
import string
import torch
import torchvision.models as models
from torchvision import datasets, transforms
from PIL import Image
from multiprocessing import Process

import disjoint


# Set the path to the directory containing the data set for running inference.
infdir = '/datasets01/imagenet_full_size/061417/train'


def scores_and_results(output, target, probs):
    """
    Computes the scores and corresponding correctness

    Given output from a classifier, computes the score from the cross-entropy
    and checks whether the most likely class matches target.

    Parameters
    ----------
    output : array_like
        confidences (often in the form of scores prior to a softmax that would
        output a probability distribution over the classes) in classification
        of each example into every class
    target : array_like
        index of the correct class for each example
    probs : bool
        set to True if the scores being returned should be probabilities;
        set to False if the scores should be negative log-likelihoods

    Returns
    -------
    array_like
        scores (probabilities if probs is True, negative log-likelihoods
        if probs is False)
    array_like
        Boolean indicators of correctness of the classifications
    """
    argmaxs = torch.argmax(output, 1)

    results = argmaxs.eq(target)
    results = results.cpu().detach().numpy()

    scores = torch.nn.CrossEntropyLoss(reduction='none')
    scores = scores(output, argmaxs)
    if probs:
        scores = torch.exp(-scores)
    scores = scores.cpu().detach().numpy()
    # Randomly perturb the scores to ensure their uniqueness.
    perturb = np.ones(scores.size) - np.random.rand(scores.size) * 1e-8
    scores = scores * perturb
    scores = scores + np.random.rand(scores.size) * 1e-12

    return scores, results


def infer(inf_loader, model, num_batches, probs=False):
    """
    Conducts inference given a model and data loader

    Runs model on data loaded from inf_loader.

    Parameters
    ----------
    inf_loader : class
        instance of torch.utils.data.DataLoader
    model : class
        torch model
    num_batches : int
        expected number of batches to process (used only for gauging progress
        by printing this number)
    probs : bool, optional
        set to True if the scores being returned should be probabilities;
        set to False if the scores should be negative log-likelihoods

    Returns
    -------
    array_like
        scores (probabilities if probs is True, negative log-likelihoods
        if probs is False)
    array_like
        Boolean indicators of correctness of the classifications
    list
        ndarrays of indices of the examples classified into each class
        (the i'th entry of the list is an array of the indices of the examples
        from the data set that got classified into the i'th class)
    """
    model.eval()
    # Track the offset for appending indices to indicators (by default,
    # each minibatch gets indexed starting from 0, rather than offset).
    offset = 0
    indicators = [None] * 1000
    for k, (input, target) in enumerate(inf_loader):
        print(f'{k} of {num_batches} batches processed.')
        target = target.cuda(non_blocking=True)
        with torch.no_grad():
            input_var = torch.autograd.Variable(input)
            target_var = torch.autograd.Variable(target)
        # Run inference.
        output = model(input_var)
        # Store the scores and results from the current minibatch,
        # and record which entries have the desired target indices.
        s, r = scores_and_results(output, target, probs)
        # Record the scores and results.
        if k == 0:
            scores = s.copy()
            results = r.copy()
        else:
            scores = np.concatenate((scores, s))
            results = np.concatenate((results, r))
        # Partition the results into the 1000 classes.
        for i in range(1000):
            inds = torch.nonzero(
                target == i, as_tuple=False).cpu().detach().numpy()
            if k == 0:
                indicators[i] = inds
            else:
                indicators[i] = np.concatenate((indicators[i], inds + offset))
        # Increment offset.
        offset += target.numel()
    print(f'{k + 1} of {num_batches} batches processed.')
    for i in range(1000):
        indicators[i] = np.squeeze(indicators[i])
    print('m = *scores.shape = {}'.format(*scores.shape))
    return scores, results, indicators


def write_metrics(
        filename, lencums, n0, n1, fraction, lenscale, kuiper,
        kolmogorov_smirnov):
    """
    Saves the provided metrics to a text file

    Writes to the text file named filename the parameters lencums, n0, n1,
    fraction, lenscale, kuiper, kolmogorov_smirnov, kuiper/lenscale, and
    kolmogorov_smirnov/lenscale.

    Parameters
    ----------
    filename : string
        name of the file in which to save the metrics
    lencums : int
        integer (for example, the length of the cumulative sequence)
    n0 : int
        integer (for example, the size of one subpopulation)
    n1 : int
        integer (for example, the size of the other subpopulation)
    fraction : float
        real number (for example, the fraction of the observations considered)
    lenscale : float
        standard deviation for normalizing kuiper and kolmogorov_smirnov
        in order to gauge statistical significance
    kuiper : float
        value of the Kuiper statistic
    kolmogorov_smirnov : float
        value of the Kolmogorov-Smirnov statistic

    Returns
    -------
    None
    """
    with open(filename, 'w') as f:
        f.write('n:\n')
        f.write(f'{lencums}\n')
        f.write('n[0]:\n')
        f.write(f'{n0}\n')
        f.write('n[1]:\n')
        f.write(f'{n1}\n')
        f.write('fraction:\n')
        f.write(f'{fraction}\n')
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


def disjoint_cumulative(
        r, s, majorticks, minorticks, fraction, probs=False,
        filename='cumulative',
        title='subpop. deviation is the slope as a function of $k/n$'):
    """Thin wrapper around disjoint.cumulative for multiprocessing"""
    kuiper, kolmogorov_smirnov, lenscale, lencums = disjoint.cumulative(
        r, s, majorticks, minorticks, probs, filename + '.pdf', title,
        fraction)
    write_metrics(filename + '.txt', lencums, int(len(r[0])), int(len(r[1])),
                  fraction, lenscale, kuiper, kolmogorov_smirnov)


# Run on the second GPU in case someone else wants the first.
torch.cuda.device(1)

# Read the textual descriptions of the classes.
with open('imagenet_classes.txt') as f:
    classes = [line.strip() for line in f.readlines()]

# Conduct inference.
batch_size = 512
for probs in [True, False]:
    # Set the seeds for the random number generators.
    torch.manual_seed(89259348)
    np.random.seed(seed=3820497)
    # Load the pretrained model.
    resnet18 = models.resnet18(pretrained=True)
    resnet18 = torch.nn.DataParallel(resnet18).cuda()
    # Construct the data loader.
    normalize = transforms.Normalize(
        mean=[.485, .456, .406], std=[.229, .224, .225])
    inf_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(infdir, transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    num_batches = math.ceil(1281167 / batch_size)
    # Generate the scores, results, and subset indicators for ImageNet.
    print()
    print(f'probs = {probs}')
    if probs:
        s_prob, r_prob, inds_prob = infer(inf_loader, resnet18, num_batches,
                                          probs=True)
    else:
        s_nll, r_nll, inds_nll = infer(inf_loader, resnet18, num_batches,
                                       probs=False)

# Create the directory "unweighted" for output, if necessary.
dir = 'unweighted'
try:
    os.mkdir(dir)
except FileExistsError:
    pass
dir = dir + '/'

# Set parameters.
for (probs, fraction) in [(True, 1), (False, 1), (False, .1)]:

    # Set up for the relevant scores.
    print()
    print(f'probs = {probs}')
    print(f'fraction = {fraction}')
    if probs:
        s = s_prob
        r = r_prob
        inds = inds_prob.copy()
    else:
        s = s_nll
        r = r_nll
        inds = inds_nll.copy()

    # Sort the scores and rearrange everything accordingly.
    perm = np.argsort(s)
    s = s[perm]
    r = r[perm]
    # Construct the inverse of the permutation perm.
    iperm = np.zeros((len(perm)), dtype=np.int32)
    for k in range(len(perm)):
        iperm[perm[k]] = k
    for i in range(1000):
        inds[i] = np.sort(
            np.array([iperm[inds[i][k]] for k in range(len(inds[i]))]))

    # Generate the plots for disjoint subpopulations.
    majorticks = 10
    minorticks = 200
    procs = []
    # List the indices of which classes will get plotted.
    indices = [60, 68, 248, 293, 323, 342, 837]
    for i in indices:
        for j in [k for k in indices if k != i]:
            if probs:
                prefix = 'prob'
            else:
                prefix = 'nll'
            prefix += '-' + str(fraction) + '-'
            prefix = dir + prefix
            filename0 = classes[i]
            filename0 = filename0.translate(
                str.maketrans('', '', string.punctuation))
            filename0 = filename0.replace(' ', '-')
            filename1 = classes[j]
            filename1 = filename1.translate(
                str.maketrans('', '', string.punctuation))
            filename1 = filename1.replace(' ', '-')
            filename = prefix + filename0 + '_' + filename1
            r01 = [r[inds[i]], r[inds[j]]]
            s01 = [s[inds[i]], s[inds[j]]]
            procs.append(Process(
                target=disjoint_cumulative,
                args=(r01, s01, majorticks, minorticks, fraction),
                kwargs={'probs': probs, 'filename': filename}))
            for nbins in [10, 30, 50]:
                procs.append(Process(
                    target=disjoint.equiscore, args=(r01, s01, nbins),
                    kwargs={'filename':
                            filename + 'equiscore' + str(nbins) + '.pdf'}))
                procs.append(Process(
                    target=disjoint.equisamps, args=(r01, s01, nbins),
                    kwargs={'filename':
                            filename + 'equisamps' + str(nbins) + '.pdf'}))
    for iproc, proc in enumerate(procs):
        print(f'{iproc + 1} of {len(procs)} plots have started....')
        proc.start()
