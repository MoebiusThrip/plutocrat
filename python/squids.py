#!/usr/bin/env python3

# squids.py for the Squid class for ml studies on ozone profiles and uv spectra

# import local classes
from cores import Core
from hydras import Hydra
from features import Feature
from formulas import Formula

# import system
import sys

# import regex
import re

# import numpy functions
import numpy


# class Squid to do OMI data reduction
class Squid(Hydra):
    """Squid class making plot files.

    Inherits from:
        Hydra
    """

    def __init__(self, sink):
        """Initialize instance.

        Arguments:
            sink: str, sink diectory
            tag: str, tag for data
        """

        # initialize the base Core instance
        Hydra.__init__(self)

        # set sink directory
        self.sink = sink

        return

    def __repr__(self):
        """Create string for on screen representation.

        Arguments:
            None

        Returns:
            str
        """

        # create representation
        representation = ' < Squid instance at {} >'.format(self.sink)

        return representation

    def ink(self, dependent, ordinate, independent, abscissa, relative, title='', auxiliaries=None):
        """Create line plot file.

        Arguments:
            dependent: str, name of dependent variable
            ordinate: numpy array, ordinate
            independent: str, name of independent variable
            abscissa: numpy array, abscissa
            relative: str, file path relative to sink
            title: str, graph title
            auxiliaries: dict of str: array, auxiliary abscissas

        Returns:
            None
        """

        # default auxiliaries
        auxiliaries = auxiliaries or {}

        # begin features
        features = []

        # make ordinate
        attributes = {'title': title, 'units': dependent}
        feature = Feature(['Categories', dependent], numpy.array(ordinate).flatten(), attributes=attributes)
        features.append(feature)

        # make abscissa
        feature = Feature(['IndependentVariables', independent], numpy.array(abscissa).flatten())
        features.append(feature)

        # add auxiliaries
        for name, array in auxiliaries.items():

            # add number
            feature = Feature(['IndependentVariables', name], numpy.array(array).flatten())
            features.append(feature)

        # stash file
        destination = '{}/{}'.format(self.sink, relative)
        self.stash(features, destination)

        return None

    def ripple(self, name, array, relative, title='', bins=100):
        """Create histogram file.

        Arguments:
            dependent: str, name of dependent variable
            ordinate: numpy array, ordinate
            independent: str, name of independent variable
            abscissa: numpy array, abscissa
            relative: str, file path relative to sink
            title: str, graph title

        Returns:
            None
        """

        # begin features
        features = []

        # make ordinate
        attributes = {'title': title, 'units': 'normalized counts', 'bins': bins}
        address = ['Categories', 'histogram_{}'.format(name)]
        feature = Feature(address, numpy.array(array).flatten(), attributes=attributes)
        features.append(feature)

        # stash file
        destination = '{}/{}'.format(self.sink, relative)
        self.stash(features, destination)

        return None

    def pulsate(self, texts, arrays, polygons, folder, bounds=None, title='', units='', precision='f2'):
        """Create a polygon heatmap file.

        Arguments:
            texts: list of str, name of tracer, ardinate, and abscissa
            arrays: list of numpy.array, the tracer, ordinate, and abscissa
            polygons: numpy array
            folder: str, file path folder to sink
            bounds: list of floats, the bracket boundaries
            title: str, graph title
            units: str, graph units
            precision: str, precision string

        Returns:
            None
        """

        # unpack texts and tracers
        name, dependent, independent = texts
        tracer, ordinate, abscissa = arrays

        # begin features
        features = []

        # set default bounds
        if not bounds:

            # create default bounds
            chunks = 5
            chunk = (tracer.max() - tracer.min()) / chunks
            bounds = [index * chunk + tracer.min() for index in range(chunks + 1)]
            bounds = [round(entry, 2) for entry in bounds]

        # construct brackets and labels from bounds
        brackets = [(first, second) for first, second in zip(bounds[:-1], bounds[1:])]
        labels = ['{} to {}'.format(*bracket) for bracket in brackets]

        # make ordinate
        units = units or dependent
        attributes = {'title': title, 'units': units, 'bounds': bounds, 'brackets': brackets, 'labels': labels}

        # make abscissa
        arrays = [numpy.array(array).flatten() for array in (tracer, ordinate, abscissa)]
        stack = numpy.vstack(arrays)
        polygons = polygons.transpose(1, 0)
        stack = numpy.vstack([stack, polygons])

        # reduce precision
        stack = stack.astype(precision)

        # add feature
        address = ['Categories', 'heatmap_{}'.format(name)]
        feature = Feature(address, stack, attributes=attributes)
        features.append(feature)

        # stash file
        destination = '{}/{}'.format(self.sink, folder)
        self.stash(features, destination)

        return None

    def shimmer(self, name, tracer, dependent, ordinate, independent, abscissa, folder, bounds=None, title='', units=''):
        """Create heatmap file.

        Arguments:
            name: str, name of parameter
            tracer: numpy array, parameter values
            dependent: str, name of dependent variable
            ordinate: numpy array, ordinate
            independent: str, name of independent variable
            abscissa: numpy array, abscissa
            folder: str, file path folder to sink
            bounds: list of floats, the bracket boundaries
            title: str, graph title
            units: str, units for heatmap
            polygons: numpy.array of polygon bounds

        Returns:
            None
        """

        # begin features
        features = []

        # set default bounds
        if not bounds:

            # create default bounds
            chunks = 5
            chunk = (tracer.max() - tracer.min()) / chunks
            bounds = [index * chunk + tracer.min() for index in range(chunks + 1)]
            bounds = [round(entry, 2) for entry in bounds]

        # construct brackets and labels from bounds
        brackets = [(first, second) for first, second in zip(bounds[:-1], bounds[1:])]
        labels = ['{} to {}'.format(*bracket) for bracket in brackets]

        # make ordinate
        units = units or name.replace('_', ' ')
        attributes = {'title': title, 'units': units, 'bounds': bounds, 'brackets': brackets, 'labels': labels}

        # make abscissa
        arrays = [numpy.array(array).flatten() for array in (tracer, ordinate, abscissa)]
        stack = numpy.vstack(arrays)
        address = ['Categories', 'heatmap_{}'.format(name)]
        feature = Feature(address, stack, attributes=attributes)
        features.append(feature)

        # stash file
        destination = '{}/{}'.format(self.sink, folder)
        self.stash(features, destination)

        return None

    def splatter(self, dependent, ordinates, independent, abscissa, relative, title='', auxiliaries=None):
        """Create multi line plot file.

        Arguments:
            dependent: str, name of dependent variable
            ordinates: list of numpy array, ordinates
            independent: str, name of independent variable
            abscissa: numpy array, abscissa
            relative: str, file path relative to sink
            title: str, graph title
            auxiliaries: dict of name, array for auxiliary info

        Returns:
            None
        """

        # begin features
        features = []

        # make ordinate
        attributes = {'title': title, 'units': dependent}
        arrays = [numpy.array(ordinate).flatten() for ordinate in ordinates]
        stack = numpy.vstack(arrays)
        feature = Feature(['Categories', dependent], stack, attributes=attributes)
        features.append(feature)

        # make abscissa
        feature = Feature(['IndependentVariables', independent], numpy.array(abscissa).flatten())
        features.append(feature)

        # add auxiliaries
        auxiliaries = auxiliaries or {}
        for name, array in auxiliaries.items():

            # add abscissas
            feature = Feature(['IndependentVariables', 'z_{}'.format(name)], numpy.array(array).flatten())
            features.append(feature)

        # stash file
        destination = '{}/{}'.format(self.sink, relative)
        self.stash(features, destination, scan=True)

        return None

    def tangle(self, singlets, multiplet, title=None):
        """Combine singlet plots into a multiplet.

        Arguments:
            multiplet: str, filename for multiplet
            singlets: list of singlet files
            title: str, title

        Returns:
            None
        """

        # get both features from first path
        hydra = Hydra(singlets[0])
        hydra.ingest()

        # begin data and attributes
        data = {}
        attributes = {}

        # grab independent variables
        independent = hydra.dig('IndependentVariables')[0]
        data[independent.slash] = independent.distil()
        attributes[independent.slash] = independent.attributes

        # begin cateogory, putting array in new list
        category = hydra.dig('Categories')[0]
        data[category.slash] = [category.distil()]
        attributes[category.slash] = category.attributes

        # if title given
        if title:

            # add to attributes
            attributes['title'] = title

        # for all other paths
        for path in singlets[1:]:

            # grab data
            hydra = Hydra(path)
            hydra.ingest()
            array = hydra.grab('Categories')

            # stack into data
            data[category.slash].append(array)

        # make into array
        data[category.slash] = numpy.array(data[category.slash])

        # create multiplet
        hydra.spawn(multiplet, data, attributes)

        return None






