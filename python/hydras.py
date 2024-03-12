#!/usr/bin/env python3

# hydras.py for the Hydra class to parse hdf files

# import local classes
from cores import Core
from features import Feature

# try to
try:

    # import formulas
    from formulas import Formula

# unless not avaialbe
except ImportError:

    # nevermind
    pass

# import general tools
import os
import re
import subprocess
import yaml

# import time and datetime
import time
import datetime
import calendar

# import collection
from collections import Counter

# import numpy nand math
import numpy
import math

# add warnings module
import warnings

# import netcdf4
import netCDF4

# import h5py to read h5 files
import h5py

# try to
try:

    # import pyhdf to handle Collection 3 hdf4 data
    from pyhdf.HDF import HDF, HDF4Error, HC
    from pyhdf.SD import SD, SDC
    from pyhdf.V import V
    from pyhdf.error import HDF4Error

# unless it is not installed
except (ImportError, SystemError):

    # in which case, nevermind
    pass

# import random forest
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier


# class Hydra to parse hdf files
class Hydra(Core):
    """Hydra class to parse hdf files.

    Inherits from:
        cores.Core
    """

    def __init__(self, source='', start='', finish=''):
        """Initialize a Hydra instance.

        Arguments:
            sink: str, filepath for data dump
            source: str, filepath of source files
            start: str, date-based subdirectory
            finish: str, date-based subdirectory

        Returns:
            None
        """

        # initialize the base Core instance
        Core.__init__(self)

        # set directory information
        self.source = source
        self.start = start
        self.finish = finish

        # set accepted file extensions
        self.extensions = ('.nc', '.he4', '.h5', '.nc4', '.he5', '.met')

        # gather all relevant paths
        self.paths = []
        self.destinations = []
        self._register()

        # reference for current features
        self.reference = {}

        # reference for current file
        self.current = ''

        # default tree to None
        self.tree = None

        # default net to fale for Netcdf4
        self.net = False

        return

    def __repr__(self):
        """Create string for on screen representation.

        Arguments:
            None

        Returns:
            str
        """

        # display contents
        self._tell(self.paths)

        # create representation
        representation = ' < Hydra instance at: {}, current: ( {} ) >'.format(self.source, self._file(self.current))

        return representation

    def _bite(self, number):
        """Determine bit flags in an integer.

        Arguments:
            number: int

        Returns:
            list of ints, the bit positions
        """

        # get binary string, removing leading 0b
        binary = bin(int(number))[2:]

        # determine length
        length = len(binary)

        # get bit positions
        positions = [length - index - 1 for index, bit in enumerate(binary) if bit == '1']
        positions.sort()

        return positions

    def _cache(self, features, destination, link=None, mode='w', compression=None, scan=False):
        """Stash a group of features in an hdf4 file.

        Arguments:
            features: list of dicts
            destination: str, destination filepath
            link: str, name of link folder
            mode: str, writemode
            compression: compression option
            scan: scan feature names?

        Returns:
            None
        """

        # fill all features if not yet filled
        [feature.fill() for feature in features]

        # link all Categories field, but not IndependentVariables
        [feature.update({'link': False}) for feature in features if 'IndependentVariables' in feature.slash]
        [feature.update({'link': True}) for feature in features if 'Categories' in feature.slash]

        # begin hdf4 file
        four = HDF(destination, HC.WRITE | HC.CREATE)
        science = SD(destination, SDC.WRITE | SDC.CREATE)

        # initialize group and table interfaces
        groups = four.vgstart()
        tables = four.vstart()

        # collect all route members
        members = [step for feature in features for step in feature.route[:-1]]
        members = self._skim(members)

        # create all groups
        registry = {member: groups.create(member) for member in members}

        # for each feature
        for feature in features:

            # add data as SD inside group
            array = science.create(feature.name, SDC.FLOAT64, feature.data.shape)
            array[:] = feature.data[:]
            tag = array.ref()
            print(tag)

            # get the route
            route = feature.route

            print(route)

            # create group for first step
            group = registry[route[0]]
            for step in route[1:-1]:

                print(dir(group))
                print(step)
                print(group.tagrefs())

                # add group
                groupii = registry[step]
                try:

                    # to add group
                    group.insert(groupii)

                # unless already
                except HDF4Error:

                    # skip
                    pass

                # walk down to next member
                group = groupii

            #  insert the science
            group.add(HC.DFTAG_NDG, tag)
            array.endaccess()

        # close groups, tables, and files
        groups.end()
        tables.end()
        science.end()
        four.close()

        return None

    def _cameo(self, name):
        """Make a name into camel case.

        Arguments:
            name: str

        Returns:
            str
        """

        # split at underscore
        sections = name.split('_')

        # for each section
        camels = []
        for section in sections:

            # capitalize first letter
            section = section[0].upper() + section[1:]

            # findall capitalized sections
            camels += re.findall('[A-Z][a-z,0-9]*', section)

        # make lower case
        camels = [camel.capitalize() for camel in camels]

        # join at underscore
        cameo = ''.join(camels)

        # # split on colons
        # camels = []
        # sections = name.split('/')
        # for section in sections:
        #
        #     # split by underscores
        #     words = section.split('_')
        #
        #     # capitalize each part
        #     words = [word.capitalize() for word in words]
        #
        #     # rejoin
        #     camel = ''.join(words)
        #     camels.append(camel)
        #
        # # connect main name with underscore
        # cameo = '_'.join(camels)

        return cameo

    def _collect(self, four, science, names=None):
        """Collect all routes and shapes from a datafile.

        Arguments:
            four: hdf4 file pointer
            science: hdf4 sd pointer
            names: list of str, select feature names

        Returns:
            list of dicts
        """

        # initialize collection and route
        collection = []
        route = []

        # unpack interaces tuple for hdf4 object and dataset object
        registry = self._scan(four)

        # initialize group and table interfaces
        groups = four.vgstart()
        tables = four.vstart()

        # go through registry
        for reference, steps in registry.items():

            # get tags
            group = groups.attach(reference)
            tags = group.tagrefs()
            for tag, number in tags:

                # check for scientific dataset
                if tag == HC.DFTAG_NDG:

                    # get attributes
                    dataset = science.select(science.reftoindex(number))
                    name, rank, dims, type, _ = dataset.info()

                    # convert dims to list
                    try:

                        # get the length
                        length = len(dims)

                    # unless no a list
                    except TypeError:

                        # create a list from dims
                        dims = [dims]

                    # check for specific name
                    if not names or name in names:

                        # get data and add to feature
                        parcel = dataset.get()

                        # create feature
                        feature = Feature(route + steps + [name], numpy.array([parcel]), tuple(dims), '', 'sd')

                        # add to collection
                        collection.append(feature)

                # check for vdata
                elif tag == HC.DFTAG_VH:

                    # get attributes
                    table = tables.attach(number)
                    records, _, _, size, name = table.inquire()

                    # check for specific name
                    if not names or name in names:

                        # read all lines to get data
                        descending = lambda line: line[0] if len(line) < 2 else line
                        parcel = [descending(table.read()[0]) for _ in range(records)]

                        # create feature
                        feature = Feature(route + steps + [name], numpy.array([parcel]), (records, size), '', 'vdata')

                        # add to collection
                        collection.append(feature)

                    # close table
                    table.detach()

        # close groups, tables, and files
        groups.end()
        tables.end()
        science.end()
        four.close()

        return collection

    def _compress(self, corners, trackwise, rowwise):
        """Compress the latitude or longitude bounds based on compression factors.

        Arguments:
            corners: dict of corner arrays
            trackwise: north-south compression factor
            rowwise: east-west compression factor

        Returns:
            numpy array
        """

        # set directions
        compasses = ['southwest', 'southeast', 'northeast', 'northwest']

        # # for each corner
        # for compass in compasses:
        #
        #     # shift negative longitudes to positive
        #     corner = corners[compass][:, :, 1]
        #     corners[compass][:, :, 1] = numpy.where(corner < 0, corner + 360, corner)

        # get shape
        shape = corners['southwest'].shape

        # get selected trackwise indices and rowwise indices
        selection = numpy.array([index for index in range(shape[0]) if index % trackwise == 0])
        selectionii = numpy.array([index for index in range(shape[1]) if index % rowwise == 0])

        # adjust based on factors
        adjustment = numpy.array([min([index + trackwise - 1, shape[0] - 1]) for index in selection])
        adjustmentii = numpy.array([min([index + rowwise - 1, shape[1] - 1]) for index in selectionii])

        # begin compression
        compression = {}

        # adjust southwest
        compression['southwest'] = corners['southwest'][selection]
        compression['southwest'] = compression['southwest'][:, selectionii]
        compression['southeast'] = corners['southeast'][selection]
        compression['southeast'] = compression['southeast'][:, adjustmentii]
        compression['northeast'] = corners['northeast'][adjustment]
        compression['northeast'] = compression['northeast'][:, adjustmentii]
        compression['northwest'] = corners['northwest'][adjustment]
        compression['northwest'] = compression['northwest'][:, selectionii]

        # # for each corner
        # for compass in compasses:
        #
        #     # shift negative longitudes to positive
        #     corner = compression[compass][:, :, 1]
        #     compression[compass][:, :, 1] = numpy.where(corner >= 180, corner - 360, corner)

        return compression

    def _convert(self, path, destination, names=None):
        """Convert an hdf4 file path to an hdf5 file.

        Arguments:
            path: str, pathname
            destination: str, dump pathname

        Returns:
            None
        """

        # begin conversion
        self._print('converting {} to h5...'.format(path))

        # open up hdf4 interfaces
        four = HDF(path)
        science = SD(path)

        # convert into features
        features = self._collect(four, science, names)

        # stash as h5 file
        self._print('stashing...')
        self.stash(features, destination)

        # reopen as contents
        contents = h5py.File(destination, 'r')

        # timestamp
        self._print('{} converted.'.format(path))

        return contents

    def _cross(self, bounds):
        """Adjust for longitude bounds that cross the dateline.

        Arguments:
            bounds: numpy array of bounds

        Returns:
            numpy array
        """

        # for each image
        for image in range(bounds.shape[0]):

            # and each row
            for row in range(bounds.shape[1]):

                # get the bounds
                polygon = bounds[image][row]

                # check for discrepancy
                if any([(entry < -170) for entry in polygon]) and any([entry > 170 for entry in polygon]):

                    # adjust bounds
                    bounds[image][row] = numpy.where(polygon > 170, polygon, polygon + 360)

        return bounds

    def _depopulate(self):
        """Depopulate the instance.

        Arguments:
            None

        Returns:
            None

        Populates:
            self
        """

        # depopulate the instances
        while len(self) > 0:

            # pop off
            self.pop()

        return None

    def _diagram(self, features, total, level=0, limit=5):
        """Print a chart of the contents with size information.

        Arguments:
            features: list of feature instances
            total: total size of all features
            level: int, nesting level
            limit: int, total number of leaves to show

        Returns:
            dict
        """

        # begin blank blueprint
        blueprint = []

        # separate into branches and leaves
        leaves = [feature for feature in features if len(feature.route) <= level + 1]
        twigs = [feature for feature in features if len(feature.route) > level + 1]

        # add entries for each visible leaf
        visibles = []
        for leaf in leaves[:limit]:

            # get the data attributes
            data = leaf.distil()
            shape = data.shape
            form = data.dtype
            name = leaf.name
            tab = ' ' * 10 * level

            # separate name at captials
            #name = ''.join([' ' + letter if letter.isupper() else letter for letter in name]).strip()
            visibles.append(name)

            # create entry
            formats = (tab, shape, name, form)
            entry = '{}{} {} ( {} )'.format(*formats)
            blueprint.append(entry)

        # for every other entry
        invisibles = []
        ellipses = False
        for leaf in leaves[limit:]:

            # set ellipsese flag
            ellipses = True

            # get words from name
            name = ''.join([' ' + letter if letter.isupper() else letter for letter in leaf.name]).strip()
            invisibles.append(name)

        # if ellipses:
        if ellipses:

            # join all visible names
            visibles = ' '.join(visibles)

            # get all novel wprds amongst invisible phrases, one position at a time
            novels = []
            index = 0
            phrases = [name.split() for name in invisibles]
            while len(phrases) > 0:

                # subset phrsess
                phrases = [phrase for phrase in phrases if len(phrase) > index]

                # get all unique elements at that position
                elements = [phrase[index] for phrase in phrases if phrase[index] not in visibles]
                novels += self._skim(elements, maintain=True)

                # increment for next position
                index += 1

            # add ellipses entry, based on all words un accounted for
            tab = ' ' * 10 * level

            # go through chunks of 8
            length = 8
            chunks = math.ceil(len(novels) / length)
            for chunk in range(chunks):

                # add entry
                entry = '{}...( {} )'.format(tab, ' '.join(novels[chunk * length: length + chunk * length]))
                blueprint.append(entry)

            # add spacer
            blueprint.append('')

        # group twigs according to route
        groups = self._group(twigs, lambda feature: feature.route[level])
        for twig, members in groups.items():

            # add next iteration
            tab = ' ' * 10 * level
            size = sum([feature.data.size * feature.data.itemsize for feature in members])
            kilobytes = round(size / 1024, 2)
            percent = round(100 * size / total, 2)
            blueprint.append('{}( {} % ) {} ( {} kb, {} entries )'.format(tab, percent, twig, kilobytes, len(members)))
            blueprint += self._diagram(members, total, level + 1, limit)

        return blueprint

    def _differ(self, first, second):
        """Get a list of differences between two arrays.

        Arguments:
            first: numpy array
            second: numpy array

        Returns:
            list of ( float, float ) tuples
        """

        # get a mask where first and sceond differ
        mask = (first != second)

        # get pixels for differences
        pixels = list(zip(*numpy.where(mask)))

        # collect pairs
        pairs = [(pixel, one, two) for pixel, one, two in zip(pixels, first[mask], second[mask])]

        return pairs

    def _excise(self, polygons, tracer):
        """Excise bad polygons from the dataset.

        Arguments:
            polygons: numpy array
            tracer: numpy array

        Returns:
            tuple of numpy arrays
        """

        # extract longitudes from polygons
        longitudes = polygons[:, :, 4:]

        # calculation standard deviations
        deviation = longitudes.std(axis=2)
        mask = (deviation < 20)

        # apply mask
        polygonsii = polygons[mask]
        tracerii = tracer[mask]

        return polygonsii, tracerii

    def _fetch(self, path):
        """Link to the contents of an hdf5 file.

        Arguments:
            path: str, file path
            net: boolean, use netcdf

        Returns:
            hdf5 file
        """

        # if netCDF4
        if self.net:

            # fetch from netCDF4
            five = netCDF4.Dataset(path)

        # otherwise
        else:

            # open up the hdf5 file with h5py
            five = h5py.File(path, 'r')

        return five

    def _frame(self, latitude, longitude):
        """Construct the corners of polygons from latitude and longitude coordinates.

        Arguments:
            latitude: numpy array
            longitude: numpy array

        Returns:
            dict of numpy arrays, the corner points
        """

        # get main shape
        shape = latitude.shape

        # # adjust longitude for negative values
        # longitude = numpy.where(longitude < 0, longitude + 360, longitude)

        # initialize all four corners, with one layer for latitude and one for longitude
        northwest = numpy.zeros((*shape, 2))
        northeast = numpy.zeros((*shape, 2))
        southwest = numpy.zeros((*shape, 2))
        southeast = numpy.zeros((*shape, 2))

        # create latitude frame, with one more row and image on each side
        frame = numpy.zeros((shape[0] + 2, shape[1] + 2))
        frame[1: shape[0] + 1, 1: shape[1] + 1] = latitude

        # extend sides of frame by extrapolation
        frame[1: -1, 0] = 2 * latitude[:, 0] - latitude[:, 1]
        frame[1: -1, -1] = 2 * latitude[:, -1] - latitude[:, -2]
        frame[0, 1:-1] = 2 * latitude[0, :] - latitude[1, :]
        frame[-1, 1:-1] = 2 * latitude[-1, :] - latitude[-2, :]

        # extend corners
        frame[0, 0] = 2 * frame[1, 1] - frame[2, 2]
        frame[0, -1] = 2 * frame[1, -2] - frame[2, -3]
        frame[-1, 0] = 2 * frame[-2, 1] - frame[-3, 2]
        frame[-1, -1] = 2 * frame[-2, -2] - frame[-3, -3]

        # create longitude frame, with one more row and image on each side
        frameii = numpy.zeros((shape[0] + 2, shape[1] + 2))
        frameii[1: shape[0] + 1, 1: shape[1] + 1] = longitude

        # extend sides of frame by extrapolation
        frameii[1: -1, 0] = 2 * longitude[:, 0] - longitude[:, 1]
        frameii[1: -1, -1] = 2 * longitude[:, -1] - longitude[:, -2]
        frameii[0, 1:-1] = 2 * longitude[0, :] - longitude[1, :]
        frameii[-1, 1:-1] = 2 * longitude[-1, :] - longitude[-2, :]

        # extend corners
        frameii[0, 0] = 2 * frameii[1, 1] - frameii[2, 2]
        frameii[0, -1] = 2 * frameii[1, -2] - frameii[2, -3]
        frameii[-1, 0] = 2 * frameii[-2, 1] - frameii[-3, 2]
        frameii[-1, -1] = 2 * frameii[-2, -2] - frameii[-3, -3]

        # populate interior polygon corners image by image
        for image in range(0, shape[0]):

            # and row by row
            for row in range(0, shape[1]):

                # frame indices are off by 1
                imageii = image + 1
                rowii = row + 1

                # by averaging latitude frame at diagonals
                northwest[image, row, 0] = (frame[imageii, rowii] + frame[imageii + 1][rowii - 1]) / 2
                northeast[image, row, 0] = (frame[imageii, rowii] + frame[imageii + 1][rowii + 1]) / 2
                southwest[image, row, 0] = (frame[imageii, rowii] + frame[imageii - 1][rowii - 1]) / 2
                southeast[image, row, 0] = (frame[imageii, rowii] + frame[imageii - 1][rowii + 1]) / 2

                # and by averaging longitude longitudes at diagonals
                northwest[image, row, 1] = (frameii[imageii, rowii] + frameii[imageii + 1][rowii - 1]) / 2
                northeast[image, row, 1] = (frameii[imageii, rowii] + frameii[imageii + 1][rowii + 1]) / 2
                southwest[image, row, 1] = (frameii[imageii, rowii] + frameii[imageii - 1][rowii - 1]) / 2
                southeast[image, row, 1] = (frameii[imageii, rowii] + frameii[imageii - 1][rowii + 1]) / 2

        # fix longitude edge cases, by looking for crisscrossing, and truncating
        northwest[:, :, 1] = numpy.where(northwest[:, :, 1] < longitude, northwest[:, :, 1], longitude)
        northeast[:, :, 1] = numpy.where(northeast[:, :, 1] > longitude, northeast[:, :, 1], longitude)
        southwest[:, :, 1] = numpy.where(southwest[:, :, 1] < longitude, southwest[:, :, 1], longitude)
        southeast[:, :, 1] = numpy.where(southeast[:, :, 1] > longitude, southeast[:, :, 1], longitude)

        # average all overlapping corners
        overlap = southeast[1:, :-1, :] + southwest[1:, 1:, :] + northwest[:-1, 1:, :] + northeast[:-1, :-1, :]
        average = overlap / 4

        # transfer average
        southeast[1:, :-1, :] = average
        southwest[1:, 1:, :] = average
        northwest[:-1, 1:, :] = average
        northeast[:-1, :-1, :] = average

        # average western edge
        average = (southwest[1:, 0, :] + northwest[:-1, 0, :]) / 2
        southwest[1:, 0, :] = average
        northwest[:-1, 0, :] = average

        # average eastern edge
        average = (southeast[1:, -1, :] + northeast[:-1, -1, :]) / 2
        southeast[1:, -1, :] = average
        northeast[:-1, -1, :] = average

        # average northern edge
        average = (northwest[-1, 1:, :] + northeast[-1, :-1, :]) / 2
        northwest[-1, 1:, :] = average
        northeast[-1, :-1, :] = average

        # average southern edge
        average = (southwest[0, 1:, :] + southeast[0, :-1, :]) / 2
        southwest[0, 1:, :] = average
        southeast[0, :-1, :] = average

        # # subtact 360 from any longitudes over 180
        # northwest[:, :, 1] = numpy.where(northwest[:, :, 1] >= 180, northwest[:, :, 1] - 360, northwest[:, :, 1])
        # northeast[:, :, 1] = numpy.where(northeast[:, :, 1] >= 180, northeast[:, :, 1] - 360, northeast[:, :, 1])
        # southeast[:, :, 1] = numpy.where(southeast[:, :, 1] >= 180, southeast[:, :, 1] - 360, southeast[:, :, 1])
        # southwest[:, :, 1] = numpy.where(southwest[:, :, 1] >= 180, southwest[:, :, 1] - 360, southwest[:, :, 1])

        # collect corners
        corners = {'northwest': northwest, 'northeast': northeast}
        corners.update({'southwest': southwest, 'southeast': southeast})

        return corners

    def _garner(self, data, path, route=None, mode=None, scan=False):
        """Gather all routes and shapes from a datafile.

        Arguments:
            data: dict or h5
            path: str, file path
            route=None: current route
            mode=None: mode to restrict gathering with

        Returns:
            list of dicts
        """

        # initialize routes for first round
        route = route or []

        # initialize collection
        collection = []

        # test mode condition
        allow = True
        if mode:

            # if the specific mode is not in the route
            address = ':'.join(route)
            if len(route) > 2 and mode not in address:

                # stop gathering
                allow = False

        # only proceed if allowed by mode condition
        if allow:

            # try to
            try:

                # get all fields
                for field in data.groups:

                    # if scan
                    if scan:

                        # print the field
                        self._print(field)

                    # # check for variables
                    # slash = '/'.join([''] + route + [field])
                    #
                    # self._print(slash)

                    # get variables
                    variables = data[field].variables

                    # for each variable
                    for variable, info in variables.items():

                        # begin attributes with dimensions
                        attributes = {'dimensions': info.dimensions, 'netCDF': True}
                        for attribute in info.ncattrs():

                            # add to attributes
                            attributes[attribute] = info.getncattr(attribute)

                        # add entry to collection
                        parameters = {'route': route + [field, variable], 'shape': info.shape, 'path': path}
                        parameters.update({'attributes': attributes, 'format': info.dtype})
                        feature = Feature(**parameters)
                        collection.append(feature)

                    # get dimensions
                    dimensions = data[field].dimensions

                    # for each variable
                    for dimension, info in dimensions.items():

                        # begin attributes with dimensions
                        attributes = {'netCDF': True, 'dimension': True}
                        attributes.update({'name': info.name, 'size': info.size, 'group': info.group().name})
                        attributes.update({'isunlimited': info.isunlimited()})

                        # add net attribute
                        attributes.update({'net': True})

                        # add entry to collection
                        parameters = {'route': route + [field, dimension], 'shape': (info.size,), 'path': path}
                        parameters.update({'attributes': attributes, 'format': int})
                        feature = Feature(**parameters)
                        collection.append(feature)

                    # get groups
                    groups = data[field].groups

                    # if there are no groups, variables or dimensions
                    if not groups and not variables and not dimensions:

                        # begin attributes with dimensions
                        info = {attribute: data[field].getncattr(attribute) for attribute in data[field].ncattrs()}
                        attributes = {'netCDF': True, 'metadata': True}
                        attributes.update(info)

                        # add entry to collection
                        parameters = {'route': route + [field], 'shape': (0,), 'path': path}
                        parameters.update({'attributes': attributes, 'format': str})
                        feature = Feature(**parameters)
                        collection.append(feature)

                    # and add each field to the collection
                    collection += self._garner(data[field], path, route + [field], mode=mode, scan=scan)

            # unless it is an endpoint
            except AttributeError:

                self._print('error: {}'.format(route))

        return collection

    def _gather(self, data, path, route=None, mode=None, scan=False):
        """Gather all routes and shapes from a datafile.

        Arguments:
            data: dict or h5
            path: str, file path
            route=None: current route
            mode=None: mode to restrict gathering with

        Returns:
            list of dicts
        """

        # initialize routes for first round
        route = route or []

        # initialize collection
        collection = []

        # test mode condition
        allow = True
        if mode:

            # if the specific mode is not in the route
            address = ':'.join(route)
            if len(route) > 2 and mode not in address:

                # stop gathering
                allow = False

        # only proceed if allowed by mode condition
        if allow:

            # try to
            try:

                # get all fields
                for field in data.keys():

                    # if scan
                    if scan:

                        # print the field
                        self._print(field)

                    # and add each field to the collection
                    collection += self._gather(data[field], path, route + [field], mode=mode, scan=scan)

            # unless it is an endpoint
            except AttributeError:

                # try to
                try:

                    # determine shape and type
                    shape = data.shape
                    form = data.dtype

                    # if scanning
                    if scan:

                        # print data
                        for name, value in data.attrs.items():

                            # print
                            self._print(name, value)

                    # get attributes, skipping python2 problematic fields for now
                    problems = ('DIMENSION_LIST', 'REFERENCE_LIST')
                    attributes = {name: value for name, value in data.attrs.items() if name not in problems}

                    # if the type is simple
                    if len(form) < 1:

                        # add entry to collection
                        parameters = {'route': route, 'shape': shape, 'path': path}
                        parameters.update({'attributes': attributes, 'format': form})
                        feature = Feature(**parameters)
                        collection.append(feature)

                    # otherwise the type is complex
                    else:

                        # convert to numpy dtype
                        conversion = numpy.dtype(form)

                        # get attributes
                        problems = ('DIMENSION_LIST', 'REFERENCE_LIST')
                        attributes = {name: value for name, value in data.attrs.items() if name not in problems}

                        # for each member of the type
                        for name in conversion.names:

                            # get form
                            form = conversion.fields[name]

                            # create feature
                            parameters = {'route': route + [name], 'shape': shape, 'path': path}
                            parameters.update({'attributes': attributes, 'format': form})
                            feature = Feature(**parameters)
                            collection.append(feature)

                # otherwise assume non valid data
                except AttributeError:

                    # and skip
                    pass

        return collection

    def _grow(self, level=0, destination=None, subset=None):
        """Make a tree of the file contents of an h5 file, to a certain level.

        Arguments:
            level=2: the max nesting level to see
            destination: str, file path for destination
            subset: str, slashed field list

        Returns:
            None
        """

        # begin tree
        tree = {}

        # for each feature
        for feature in self:

            # decompose slash
            slash = feature.slash.split('/')

            # for all but the last
            branch = tree
            for member in slash[:-1]:

                # move to next member
                branch.setdefault(member, {})
                branch = branch[member]

            # at last entry, add shape
            branch[feature.name] = feature.shape

        # if a subset is given
        if subset:

            # for each member
            for field in subset.split('/'):

                # subset the tree
                tree = tree[field]

        # crete tree
        self._look(tree, level, destination)

        # set tree
        self.tree = tree

        return None

    def _insert(self, array, destination, name, category):
        """Insert a feature into an hdf4 file.

        Arguments:
            array: numpy array,
            destination: str, filepath
            name: str, name of science
            category: list of str, address of group

        Returns:
            None
        """

        # begin hdf4 file
        four = HDF(destination, HC.WRITE | HC.CREATE)
        science = SD(destination, SDC.WRITE | SDC.CREATE)
        groups = four.vgstart()
        tables = four.vstart()

        # scan for registry
        registry = list(self._scan(four).items())

        # sort for best match
        registry.sort(key=lambda pair: category == pair[1], reverse=True)
        identity = registry[0][0]

        # find the correctly named sd
        group = groups.attach(identity)
        tags = group.tagrefs()
        for tag, number in tags:

            # try to
            try:

                # get attributes
                dataset = science.select(science.reftoindex(number))
                nameii, rank, dims, type, _ = dataset.info()

                # insert new data
                if nameii == name:

                    # insert
                    print('inserting {}...'.format(name))
                    dataset[:] = array[0]

            # otherwise
            except HDF4Error as error:

                # alert
                pass
                #self._print(error)

        # close all apis
        groups.end()
        tables.end()
        science.end()
        four.close()

        return None

    def _orient(self, degrees, east=False):
        """Construct an oriented latitude or longitude tag from a signed decimal.

        Arguments:
            degrees: float, degrees latitutde
            east: boolean, use east west for longitude instead?

        Returns:
            string
        """

        # construct tag assuming north - south
        tag = self._pad(int(abs(degrees)), 2) + 'N' * (degrees >= 0) + 'S' * (degrees < 0)

        # if not east west
        if east:

            # construct tag
            tag = self._pad(int(abs(degrees)), 3) + 'E' * (degrees >= 0) + 'W' * (degrees < 0)

        return tag

    def _orientate(self, latitude, longitude):
        """Create corners by orienting latitude bounds and longitude bounds.

        Arguments:
            latitude: numpy array, latitude bounds
            longitude: numpy array, longitude bounds

        Returns:
            dict of numpy arrays
        """

        # add 360 to all negative longitude bounnds
        longitude = numpy.where(longitude < 0, longitude + 360, longitude)

        # get all cardinal directions
        cardinals = {'south': latitude.min(axis=2), 'north': latitude.max(axis=2)}
        cardinals.update({'west': longitude.min(axis=2), 'east': longitude.max(axis=2)})

        # begin corners with zeros
        compasses = ('northwest', 'northeast', 'southeast', 'southwest')
        compassesii = ('northeast', 'southeast', 'southwest', 'northwest')
        corners = {compass: numpy.zeros((latitude.shape[0], latitude.shape[1], 2)) for compass in compasses}

        # double up bounds
        latitude = latitude.transpose(2, 1, 0)
        longitude = longitude.transpose(2, 1, 0)
        latitude = numpy.vstack([latitude, latitude, latitude]).transpose(2, 1, 0)
        longitude = numpy.vstack([longitude, longitude, longitude]).transpose(2, 1, 0)

        # go through each position
        for index in range(4):

            # check against corners and add if south and west are the same
            mask = (latitude[:, :, index] == cardinals['north'])
            maskii = (longitude[:, :, index] < longitude[:, :, index + 2])
            masque = numpy.logical_and(mask, maskii)

            # for each point
            for way, compass in enumerate(compasses):

                # add corners
                corners[compass][:, :, 0] = numpy.where(masque, latitude[:, :, index + way], corners[compass][:, :, 0])
                corners[compass][:, :, 1] = numpy.where(masque, longitude[:, :, index + way], corners[compass][:, :, 1])

            # check against corners and add if south and west are the same
            mask = (latitude[:, :, index] == cardinals['north'])
            maskii = (longitude[:, :, index] > longitude[:, :, index + 2])
            masque = numpy.logical_and(mask, maskii)

            # for each point
            for way, compass in enumerate(compassesii):

                # add corners
                corners[compass][:, :, 0] = numpy.where(masque, latitude[:, :, index + way], corners[compass][:, :, 0])
                corners[compass][:, :, 1] = numpy.where(masque, longitude[:, :, index + way], corners[compass][:, :, 1])

        # subtract 360 from longitudes
        for compass in compasses:

            # subtract form longitude
            corner = corners[compass][:, :, 1]
            corners[compass][:, :, 1] = numpy.where(corner > 180, corner - 360, corner)

        return corners

    def _parse(self, path):
        """Parse file name for orbital context information.

        Arguments:
            path: str, filepath name

        Returns:
            dict
        """

        # extract the orbital details from the path name using regex for YYYYmMMDDtHHmm-oNNNNNN
        details = re.search('[0-9]{4}m[0-9]{4}t[0-9]{4,6}-o[0-9]{5,6}', path).group()

        # unpack orbit number and begin context dictionary
        number = re.search('-o[0-9]{5,6}', details).group()[2:]
        context = {'number': int(number)}

        # unpack time information
        year = re.search('[0-9]{4}m', details).group()[:4]
        month = re.search('m[0-9]{4}t', details).group()[1:3]
        date = re.search('m[0-9]{4}t', details).group()[3:5]
        hour = re.search('t[0-9]{4}', details).group()[1:3]
        minute = re.search('t[0-9]{4}', details).group()[3:]

        # default second counter to zero, as it is not in the pathname
        second = 0.0

        # use datetime to get day of year
        clock = datetime.datetime(int(year), int(month), int(date), int(hour), int(minute))
        day = float(clock.utctimetuple().tm_yday)
        stamp = float(calendar.timegm(clock.utctimetuple()))
        milliseconds = stamp * 1000

        # calculate the time as a timestamp
        beginning = float(calendar.timegm(datetime.datetime(int(year), 1, 1).utctimetuple()))
        ending = float(calendar.timegm(datetime.datetime(int(year) + 1, 1, 1).utctimetuple()))
        fraction = float(year) + (stamp - beginning) / (ending - beginning)

        # update context with time information
        names = ['yr', 'mon', 'd_o_m', 'hr', 'min', 'sec', 'd_o_y']
        data = (year, month, date, hour, minute, second, day)
        context.update({'start_{}'.format(name): int(datum) for name, datum in zip(names, data)})

        # add fraction in decimals and startimefryr in milliseconds
        context.update({'start_year_fraction': float(fraction)})
        context.update({'start_time_fr_yr': float(milliseconds)})

        # add orbit prefix
        context = {'orbit_{}'.format(name): datum for name, datum in context.items()}

        return context

    def _pick(self, word):
        """Pick the first path with the given keyword.

        Arguments:
            word: str, keyword

        Returns:
            list of str
        """

        # get list of path indices
        names = [path for path in self.paths if word in path.split('/')[-1]]

        return names

    def _pin(self, targets, arrays, number=5, weights=None):
        """Pinpoint the coordinates that are closest to the target in an array.

        Arguments:
            targets: float, target value
            arrays: numpy array
            number: number of entries to retrieve
            weights: list of weights

        Returns:
            list of ( int ) tuples
        """

        # try to
        try:

            # access first target
            _ = targets[0]

        # unless not a list
        except (TypeError, IndexError):

            # in which case, recast as list
            targets = [targets]
            arrays = [arrays]

        # set weights
        weights = weights or [1.0 for _ in targets]

        # compute the squared distance from target
        squares = [(weight * (array - target)) ** 2 for weight, array, target in zip(weights, arrays, targets)]
        summation = sum(squares)

        # get the shape
        shape = summation.shape

        # get ordering
        order = summation.flatten().argsort()

        # for each entry
        indices = []
        for entry in order[:number]:

            # get the indices
            remainder = entry
            point = []
            for place, span in enumerate(shape):

                # get sum of all shapes past the relevant
                block = numpy.prod(shape[place + 1:])
                position = int(numpy.floor(remainder / block))

                # add to point
                point.append(position)

                # recalculate remainder
                remainder -= (position * block)

            # add point to indices
            indices.append(tuple(point))

        return indices

    def _point(self, vector, table, nodes, show=False):
        """Linearly Interpolate a vector of values onto a table.

        Arguments:
            vector: unpacked list of floats
            table: numpy.array of floats, the interpolation table
            nodes: list of lists of floats, the node coordinates
            show: boolean, show nodes and brackets?

        Returns:
            float, the interpolated value
        """

        # For each entry
        brackets = []
        for quantity, node in zip(vector, nodes):

            # if descending
            if node[-1] < node[0]:

                # get the bracket indices, inclusive
                last = [quantity > tick for tick in node].index(True)
                first = last - 1
                bracket = (first, last)

            # otherwise, assume ascending
            else:

                # get the bracket indices, inclusive
                last = [quantity < tick for tick in node].index(True)
                first = last - 1
                bracket = (first, last)

            # if first bracket is too small
            if first < 0:

                # set to last two
                bracket = (0, 1)

            # if last bracket is too big
            if last > len(node) - 1:

                # set to last two
                bracket = (len(node) - 2, len(node) - 1)

            # append to brackets
            brackets.append(bracket)

            # if showing
            if show:

                # print bracket
                self._print(quantity)
                self._print(bracket)
                self._print(node)
                self._print('')

        # create miniature table
        cube = table
        for axis, bracket in enumerate(brackets):

            # take subset
            cube = numpy.take(cube, bracket, axis)

        # for each bracket
        result = cube
        for bracket, node, quantity in zip(brackets, nodes, vector):

            # unpack bracket
            first, last = bracket

            # find the weight, a ( 1 - w ) + b ( w ) = r, w = ( r - a ) / ( b - a )
            weight = (quantity - node[first]) / (node[last] - node[first])

            # reduce along axis
            result = (1 - weight) * result[0] + weight * result[1]

        return result

    def _polymerize(self, latitude, longitude, resolution):
        """Construct polygons from latitude and longitude bounds.

        Arguments:
            latitude: numpy array of latitude bounds
            longitude: numpy array of longitude bounds
            resolution: int, vertical compression factor

        Returns:
            numpy.array
        """

        # squeeze arrays
        latitude = latitude.squeeze()
        longitude = longitude.squeeze()

        # if bounds are given
        if len(latitude.shape) > 2:

            # orient the bounds
            corners = self._orientate(latitude, longitude)

        # otherwise
        else:

            # get corners from latitude and longitude points
            corners = self._frame(latitude, longitude)

        # compress corners vertically
        corners = self._compress(corners, resolution, 1)

        # construct bounds
        compasses = ('southwest', 'southeast', 'northeast', 'northwest')
        arrays = [corners[compass][:, :, 0] for compass in compasses]
        bounds = numpy.stack(arrays, axis=2)
        arrays = [corners[compass][:, :, 1] for compass in compasses]
        boundsii = numpy.stack(arrays, axis=2)

        # adjust polygon  boundaries to avoid crossing dateline
        boundsii = self._cross(boundsii)

        # make polygons
        polygons = numpy.dstack([bounds, boundsii])

        return polygons

    def _populate(self, features, discard=True):
        """Populate the instance with feature records.

        Arguments:
            features: list of feature instances
            discard: boolean, depopulate first?

        Returns:
            None

        Populates:
            self
        """

        # if empty is set true
        if discard:

            # depopulate the instances
            self._depopulate()
            self.reference = {}

        # populate with new instances
        [self.append(feature) for feature in features]

        # add to reference
        self.reference = self._refer(features, self.reference)

        return None

    def _post(self, path, error, destination):
        """Post an entry to the log file.

        Arguments:
            path: str, file path
            error: str, error text

        Returns:
            None

        Populates:
            self.log
        """

        # add error to log
        self.log.append(' ')
        self.log.append(str(datetime.datetime.now()))
        self.log.append(path)
        self.log.append(str(error))
        self.log.append(str(error.args))

        # save log file
        self._jot(self.log, destination, 'a')

        return None

    def _project(self, novel, abscissa, ordinate):
        """Project a new abscissa onto another abscissas and ordinate by linear interpolation.

        Arguments:
            novel: list of floats, new x values
            abscissas: list of floats, x values
            ordinate: list of floats, y values

        Returns:
            numpy array, new y values
        """

        # create squared arrays with new axis for new shape
        ordinateii = numpy.array([ordinate] * len(novel))
        abscissaii = numpy.array([abscissa] * len(novel))
        novelii = numpy.array([novel] * len(abscissa)).transpose(1, 0)

        # create lesser and greater matrices
        lesser = abscissaii < novelii
        greater = abscissaii > novelii

        # make sure at least the first lesser value is true
        lesser[:, 0] = True

        # make sure at least the last greater value is true
        greater[:, -1] = True

        # create left and right mask
        left = lesser & numpy.roll(greater, -1, axis=1)
        right = numpy.roll(lesser, 1, axis=1) & greater

        # create horizontal brackets
        horizontal = (abscissaii * left.astype(int)).sum(axis=1)
        horizontalii = (abscissaii * right.astype(int)).sum(axis=1)

        # create vertical brackets
        vertical = (ordinateii * left.astype(int)).sum(axis=1)
        verticalii = (ordinateii * right.astype(int)).sum(axis=1)

        # perform interpolation
        slope = (verticalii - vertical) / (horizontalii - horizontal)
        projection = vertical + slope * (novel - horizontal)

        return projection

    def _register(self):
        """Construct all file paths.

        Arguments:
            None

        Returns:
            None

        Populates:
            self.paths
        """

        # get all file paths
        paths = []
        self._print('collecting paths...')

        # if a source directory is given
        if self.source:

            # and start and finish are given
            if self.start and self.finish:

                # get all folders
                folders = self._see(self.source)
                for folder in folders:

                    # try to:
                    try:

                        # check for inclusion in range
                        if int(self.start) <= int(folder.split('/')[-1]) <= int(self.finish):

                            # add to paths
                            paths += self._see(folder)

                    # unless the folder is not compatible
                    except ValueError:

                        # in which case, skip
                        pass

            # otherwise
            else:

                # try to
                try:

                    # get all paths in the directory
                    paths += self._see(self.source)

                # unless it is not a directory
                except NotADirectoryError:

                    # in which case, assume it is a file
                    paths += [self.source]

        # retain only certain file types
        paths = [path for path in paths if any([path.endswith(extension) for extension in self.extensions])]
        paths.sort()

        # print paths
        self._tell(paths)
        self._print('{} paths collected.\n'.format(len(paths)))

        # set attribute
        self.paths = paths

        return None

    def _resolve(self, array, resolution):
        """Pare down an array to only mod zero entries based on resolution.

        Arguments:
            array: numpy array
            resolution: int, number of pixels to combine

        Returns:
            numpy array
        """

        # get indices at mod 0
        indices = numpy.array([index for index in range(array.shape[0]) if index % resolution == 0])

        # apply to array
        resolution = array[indices]

        return resolution

    def _refer(self, features, reference=None):
        """Create a reference for a set of features for quicker lookup.

        Arguments:
            None

        Returns
            dict
        """

        # begin reference
        reference = reference or {}

        # for each feature
        for feature in features:

            # add an entry for full slash
            slashes = reference.setdefault(feature.slash, [])
            slashes.append(feature)

            # add entry for single name
            names = reference.setdefault(feature.name, [])
            names.append(feature)

        return reference

    def _relate(self, first, second, percent=False):
        """Calculate the percent difference between two arrays.

        Arguments:
            first: numpy array
            second: numpy array
            percent: boolean, compute percent difference instead of differecnce?

        Returns:
            numpy array
        """

        # calculate difference
        relation = second - first

        # if difference
        if percent:

            # calculate difference
            relation = 100 * ((second / first) - 1)

        return relation

    def _round(self, quantity, digits=2, up=False):
        """Round a value based on digits.

        Arguments:
            quantity: float
            digits: int, number of post decimal digits to keep
            up: boolean, round up?

        Returns:
            None
        """

        # multiple by power of digtsp
        power = 10 ** digits
        approximation = quantity * 10 ** digits

        # if rounding up
        if up:

            # round up
            approximation = math.ceil(approximation)

        # otherwise
        else:

            # round down
            approximation = math.floor(approximation)

        # divide by power
        approximation = approximation / power

        return approximation

    def _scan(self, four):
        """Scan all reference ids from the hdf4 file.

        Arguments:
            four: hdf4 file pointer.

        Returns:
            list of ints
        """

        # begin list of references
        references = []

        # open HDF4 group instance
        groups = four.vgstart()

        # begin with -1 and loop through references
        reference = -1
        while True:

            # try to
            try:

                # get new reference number by feeding in old
                reference = groups.getid(reference)
                references.append(reference)

            # unless the end is reached
            except HDF4Error:

                # in which case close the groups and break
                break

        # get the name of each member
        names = [groups.attach(reference)._name for reference in references]

        # try to
        try:

            # get terminal index where internal groups occur
            terminus = names.index('RIG0.0')
            names = names[:terminus]
            references = references[:terminus]

        # unless abscent
        except ValueError:

            # in which case, keep all names
            pass

        # make registry of routes
        registry = {reference: [name] for reference, name in zip(references, names)}

        # go through each referencre
        for reference, name in zip(references, names):

            # search for tags
            tags = groups.attach(reference).tagrefs()
            if all([tag[0] == HC.DFTAG_VG for tag in tags]):

                # append group name to beginning
                for tag in tags:

                    # add group name
                    registry[tag[1]] = [name] + registry[tag[1]]

                # delete from registry
                del(registry[reference])

        # close groups
        groups.end()

        return registry

    def _scrounge(self, path, route, indices):
        """Scrounge particular data from hdf4 without conversion.

        Arguments:
            path: hdf4 path
            route: list of str, partial field names
            indices: list of int, remaining indices

        Returns:
            float / int
        """

        # default code to 0
        code = 0

        # convert path to hdf4
        four = HDF(path)
        science = SD(path)

        # initialize group and table interfaces
        groups = four.vgstart()
        tables = four.vstart()

        # get all groups
        registry = self._scan(four)

        # get group registration by finding the route
        steps = route[:-1]
        parameter = route[-1]
        subset = [number for number, address in registry.items() if all([step in ' '.join(address) for step in steps])]

        # if there are available entries
        if len(subset) > 1:

            # get tags
            group = groups.attach(subset[0])
            tags = group.tagrefs()
            for tag, number in tags:

                # check for scientific dataset
                if tag == HC.DFTAG_NDG:

                    # get attributes
                    dataset = science.select(science.reftoindex(number))
                    name, rank, dims, type, _ = dataset.info()

                    # check for specific name
                    if name == parameter:

                        # get data and add to feature
                        parcel = dataset.get()

                        # for each index
                        datum = parcel
                        for index in indices:

                            # get the datum
                            datum = datum[index]

                        # set code
                        code = datum

                # check for vdata
                elif tag == HC.DFTAG_VH:

                    # get attributes
                    table = tables.attach(number)
                    records, _, _, size, name = table.inquire()

                    # check for specific name
                    if name == parameter:

                        # read all lines to get data
                        descending = lambda line: line[0] if len(line) < 2 else line
                        parcel = [descending(table.read()[0]) for _ in range(records)]

                        # for each index
                        datum = parcel
                        for index in indices:

                            # get the datum
                            datum = datum[index]

                        # set code
                        code = datum

                    # detach table
                    table.detach()

            # detach group
            group.detach()

        # close groups, tables, and files
        groups.end()
        tables.end()
        science.end()
        four.close()

        return code

    def _seek(self, group, names=None):
        """Seek nested metadata attributes in a netcdf group.

        Arguments:
            group: netCDF group
            names: list of previous groups

        Returns:
            None
        """

        # set default names
        names = names or []

        # print name
        self._print('')
        self._print('{}: '.format('/'.join(names)))

        # get all attributes
        attributes = group.ncattrs()

        # for each attribute
        [self._print('{}: {}'.format(attribute, group.getncattr(attribute))) for attribute in attributes]

        # get all nested gruops
        groups = group.groups
        for key in groups.keys():

            # check next level
            self._seek(group[key], names + [key])

        return None

    def _serpentize(self, name):
        """Make a camel case name into snake case.

        Arguments:
            name: str

        Returns:
            str
        """

        # split at underscore
        sections = name.split('_')

        # for each section
        snakes = []
        for section in sections:

            # capitalize first letter
            section = section[0].upper() + section[1:]

            # findall capitalized sections
            snakes += re.findall('[A-Z][a-z,0-9]*', section)

        # make lower case
        snakes = [snake.lower() for snake in snakes]

        # join at underscore
        serpent = '_'.join(snakes)

        # # split on colons
        # snakes = []
        # sections = name.split('_')
        # for section in sections:
        #
        #     # split by capitals
        #     words = []
        #     point = 0
        #     for index, letter in enumerate(section):
        #
        #         # if it is a capital
        #         if letter == letter.upper():
        #
        #             # add bit to words
        #             words.append(section[point: index])
        #             point = index
        #
        #     # add final chunk
        #     words.append(section[point: len(section)])
        #
        #     # lower each part
        #     words = [word.lower() for word in words if len(word) > 0]
        #
        #     # rejoin
        #     snake = '_'.join(words)
        #     snakes.append(snake)
        #
        # # connect main name with underscore
        # serpent = '_'.join(snakes)

        return serpent

    def _stage(self, path):
        """Get the file meta attributes from the file name.

        Arguments:
            None

        Returns:
            dict
        """

        # extract the orbit specifics
        date = re.search('[0-9]{4}m[0-9]{4}t[0-9]{4}', path)
        orbit = re.search('-o[0-9]{5,6}', path)
        product = re.search('OM[A-Z,0-9]{3,10}', path)
        production = re.search('[0-9]{4}m[0-9]{4}t[0-9]{4}', path.split('_')[-1])
        version = re.search('_v[0-9]{3,4}', path)
        extension = '.{}'.format(path.split('.')[-1])

        # gather up the date details
        details = {}
        details['date'] = date.group() if date else '_'
        details['year'] = details['date'][:4] if len(str(date)) > 1 else '_'
        details['month'] = details['date'][:7] if len(str(date)) > 1 else '_'
        details['day'] = details['date'][:9] if len(str(date)) > 1 else '_'

        # and other details
        details['orbit'] = ('00' + orbit.group().split('-o')[1])[-6:] if orbit else '_'
        details['product'] = product.group() if product else '_'
        details['production'] = production.group() if production else '_'
        details['version'] = version.group().strip('_v') if version else '_'
        details['collection'] = '3' if '3' in details['version'] else '4'
        details['extension'] = extension

        return details

    def _take(self, *indices, paths=None):
        """Take a subset of paths based on ther index.

        Arguments:
            *indices: unpacked list of ints
            paths: list of paths

        Returns:
            list of str
        """

        # set default paths
        paths = paths or self.paths

        # get the paths
        subset = [paths[index] for index in indices]

        return subset

    def _unite(self, *bits):
        """Recombine a list of bits into a decimal.

        Arguments:
            *bits: list of ints, the bit flags

        Returns:
            int
        """

        # default combination to 0
        combination = 0

        # for each bit
        for bit in bits:

            # add 2 to the power of the bit
            combination += 2 ** bit

        return combination

    def _view(self, data, pixel=None, fields=None):
        """Print list of data shapes from a dataset.

        Arguments:
            data: dict of numpy arrays.
            fields: list of str, the fields to see
            pixel: tuple of ints, particular pixel

        Returns:
            None
        """

        # print spacer
        self._print('')

        # set fields
        fields = fields or list(data.keys())

        # for each item
        for name, array in data.items():

            # check for field membership
            if name in fields:

                # try to
                try:

                    # if pariruclar pixel
                    if pixel:

                        # print each one
                        self._print(name, array.shape, '{}: {}'.format(pixel, array[pixel[:len(array.shape)]]))

                    else:

                        # print each one
                        self._print(name, array.shape, '{} to {}'.format(array.min(), array.max()))

                # unless error
                except (ValueError, IndexError, TypeError):

                    # print alert
                    self._print('{}: pass'.format(name))
                    pass

        # print spacer
        self._print('')

        return None

    def apply(self, filter, features=None, discard=False):
        """Apply a filter to a list of records.

        Arguments:
            filter: function object
            features=None: list of dicts
            discard: boolean, discard the rest

        Returns:
            list
        """

        # default features to entire collection
        features = features or list(self)

        # apply filter
        survivors = [feature for feature in features if filter(feature)]

        # sort survivors first by product of dimensions, then by number of dimensions
        survivors.sort(key=lambda feature: feature.name)
        survivors.sort(key=lambda feature: numpy.prod(feature.shape), reverse=True)
        survivors.sort(key=lambda feature: len(feature.shape), reverse=True)

        # if it is desired to discard the features that don't meet the condition
        if discard:

            # repopulate
            self._populate(survivors, discard=discard)

        return survivors

    def assemble(self, paths, three=True):
        """Assemble data from all paths into a dataset.

        Arguments:
            paths: list of str, filepaths
            three: boolean, include three dimensions?

        Returns:
            dict
        """

        # get current file for later
        current = self.current

        # begin data
        data = {}

        # for each path
        for path in paths:

            # ingest the data
            self.ingest(path)

            # get a counter for the shapes
            counter = list(Counter([feature.shape for feature in self if len(feature.shape) == 2]).items())
            counter.sort(key=lambda pair: pair[1], reverse=True)
            shape = counter[0][0]

            # get all 2-D feature names
            names = [feature.name for feature in self if feature.shape == shape]

            # set all data names to empty list
            data.update({name: data.setdefault(name, []) for name in names})

            # append all new data
            [data[name].append(self.grab(name)) for name in names]

            # if adding 3-D data
            if three:

                # get all 3-D feature names
                threes = [feature for feature in self if len(feature.shape) == 3]
                names = [feature.name for feature in threes if feature.shape[:2] == shape]

                # for each name
                for name in names:

                    # grab array
                    array = self.grab(name)
                    size = array.shape[2]

                    # for each position
                    for position in range(size):

                        # construct new name
                        nameii = '{}_{}'.format(name, self._pad(position))

                        # set all 3-D names to empty list
                        data.update({nameii: data.setdefault(nameii, [])})

                        # append all new data
                        data[nameii].append(array[:, :, position])

            # add rows and scanlines
            data.update({name: data.setdefault(name, []) for name in ('row', 'scan')})

            # create row and scan matrices
            row = numpy.array([list(range(shape[1]))] * shape[0])
            scan = numpy.array([list(range(shape[0]))] * shape[1]).transpose(1, 0)
            data['row'].append(row)
            data['scan'].append(scan)

        # concatenate all arrays
        data = {name: numpy.vstack(arrays) for name, arrays in data.items()}

        # if current is not blank
        if current:

            # reingest current file
            self.ingest(current)

        return data

    def attribute(self, five=None, look=True):
        """Print global file attributes.

        Arguments:
            five: hdf five object
            look: boolean, print to screen?

        Returns:
            None
        """

        # get default five
        five = five or self._fetch(self.current)

        # for each attribute
        globals = {}

        # if netcdf is flals
        if not self.net:

            # get attributes
            for name, contents in five.attrs.items():

                # print
                globals[name] = contents
                # self._print('{}: {}'.format(name, contents))

        # if netcdf
        if self.net:

            # get attributes
            for name in five.ncattrs():

                # print
                globals[name] = five.getncattr(name)

        # try to
        try:

            # close file
            five.close()

        # otheerwise
        except AttributeError:

            # skip
            pass

        # if look option
        if look:

            # print to screen
            self._look(globals, 3)

        return globals

    def augment(self, path, features, destination):
        """Augment a file with a feature.

        Arguments:
            path: str, path of hdf5 file
            features: list of Feature instances
            destination: str, new file path

        Returns:
            None
        """

        # ingest path
        hydra = Hydra(path)
        hydra.ingest(0)

        # grab all existing features if addresses are different
        slashes = [feature.slash for feature in features]
        compilation = [feature for feature in hydra if feature.slash not in slashes]

        # add new features
        compilation += features

        # stash
        self.stash(compilation, destination)

        return None

    def cascade(self, formula, reference=None, scan=True):
        """Cascade the features from one file to another, using function objects.

        Arguments:
            formula: formula instance
            reference: dict of lists of features, or list of features
            scan: boolean, view each parameter?

        Returns:
            list of features
        """

        # set reference to all features
        reference = reference or self.reference

        # if the reference is not a dictionary
        if reference == list(reference):

            # assume list of features and convert
            reference = self._refer(reference)

        # begin features and sources
        features = []
        sources = {}

        # create new features
        for parameters, function, names, addresses, attributes in formula:

            # for each parameter
            for parameter in parameters:

                # if not already in sources
                if parameter not in sources.keys():

                    # check for scan
                    if scan:

                        # print
                        self._print(parameter)

                    # get feature and fill
                    feature = self.dig(parameter, reference)[0]
                    feature.fill()

                    # add to sources
                    sources[parameter] = feature

            # get tensors
            tensors = [sources[parameter].data for parameter in parameters]

            # get first parameter, used for default info
            first = sources[parameters[0]]

            # perform calculation
            calculation = function(*tensors)

            # if there is only one output name
            if len(names) == 1:

                # put tensor into list
                calculation = [calculation]

            # construct feature
            for tensor, name, address, attribution in zip(calculation, names, addresses, attributes):

                # get address from first parameter
                name = name or first.name
                address = address or first.slash.replace('/' + first.name, '')

                # get attributes from first input if not given
                attribution = attribution or first.attributes

                # create new feature
                feature = Feature(address.split('/') + [name], tensor, attributes=attribution)

                # append to features and sources
                features.append(feature)
                sources[name] = feature

        return features

    def chart(self, *queries, destination='', limit=5):
        """Print a chart of the contents with size information.

        Arguments:
            queries: unpacked list of str, queries to use
            destination: str, path for text file dump
            limit: int, number of leaves to display

        Returns:
            None
        """

        # set default queries
        queries = queries or ['']

        # begin blueprints
        blueprints = ['Breakdown of {}\n'.format(self[0].path)]

        # for each query
        for query in queries:

            # get the associated featues and their data
            features = self.dig(query)

            # retrieve all data and get the total size
            [feature.fill() for feature in features]
            total = sum([feature.data.size * feature.data.itemsize for feature in features])

            # create a blueprint
            blueprint = self._diagram(features, total, limit=limit)
            blueprints += blueprint

        # if destination given
        if destination:

            # print to destination
            self._jot(blueprints, destination)

        # otherwise
        else:

            # for each line
            for blue in blueprints:

                # print to screen
                self._print(blue)

        return None

    def compare(self, path, pathii, fraction=None):
        """Compare two hdf5 paths using hdiff.

        Arguments:
            path: str, first file path
            pathii: str, second file path
            fraction: float, minimum relative difference

        Returns:
            None
        """

        # construct call
        call = ['h5diff', path, pathii]

        # if fraction
        if fraction:

            # construct call
            call = ['h5diff', '-p {}'.format(fraction), path, pathii]

        # run with subprocess
        self._print(' '.join(call))
        subprocess.call(call)

        return None

    def contrast(self, path, pathii, percent=False):
        """Determine span of percent differences for all arrays in the collection.

        Arguments:
            path: str, pathname
            pathii: str, pathname
            percent: boolean, use percent difference instead?

        Returns:
            None
        """

        # silencr runtime warnings
        warnings.filterwarnings("ignore", category=RuntimeWarning)

        # set symbols depending on percent or difference
        symbols = {False: '', True: '%'}

        # get current path
        current = self.current

        # get first path arrays
        self.ingest(path)
        arrays = {feature.name: feature.distil() for feature in self}

        # get second path arrays
        self.ingest(pathii)
        arraysii = {feature.name: feature.distil() for feature in self}

        # reingest current path
        self.ingest(current)

        # get collection of intersecting names
        names = set(arrays.keys()) & set(arraysii.keys())
        names = list(names)
        names.sort()

        # print
        self._print('')
        self._print('contrasting {} with {}:'.format(self._file(path, 1), self._file(pathii, 1)))
        self._print('')

        # for each name
        for name in names:

            # get arrays
            array = arrays[name]
            arrayii = arraysii[name]

            # try to
            try:

                # make mask for finites
                mask = (numpy.isfinite(array)) & (numpy.isfinite(arrayii))

                # get number of differences
                differences = (array[mask] != arrayii[mask]).sum()

                # if there are differences
                if differences > 0:

                    # calculate percent difference
                    relation = self._relate(array[mask], arrayii[mask], percent)

                    # get mask for finite relations
                    maskii = numpy.isfinite(relation)

                    # set formats
                    formats = [name, differences, maskii.sum(), numpy.prod(array.shape)]

                    # if there are entrei
                    if maskii.sum() > 0:

                        # add formats
                        formats += [self._round(relation[maskii].min(), 4), symbols[percent]]
                        formats += [self._round(relation[maskii].max(), 4), symbols[percent]]

                    # otherwise
                    else:

                        # add formats
                        formats += ['NA', symbols[percent]]
                        formats += ['NA', symbols[percent]]

                    # print to screen
                    self._print('{}: ( {} ) differences, ({} / {}), {} {} to {} {}'.format(*formats))

            # unless an error
            except TypeError:

                # pass
                pass

        # reset warnings
        warnings.resetwarnings()

        return None

    def correlate(self, abscissa, ordinate, masking=True):
        """Calculate the correlation coefficient between two variables.

        Arguments:
            abscissa: numpy array
            ordinate: numpy array
            masking: boolean, filter out fills and nans?

        Returns:
            float, correlation coefficient
        """

        # if masking:
        if masking:

            # set fills
            fill = -999
            fillii = 1e20

            # get valid data mask
            mask = (abscissa > fill) & (abs(abscissa) < fillii) & numpy.isfinite(abscissa)
            maskii = (ordinate > fill) & (abs(ordinate) < fillii) & numpy.isfinite(ordinate)
            masque = mask & maskii

            # apply mask
            abscissa = abscissa[masque]
            ordinate = ordinate[masque]

        # compute means
        mean = abscissa.mean()
        meanii = ordinate.mean()
        bias = (abscissa - mean)
        biasii = (ordinate - meanii)

        # compute correlation
        numerator = (bias * biasii).sum()
        denominator = numpy.sqrt((bias ** 2).sum() * (biasii ** 2).sum())
        correlation = numerator / denominator

        return correlation

    def dig(self, search, reference=None):
        """Dig for features with specific members in their route.

        Arguments:
            search: slashed search string
            reference: dict of feature lists, or feature list
            partial: boolean, search for partial matches?

        Returns:
            list of dicts
        """

        # set reference to all features
        reference = reference or self.reference

        # if the reference is not a dictionary
        if reference == list(reference):

            # assume list of features and convert
            reference = self._refer(reference)

        # if the search term is in the reference
        treasure = []
        if search in reference.keys():

            # get features
            treasure += reference[search]

        # otherwise
        else:

            # check for all terms in the keys, excluding single keys
            #keys = [key for key in reference.keys() if '/' in key]
            keys = list(reference.keys())
            keys = [key for key in keys if all([field in key for field in search.split('/')])]
            for key in keys:

                # add references
                treasure += reference[key]

        # remove non unique entries
        treasure = {'{}/{}'.format(feature.path, feature.slash): feature for feature in treasure}
        treasure = list(treasure.values())

        # sort to put exact last term up top
        term = search.split('/')[-1]
        treasure.sort(key=lambda feature: feature.name == term, reverse=True)

        return treasure

    def extract(self, reference=None, addresses=False):
        """Extract all data arrays into a dictionary.

        Arguments:
            reference: reference array with which to match shapes
            addresses: boolean, use full addresses?

        Return:
            dict of data arrays
        """

        # begin data
        data = {}

        # if addresses selected
        if addresses:

            # make data from addresses
            data.update({feature.slash: feature.distil() for feature in self})

        # otherwise
        else:

            # use names
            data.update({feature.name: feature.distil() for feature in self})

        # if reference given
        if reference:

            # screen for shape
            shape = data[reference].shape
            data = {name: array for name, array in data.items() if array.shape == shape}

        return data

    def gist(self, pixel=None, address=False):
        """Summarize data in the file.

        Arguments:
            pixel: tuple of ints, particular pixel
            address: boolean, use full addres?

        Returns:
            None
        """

        # collect data with feature names
        data = {feature.name: feature.distil() for feature in self}

        # if full addresses
        if address:

            # collect data with feature names
            data = {feature.slash: feature.distil() for feature in self}

        # print to screen
        self._view(data, pixel=pixel)

        return None

    def glimpse(self, query, paths=None):
        """Glimpse a list of features matching the query for each file.

        Arguments:
            query: str
            paths: list of str, the particular paths

        Returns:
            None
        """

        # set default paths
        paths = paths or self.paths

        # for each path
        for index, path in enumerate(paths):

            # print the path
            self._print('{}) query {} in {}...\n'.format(index, query, path))

            # ingest the contents
            self.ingest(path)

            # print the query results
            results = self.dig(query)
            self._tell(results)

        return None

    def grab(self, search, index=0, reference=None):
        """Grab the first array based on the search.

        Arguments:
            search: slashed search string
            index: int, index in list
            reference: dict of feature lists, or feature list

        Returns:
            list of dicts
        """

        # default array
        array = None

        # dig up features
        features = self.dig(search, reference)

        # try to
        try:

            # get the first
            array = features[index].distil()

        # unless nothing found
        except IndexError:

            # print alert
            self._print('{} came up empty!'.format(search))

        return array

    def ingest(self, path=0, mode=None, discard=True, names=None, folder='tmp', scan=False, net=False):
        """Ingest the data from a particular path, populating the instance with features.

        Arguments:
            path: str or int, file path or index of self.paths
            mode: specific mode to allow
            discard: boolean, remove old features?
            names: selection of names for selective conversion
            folder: name of directory for hdf4 conversions
            scan: boolean, show field names as ingesting?

        Returns:
            None

        Populates:
            self
        """

        # assume path is an integer
        try:

            # in which case set path
            path = self.paths[path]

        # unless not an integer
        except TypeError:

            # look for keyword
            paths = [entry for entry in self.paths if path in entry]

            # try to
            try:

                # get first entry
                path = paths[0]

            # unless not in folder
            except IndexError:

                # in which case, leave path alonge
                pass

        # retrieve the collection
        if self._stage(path)['extension'] in ('.he4', '.he'):

            # make conversions directory
            self._make(folder)

            # create conversion path
            date = self._stage(path)['date'].split('t')[0][:7]
            product = self._stage(path)['product']
            conversion = '{}/{}_conversion.h5'.format(folder, path.split('/')[-1].split('.')[0])

            # convert select parameter names into the file
            self._convert(path, conversion, names)

            # replace path
            path = conversion

        # set net attribute
        self.net = net

        # if netcdf
        if self.net:

            # fetch the netcdf4 file
            with self._fetch(path) as five:

                # collect all features
                features = self._garner(five, path, mode=mode, scan=scan)
                self._populate(features, discard=discard)

        # otherwise
        else:

            # fetch the hdf5 file
            with self._fetch(path) as five:

                # collect all features
                features = self._gather(five, path, mode=mode, scan=scan)
                self._populate(features, discard=discard)

        # set current path
        self.current = path

        return None

    def isolate(self, path, pathii):
        """Isolate the different features amongst two paths.

        Arguments:
            path: str, filepath
            pathii: str, filepath

        Returns:
            None
        """

        # ingest both paths
        self.ingest(path)
        self.ingest(pathii, discard=False)

        # group by path
        groups = self._group(self, lambda feature: feature.path)

        # partition based on membership
        names = [feature.slash for feature in groups[path]]
        namesii = [feature.slash for feature in groups[pathii]]

        # get intersection
        intersection = [name for name in names if name in namesii] + [name for name in namesii if name in  names]
        intersection = list(set(intersection))
        self._print('\nintersection: {}'.format(len(intersection)))

        # get outliers
        outliers = [name for name in names if name not in namesii]
        outliersii = [name for name in namesii if name not in namesii]

        # print outliers
        self._print('\noutliers: ')
        self._tell(outliers)
        self._print('\noutliersii: ')
        self._tell(outliersii)

        # go through intersection
        one = {feature.slash: feature.distil() for feature in groups[path]}
        two = {feature.slash: feature.distil() for feature in groups[pathii]}
        def comparing(name): return numpy.all(one[name] == two[name])
        truths = [comparing(name) for name in intersection]

        # count
        self._print(str(Counter(truths)))

        # display Falses
        for name, truth in zip(intersection, truths):

            # if false
            if not truth:

                # print
                self._print(name)
                self._print(str(one[name].shape))
                self._print(str(two[name].shape))

        return None

    def meld(self, singlets, fusions, destination, field='orbit_number'):
        """Meld all singlet hdf files into one fused file, incorporating previous fusions.

        Arguments:
            singlets: list of str, filepaths
            fusions: list of str, filepaths
            destination: str, destination file path
            field: str, field for matching

        Returns:
            None
        """

        # define default aggregate values
        fusion = None
        modified = 0
        aggregates = []

        # if there is an appropriate fusion file
        if len(fusions) > 0:

            # get the date modified
            modified = os.stat(fusions[0]).st_mtime

            # open the file
            fusion = self._fetch(fusions[0])

            # grab the orbit numbers
            orbits = fusion['IndependentVariables'][field][:]
            orbits = [('00' + str(int(orbit)))[-6:] for orbit in orbits]
            aggregates += [(orbit, index) for index, orbit in enumerate(orbits)]

            # print status
            self._print('fusion file found with {} orbits.'.format(len(orbits)))

        # get the orbit numbers for each member and link to path
        orbits = [self._parse(path)[field] for path in singlets]
        orbits = [('00' + str(int(orbit)))[-6:] for orbit in orbits]
        raw = [(orbit, path, os.stat(path).st_mtime) for orbit, path in zip(orbits, singlets)]

        # only keep raw orbits if they have unique orbit numbers or a later date modified
        aggregated = [orbit for orbit, _ in aggregates]
        raw = [(orbit, path) for orbit, path, clock in raw if orbit not in aggregated or clock > modified]

        # remove any aggregates that overlap with refreshed raws
        refreshed = [orbit for orbit, _ in raw]
        aggregates = [(orbit, index) for orbit, index in aggregates if orbit not in refreshed]

        # print status
        self._print('{} new orbits found.'.format(len(raw)))

        # assuming there is more than one member
        if len(raw) > 0:

            # create hydra and ingest first file
            folder = '/'.join(raw[0][1].split('/')[:-1])
            hydra = Hydra(folder)
            hydra.ingest(hydra.paths[0])

            # get all features not in 'Data' links, and add link to categorical data
            features = hydra.apply(lambda feature: 'Data' not in feature.route)
            [feature.update({'link': True}) for feature in features if 'Categories' in feature.route]

            # combine and sort the streams
            stream = raw + aggregates
            stream.sort(key=lambda pair: pair[0])

            # for every feature
            for feature in features:

                # allocate array space for entire dataset
                shape = tuple([len(stream)] + list(feature.shape))
                feature.instil(numpy.empty(shape, dtype=feature.type))

            # timestamp
            self._print('melding...')

            # for each orbit address
            for index, (orbit, address) in enumerate(stream):

                # print update
                if int(index) % 100 == 0:

                    # print orbit number
                    self._print('orbit: {}, {} of {}...'.format(orbit, index, len(stream)))

                # assuming the address is a string
                if address == str(address):

                    # open up the path
                    five = self._fetch(address)

                    # for each feature
                    for feature in features:

                        # add to the data
                        feature.data[index] = feature.grab(five)[:]

                    # close hdf5
                    five.close()

                # otherwise, assume it is a fusion file index
                else:

                    # for each feature
                    for feature in features:

                        # add to the data
                        feature.data[index] = feature.grab(fusion)[address]

            # for each feature
            for feature in features:

                # if the dimensions are greater than 1
                if len(feature.shape) > 1:

                    # squeeze out trivial dimensions
                    feature.squeeze()

                # add extra dimension for small streams
                if len(stream) < 2:

                    # add extra dimension
                    feature.deepen()

            # if there was a fusion file already
            if fusion:

                # close it before overwriting
                fusion.close()

            # create fusion file
            self.stash(features, destination, 'Data')

        return None

    def merge(self, paths, destination, lead=False):
        """Merge together several congruent hdf5 files in order

        Arguments:
            paths: list of str
            destination: str
            lead: boolean, add new dimension?

        Returns:
            None
        """

        # split paths into primary and secondaries by default
        primary = paths[0]
        secondaries = paths[1:]

        # search for first non empty path
        for index, path in enumerate(paths):

            # ingest primary and collect all features
            self._print('ingesting primary feature set...')
            self.ingest(path)

            # check for nonzero length
            if len(self) > 0:

                # set primary to this first
                primary = paths[index]
                secondaries = paths[index + 1:]

                # break when found
                break

        # fill all features after weeding out Data duplicates
        self._print('filling features...')
        features = list(self)
        # features = self.apply(lambda feature: 'IndependentVariables' in feature.slash)
        # features += self.apply(lambda feature: 'Categories' in feature.slash)
        [feature.fill() for feature in features]

        # if wanting leading dimension
        if lead:

            # add leading dimension to primary features
            [feature.instil(numpy.array([feature.spill()])) for feature in features]

        # for each secondary file
        self._print('adding secondaries...')
        for path in secondaries:

            # print path
            self._print('{}...'.format(path))

            # ingest the path
            self.ingest(path)

            # as long as it is not empty
            if len(self) > 0:

                # for each feature
                for feature in features:

                    # try to
                    try:

                        # find the equivalent in the secondary
                        equivalent = self.dig(feature.slash)[0]

                        # try to
                        try:

                            # convert tensors to two dimensions if needed
                            array = feature.spill()
                            if len(array.shape) < 2:

                                # convert
                                array = numpy.array([array]).transpose(1, 0)

                            # convert equivalent to 2-D
                            arrayii = equivalent.distil()

                            # if requesting leading dimenetion
                            if lead:

                                # add leading dimension
                                arrayii = numpy.array([arrayii])

                            # if less than 2-d
                            if len(arrayii.shape) < 2:

                                # convert
                                arrayii = numpy.array([arrayii]).transpose(1, 0)

                            # concatenate the data, adding leading dimension
                            tensor = numpy.vstack([array, arrayii])
                            feature.instil(tensor)

                        # unless a mismatch occurs
                        except ValueError:

                            # zero out feature
                            self._print('mismatch found for: {}, zeroing...'.format(feature.name))
                            feature.instil(numpy.array([0]))

                    # unless not found
                    except IndexError:

                        # zero out feature
                        self._print('no entry for: {}, skipping...'.format(feature.name))

        # add link conditions
        #[feature.update({'link': True}) for feature in features if 'Categories' in feature.slash]
        #[feature.update({'link': False}) for feature in features if 'IndependentVariables' in feature.slash]
        [feature.update({'link': False}) for feature in features]

        # stash
        self._print('stashing {}...'.format(destination))
        self.stash(features, destination, mode='w')

        # print status
        self._print('stashed {}.'.format(destination))

        return None

    def mimic(self, paths, old, new, names, aliases=None, functions=None, addresses=None, attributes=None, tag='alt', transfer=True):
        """Recreate a file, with altered names and transformations for selected fields.

        Arguments:
            paths: list of str, file names
            old: str, filepath portion to replace
            new: str, replacement for old in destination name
            names: list of str, parameter names
            aliases: list of str, new names
            functions: list of formula objects
            addresses: list of str
            attributes: list of dicts
            tag: tag for designating new filename
            transfer: boolean, transfer other categories too?

        Returns:
            None
        """

        # create destinations
        destinations = [path.replace(old, new).replace('.h5', '_{}.h5'.format(tag)) for path in paths]

        # set default functions and addresses
        aliases = aliases or [None] * len(names)
        functions = functions or [None] * len(names)
        addresses = addresses or [None] * len(names)
        attributes = attributes or [{}] * len(names)

        # for each path and destination
        for path, destination in zip(paths, destinations):

            # ingest the path
            self.ingest(path)

            # get subset without data categories
            features = self.apply(lambda feature: 'Data' not in feature.slash.split('/')[0])

            # begin formula
            formula = Formula()

            # for every feature
            for feature in features:

                # for each name
                designated = False
                for name, alias, function, address, attribute in zip(names, aliases, functions, addresses, attributes):

                    # if there is a match
                    if name in feature.slash:

                        # replace the formulation with the alias
                        designated = True
                        feature.attributes.update(attribute)
                        formulation = (feature.slash, function, alias, address)
                        formula.formulate(*formulation)

                # otherwise
                if not designated and transfer:

                    # add default formulation
                    formulation = (feature.slash, None, feature.name, None)
                    formula.formulate(*formulation)

            # create the cascade
            cascade = self.cascade(formula, features)

            # link Categories to Data and stash
            [feature.update({'link': True}) for feature in cascade if 'Categories' in feature.slash]
            [feature.update({'link': False}) for feature in cascade if 'IndependentVariables' in feature.slash]
            self.stash(cascade, destination, 'Data')

        return None

    def nether(self, path):
        """Open a file with netcdf4.

        Arguments:
            path: str, filepath

        Returns:
            netCDF4 Dataset
        """

        # open with netCDF4
        net = netCDF4.Dataset(path)

        return net

    def plant(self, data, target, destination, masking=True, header=None, classify=False):
        """Use random forest to predict target value from data

        Arguments:
            data: dict of str, numpy array
            target: str, name of target variable
            destination: str, pathname for destination
            masking: boolean, apply nan and fill mask first?
            header: header for report
            classify: boolean, use classifier?

        Returns:
            None
        """

        # begin report
        report = header or ['Random forest for {}'.format(target)]
        report.append('')

        # if masking:
        if masking:

            # set fills
            fill = -999
            fillii = 1e20

            # get valid data mask
            masks = [(array > fill) & (abs(array) < fillii) & numpy.isfinite(array) for array in data.values()]

            # multiply all masks
            masque = masks[0]
            for mask in masks[1:]:
                # use logical and
                masque = numpy.logical_and(masque, mask)

            # check masked quantity
            masked = masque.sum()
            total = numpy.prod(masque.shape)
            report.append(self._print('masque: {} of {}, {} %'.format(masked, total, 100 * masked / total)))

            # apply masque
            data = {name: array[masque] for name, array in data.items()}

        # assemble matrix
        names = [name for name in data.keys() if name != target] + [target]
        train = numpy.array([data[name].flatten() for name in names]).transpose(1, 0)

        # split off truth
        matrix = train[:, :-1]
        truth = train[:, -1]

        # if classifier
        if classify:

            # run random forest
            self._stamp('running random forest for {}...'.format(target), initial=True)
            self._print('matrix: {}'.format(matrix.shape))
            forest = RandomForestClassifier(n_estimators=100, max_depth=5)
            forest.fit(matrix, truth)

        # otherwise
        else:

            # run random forest
            self._stamp('running random forest for {}...'.format(target), initial=True)
            self._print('matrix: {}'.format(matrix.shape))
            forest = RandomForestRegressor(n_estimators=100, max_depth=5)
            forest.fit(matrix, truth)

        # predict from training
        prediction = forest.predict(matrix)

        # calculate score
        score = forest.score(matrix, truth)
        report.append(self._print('score: {}'.format(score)))

        # get importances
        importances = forest.feature_importances_
        pairs = [(field, importance) for field, importance in zip(names, importances)]
        pairs.sort(key=lambda pair: pair[1], reverse=True)
        for field, importance in pairs:

            # add to report
            report.append(self._print('importance for {}: {}'.format(field, importance)))

        # make report
        self._jot(report, destination)

        # print status
        self._stamp('planted.')

        return None

    def renew(self):
        """Renew the file list.

        Arguments:
            None

        Returns:
            None
        """

        # register
        self._register()

        return None

    def scrounge(self, path=None):
        """Dig up the nested metadata fields and attributes.

        Arguments:
            path: str, filepath

        Returns:
            None
        """

        # set path to current by default
        path = path or self.current

        # create netcdf4 interface
        with self.nether(path) as net:

            # scrounge for all metadata
            self._seek(net)

        return None

    def sieve(self, name, fields, conditions):
        """Pull the data from a parameter based on matching conditions of other fields.

        Arguments:
            name: str, parameter name
            fields: str, auxiliary fields
            conditions: float/int, values of auxiliary fields

        Returns:
            numpy array
        """

        # create masks for each condition
        masks = []
        for field, condition in zip(fields, conditions):

            # make a boolean mask and add to masks
            array = self.dig(field)[0].distil()
            mask = array == condition
            masks.append(mask)

        # extract data
        data = self.dig(name)[0].distil()

        # apply all masks
        screen = numpy.prod(masks)
        selection = data[screen]

        return selection

    def sift(self, path, folder, *searches):
        """Sift through omp fusion files to make a subset.

        Arguments:
            path: str, filepath
            folder: str, folder name
            *searches: unpacked list of str

        Returns:
            None
        """

        # ingest path
        self.ingest(path)

        # get independent variables
        independents = self.dig('IndependentVariables')

        # get categories
        categories = self.dig('Categories')
        [feature.update({'link': True}) for feature in categories]

        # get subset based on names
        subset = []
        for search in searches:

            # add each
            subset += self.dig(search, categories)

        # fill all data
        [feature.fill() for feature in independents + subset]

        # construct destination path
        tags = '_'.join([search.replace('/', '_') for search in searches])
        name = path.split('/')[-1].split('.')[0]
        destination = '{}/{}_{}.h5'.format(folder, name, tags)

        # stash subset
        self.stash(independents + subset, destination, 'Data')

        return None

    def spawn(self, destination, data, attributes=None, net=False, globals=None):
        """Create an hdf file at destination from a dictionary of arrays.

        Arguments:
            destination: str, pathname of destination file
            data: dict of str: numpy address / array pairs
            attributes: dict of attribute dictionaries
            dimensions: dict of dimension scales
            net: boolean, use netcdf4?
            globals: dict of global attributes

        Returns:
            None
        """

        # set attributes and dimensions reservoir
        attributes = attributes or {}

        self._print('makig features..')

        # for each member
        features = []
        for address, array in data.items():

            # search for attributes
            attribute = attributes.get(address, {})

            # create feature
            # feature = Feature(address.split('/'), numpy.array(array), attributes=attribute)
            feature = Feature(address.split('/'), array, attributes=attribute)
            features.append(feature)

        # if netcdf4
        if net:

            # stash file using netCDF4
            self.store(features, destination, globals=globals)

        # otherwise
        else:

            # stash file at destination using h5py
            self.stash(features, destination)

        return None

    def staple(self, paths, destination):
        """Combine all non-overlapping parameters together.

        Arguments:
            paths: list of str, filepaths
            destination: str, destination filepath

        Returns:
            None
        """

        # go through each path, discarding old contents the first time
        discards = [True] + [False for _ in paths[1:]]
        for path, discard in zip(paths, discards):

            # and ingest the features, only depopulating before the first time
            self.ingest(path, discard=discard)

        # get only categories and independents
        features = self.apply(lambda feature: 'Categories' in feature.slash or 'IndependentVariables' in feature.slash)

        # fill all data in features
        [feature.fill() for feature in features]

        # flip link switch for category fields
        [feature.update({'link': True}) for feature in features if 'Categories' in feature.slash]

        # stash at new destination
        self.stash(features, destination, link='Data')

        return None

    def stash(self, features, destination, link=None, mode='w', compression=None, scan=False):
        """Stash a group of features in an hdf5 file.

        Arguments:
            features: list of dicts
            destination: str, destination filepath
            link: str, name of link folder
            mode: str, writemode
            compression: compression option
            scan: scan feature names?

        Returns:
            None
        """

        # fill all features if not yet filled
        [feature.fill() for feature in features]

        # remove automatic linking
        # [feature.update({'link': False}) for feature in features if 'IndependentVariables' in feature.slash]
        # [feature.update({'link': False}) for feature in features if 'Categories' in feature.slash]

        # begin file
        five = h5py.File(destination, mode, track_order=True)

        # get list of all groups, maintaining feature order
        addresses = [feature.slash.split('/')[:-1] for feature in features]
        groups = [['/'.join(address[:index + 1]) for index, _ in enumerate(address)] for address in addresses]
        groups = [address for group in groups for address in group]

        # eliminate duplicates and empty name
        groups = list({group: True for group in groups if group}.keys())

        # sort groups and features by name
        groups.sort()
        features.sort(key=lambda feature: feature.slash)

        # create groups
        for address in groups:

            # print
            self._print(address)

            # try to
            try:

                # add the group, tracking order
                group = five.create_group(address, track_order=True)

            # unless python 2, without this option
            except TypeError:

                # try to
                try:

                    # ignore track order option
                    group = five.create_group(address)

                # unless already exists
                except ValueError:

                    # pass and alert
                    self._print('{} group already exits'.format(link))
                    pass

            # unless already exists
            except ValueError:

                # pass and alert
                self._print('{} group already exits'.format(link))
                pass

        # if a link is given
        links = None
        if link:

            # try to
            try:

                # create links folder, with track_order option
                links = five.create_group(link)

            # unless already exists
            except ValueError:

                # pass and alert
                self._print('{} group already exits'.format(link))
                links = five[link]

        # go through each feature
        for feature in features:

            # if scanning
            if scan:

                # print feature
                self._print(feature.name, feature.slash, feature.data.dtype)

            # if string
            if '<U' in str(feature.data.dtype):

                # reformat
                feature.data = feature.data.astype('S')

            # try to
            try:

                # add dataset
                tensor = five.create_dataset(feature.slash, data=feature.data, compression=compression)

                # for each attribute
                for attribute, information in feature.attributes.items():

                    # add to tensor
                    tensor.attrs[attribute] = information

                # # if feature in dimensions
                # if feature.slash in dimensions.keys():
                #
                #     # also add dummy variable to activate dimension
                #     slash = 'dimensions/{}'.format(feature.slash)
                #     data = numpy.array(list(range(dimensions[feature.slash])))
                #     dummy = five.create_dataset(slash, data=data, compression=compression)
                #     dummy.make_scale(self._file(slash))
                #
                #     # attach to tensor
                #     tensor.dims[0].attach_scale(dummy)

                # create link
                if feature.link:

                    # add tensor as link
                    links[feature.name] = tensor

            # unless already exists
            except (ValueError, RuntimeError, OSError, TypeError) as error:

                # alert if desired
                if scan:

                    # print error
                    self._print('skipping {}: {}'.format(feature.slash, error))

        # close hdf5 file
        five.close()

        # print messate
        self._print('{} stashed.'.format(destination))

        return None

    def store(self, features, destination, globals=None, link=None, mode='w', compression=None, scan=False):
        """Stash a group of features in a netCDF file.

        Arguments:
            features: list of dicts
            destination: str, destination filepath
            globals: dict of global attributes
            link: str, name of link folder
            mode: str, writemode
            compression: compression option
            scan: scan feature names?
            dimensions: dict of dimension scales

        Returns:
            None
        """

        # fill all features if not yet filled
        [feature.fill() for feature in features]

        # set globals
        globals = globals or {}

        # begin netcdf4 file
        with netCDF4.Dataset(destination, mode='w', format='NETCDF4') as net:

            # for each global
            for name, contents in globals.items():

                # add to attributes
                net.setncattr(name, contents)

            # get all groups
            groups = [self._fold(feature.slash) for feature in features]
            groups = self._skim(groups)
            groups.sort()

            # separate varialbes and dimensions
            dimensions = [feature for feature in features if feature.attributes.get('dimension', False)]
            variables = [feature for feature in features if not feature.attributes.get('dimension', False)]

            # for each group
            for name in groups:

                # if name not blank
                if name:

                    # create group
                    group = net.createGroup(name)

                    # for each global
                    for name, contents in globals.items():

                        # also add to group attributes for now
                        group.setncattr(name, contents)

            # for each dimension:
            for feature in dimensions:

                # print status
                self._print('dimension: ', feature.name, feature.attributes['size'])

                # unpack label
                group = self._fold(feature.slash)
                name = self._file(feature.slash)
                size = feature.attributes['size']

                # if group is legit
                if group:

                    # create dimension
                    _ = net[group].createDimension(name, size)

                # otherwise
                else:

                    # create as root dimension
                    _ = net.createDimension(name, size)

            # for each variable
            for feature in variables:

                # print status
                self._print(feature.name, feature.shape, feature.data.shape, feature.data.dtype)

                # get dimension scales
                scales = feature.attributes.get('dimensions', ())

                # construct variable with compression features
                variable = net.createVariable(feature.slash, feature.data.dtype, scales, zlib=True, complevel=9)

                # add the data
                variable[:] = feature.data

        return None

    def survey(self, search=None):
        """Survey all features.

        Arguments:
            search: str, search string

        Returns:
            None
        """

        # default features to self
        features = list(self)
        if search:

            # dig up the features
            features = self.dig(search)

        # if there are features
        if len(features) > 0:

            # get all feature slashes
            slashes = []
            def describing(feature): return str(feature.attributes.get('long_name', '_'))
            def unifying(feature): return str(feature.attributes.get('units', '_'))
            for feature in features:

                # make info
                info = (feature.slash, feature.name, feature.shape, describing(feature), unifying(feature))
                slashes.append(info)

            # get the largest length of all shape strings
            length = max([len(str(shape)) for _, __, shape, ___, ____ in slashes])

            # group by all but last
            groups = self._group(slashes, lambda slash: '/'.join(slash[0].split('/')[:-1]))

            # for each group
            for group, members in groups.items():

                # sort members
                members.sort()

                # for eadh member
                for slash, name, shape, description, units in members:

                    # tab shape string based on maximum
                    tab = '{} {}'.format(shape, ' ' * (length - len(str(shape))))

                    # format and print based on lotal character limit
                    limit = 150
                    line = '{}  :  {}  :  {} ( {} )  :  {}'.format(tab, name, description, units, slash)[:limit]
                    self._print(line)

                # print spacer
                self._print('')

        # otherwise
        else:

            # print empty message
            self._print('empty!\n')

        return None

    def synthesize(self, path, pathii, old, new, names, namesii, function, tag='synth'):
        """Create new file by sythesizing fields in the first and fields in the second path.

        Arguments:
            path: str, filepath of numerator
            pathii: str, filepath of secondary
            old: str, old path fragment
            new: str, new path fragment
            names: list of str, the numerator fields
            namesii: list of str, the secondary fields
            function: function object
            tag: str

        Returns:
            None
        """

        # ingest the secondary file
        self.ingest(pathii)

        # grab all secondary tensors
        secondaries = [self.dig(name)[0].distil() for name in namesii]

        # ingest the numerator file
        self.ingest(path)

        # grab independent varibales and categories
        variables = self.dig('IndependentVariables')
        categories = self.dig('Categories')

        # for each name
        cascade = []
        for index, name in enumerate(names):

            # begin formula
            formula = Formula()

            # formulate with dividing function
            secondary = secondaries[index]
            def functioning(tensor): return function(tensor, secondary)
            formula.formulate(name, functioning, name + '_{}'.format(tag))

            # create cascade
            cascade += self.cascade(formula, categories, scan=True)

        # add link and stash
        destination = path.replace(old, new).replace('.h5', '_{}.h5'.format(tag))
        self.stash(variables + cascade, destination, 'Data', scan=True)

        return None

    def whittle(self, paths, field='orbit_number'):
        """Whittle the parameters in two hdf5 files down so that only common entries are left.

        Arguments:
            paths: list of two str, hdf5 filepaths
            field: str, name of field with which to match.

        Returns:
            None
        """

        # for each path
        orbits = []
        for path in paths:

            # ingest and grab set of orbits based on given field
            self.ingest(path)
            orbits.append(self.dig(field)[0].distil().squeeze())

        # create boolean masks based on overlapping orbits
        masks = [numpy.isin(orbits[0], orbits[1]), numpy.isin(orbits[1], orbits[0])]

        # go through each path with its mask
        for path, mask, orbit in zip(paths, masks, orbits):

            # reingest the path
            self.ingest(path)

            # create destination path
            destination = path.replace('.h5', '_whittled.h5')

            # grab categories and independentvariables
            features = self.dig('Categories') + self.dig('IndependentVariables')

            # begin formula
            formula = Formula()

            # for each feature
            for feature in features:

                # default function to None
                function = None

                # if the feature's shape matches the orbits
                if feature.shape[0] == len(orbit):

                    # create the masking function
                    def masking(tensor): return tensor[mask]
                    function = masking

                # transfer the feature, subsetting as appropriate
                formula.formulate(feature.slash, function, feature.name)

            # create the cascade
            cascade = self.cascade(formula, features)

            # rewrite the file
            [feature.update({'link': True}) for feature in cascade if 'Categories' in feature.slash]
            self.stash(cascade, destination, 'Data', scan=True)

        return None