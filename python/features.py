# features.py to represent dataset features

# import numpy
import numpy

# import pprint for pretty printing
import pprint

# import h5py to read h5 files
import h5py


# class Feature to represent data features
class Feature(object):
    """Class feature to store feature attributes.

    Inherits from:
        None
    """

    def __init__(self, route=None, data=None, shape=None, path='', format=None, attributes=None, link=False, tags=None):
        """Initialize a feature instance.

        Arguments:
            route: list of str
            shape: tuple of ints
            format: datatype
            path: str, file path of parent file
            attributes: dict
            index: int
        """

        # set attributes
        self.route = route or ['_']
        self.name = route[-1]

        # set slashed address
        self.slash = '/'.join(self.route)

        # set default attributes
        self.shape = shape or ()
        self.type = format
        self.path = path
        self.data = data
        self.link = link

        # update data attributes
        if self.data is not None:

            # update attributes
            self.shape = data.shape
            self.type = data.dtype

        # set attributes and decode bytestrings
        self.attributes = attributes or {}
        self._decode()

        # default internal tags attribute to empty dict
        self.tags = tags or {}

        return

    def __repr__(self):
        """Represent the feature on screen.

        Arguments:
            None

        Returns:
            str
        """

        # make a string of the feature's route
        name = self.name
        shape = self.shape
        path = self.path.split('/')[-1][:20]
        slash = self.slash.replace(self.name, '')[-100:]
        representation = '< Feature: {} {} {}../{} >'.format(shape, name, path, slash)

        return representation

    def _decode(self):
        """Decode byte strings in attributes.

        Arguments:
            None

        Returns:
            None
        """

        # for each attribute
        for name, contents in self.attributes.items():

            # try to
            try:

                # decode byte string into utf
                self.attributes[name] = contents.decode('utf8')

            # otherwise
            except AttributeError:

                # nevermind
                pass

        return None

    def _fetch(self, path):
        """Link to the contents of an hdf5 file.

        Arguments:
            path: str, file path

        Returns:
            hdf5 file
        """

        # open up the hdf5 file
        five = h5py.File(path, 'r')

        return five

    def cast(self, symbol='f8'):
        """Cast the data array into a new type.

        Arguments:
            symbol: str, symbol for type

        Returns:
            None

        Populates:
            self.data
        """

        # adjust type
        self.data = self.data.astype(symbol)

        return None

    def copy(self):
        """Copy the feature.

        Argumenets:
            None

        Returns:
            feature instance
        """

        # initialize new feature
        parameters = {'route': [entry for entry in self.route], 'shape': self.shape}
        parameters.update({'format': self.type, 'path': self.path, 'attributes': self.attributes.copy()})
        parameters.update({'link': self.link, 'tags': self.tags.copy()})
        xerox = Feature(**parameters)

        # check data
        if self.data is not None:

            # copy data
            xerox.data = self.data

        return xerox

    def deepen(self):
        """Add a dimension to the data.

        Arguments:
            None

        Returns:
            None

        Populates:
            self.data
            self.shape
        """

        # add dimension to data
        self.data = numpy.array([self.data])

        # adjust shape
        self.shape = self.data.shape
        self.type = self.data.dtype

        return None

    def detour(self, trigger, replacement):
        """Replace a step in a feature's route.

        Arguments:
            trigger: str, the keyword of the route step
            replacement: str, the replacement

        Returns:
            None

        Populates:
            self.route
        """

        # find the route indices
        indices = [index for index, step in enumerate(self.route) if trigger in step]

        # at each index
        for index in indices:

            # replace the step
            self.route[index] = replacement

        # regenerate slash and branch
        self.slash = '/'.join(self.route)

        return None

    def dictate(self):
        """Copy attributes into a dict.

        Arguments:
            None

        Returns:
            dict
        """

        # get dictionary representation
        dictionary = self.__dict__

        # alter array to min and max, shape
        array = dictionary['data']

        # if array is not None
        if array is not None:

            # rewrite with min, max
            dictionary['data'] = '{}, {} to {}'.format(array.shape, array.min(), array.max())

        return dictionary

    def distil(self, function=None, subset=None, refill=False):
        """Fill the data attribute and return a copy.

        Arguments:
            function: function to apply
            subset: list of ints
            refill: boolean, refill?

        Returns:
            None
        """

        # fill
        self.fill(function=function, subset=subset, refill=refill)

        # spill
        tensor = self.spill()

        return tensor

    def divert(self, step, index=0):
        """Divert a group to a different branch by changing the route.

        Arguments:
            step: str, name of step
            index: int, point of insertion.

        Returns:
            None

        Populates:
            self.route
            self.slash
        """

        # make diversion from old route
        route = self.route
        diversion = route[:index] + [step] + route[index:]

        # set new route and slash
        self.route = diversion
        self.name = diversion[-1]
        self.slash = '/'.join(diversion)

        return None

    def dub(self, name, step=False):
        """Rename a feature.

        Arguments:
            name: str, new name
            step: boolean, create new step in route?

        Returns:
            None

        Populates:
            self.name
            self.route
        """

        # change name and route
        self.name = name

        # if step
        if step:

            # add name to route
            self.route += [name]

        # otherwise
        else:

            # alter last entry
            self.route[-1] = name

        # recreate slash
        self.slash = '/'.join(self.route)

        return None

    def fill(self, function=None, subset=None, refill=False):
        """Fill the data attribute by drawing the data out of the hdf5 file.

        Arguments:
            five: opened hdf5 file
            indices: list of ints
            refill: boolean, refill if already filled?

        Returns:
            None
        """

        # check for None
        if refill or self.data is None:

            # open the file
            five = self._fetch(self.path)

            # get the tensor of data and apply function
            tensor = self.grab(five, subset)

            # if given a function
            if function:

                # apply it
                tensor = function(tensor)

            # set data
            self.data = tensor

            # close file
            five.close()

        # reset the shape
        self.shape = self.data.shape
        self.type = self.data.dtype

        return None

    def grab(self, five, subset=None):
        """Grab the data from the hdf5 file by following its route.

        Arguments:
            five: opened hdf5 file
            subset: list of ints, subset indices

        Returns:
            None
        """

        # try to
        try:

            # get branch from full route
            branch = five[self.slash]

        # unless not found
        except KeyError:

            # in which case, do it in two steps
            route = self.slash.split('/')
            slash = '/'.join(route[:-1])
            stub = route[-1]
            branch = five[slash][stub]

        # get tensor
        tensor = self.select(branch, subset)

        return tensor

    def instil(self, tensor):
        """Instil the data from an arbitrary matrix.

        Arguments:
            tensor: numpy array

        Returns:
            None

        Populates:
            self.data
            self.shape
        """

        # set data to tensor
        self.data = tensor

        # update shape
        self.shape = self.data.shape
        self.type = self.data.dtype

        return None

    def select(self, data, subset):
        """Select specific indices from an hdf5 dataset.

        Arguments:
            data: hdf5 dataset
            subset: list of ints

        Returns:
            numpy array
        """

        # assume no subset
        if not subset:

            # try to
            try:

                # construct array as copy
                array = data[:]

            # otherwise
            except ValueError:

                # get the scalar data
                array = data[()]

        # otherwise
        else:

            # get length of subset
            length = len(subset)

            # get None axis as placeholder for all (:)
            axis = subset.index(None)

            # construct two dimensional subset functions
            subsetting = {2: {0: lambda tensor: tensor[:, subset[1]], 1: lambda tensor: tensor[subset[0], :]}}

            # add three dimensional subset functions
            subsetting.update({3: {0: lambda tensor: tensor[:, subset[1], subset[2]]}})
            subsetting[3].update({1: lambda tensor: tensor[subset[0], :, subset[2]]})
            subsetting[3].update({2: lambda tensor: tensor[subset[0], subset[1], :]})

            # perform subset
            array = subsetting[length][axis](data)

        # retrieve scale factors and offsets
        factor = self.attributes.get('scale_factor', 1)
        offset = self.attributes.get('add_offset', 0)

        # try to
        try:

            # apply factors
            array = (array * factor) + offset

        # unless a string
        except TypeError:

            # print
            pass

        return array

    def spill(self):
        """Get a copy of the data from the feature.

        Arguments:
            None

        Returns:
            numpy array
        """

        # make a copy of the data
        tensor = self.data.copy()

        return tensor

    def squeeze(self):
        """Squeeze out excess dimensions in the data.

        Arguments:
            None

        Returns:
            None
        """

        # squeeze the data
        self.data = self.data.squeeze()

        # but if there are no dimensions left
        if len(self.data.shape) < 1:

            # add one
            self.data = numpy.array([self.data])

        # adjust shape
        self.shape = self.data.shape
        self.type = self.data.dtype

        return None

    def update(self, news):
        """Update the feature with new values.

        Arguments:
            news: dict

        Returns:
            None
        """

        # got through news
        for property, information in news.items():

            # update
            setattr(self, property, information)

        # regenerate slash and branch
        self.slash = '/'.join(self.route)

        # if there is data
        if self.data is not None:

            # regenerate
            self.shape = self.data.shape
            self.type = self.data.dtype

        return None

    def view(self):
        """View the feature.

        Arguments:
            None

        Returns:
            Nonre
        """

        # get dictionary version
        dictionary = self.dictate()

        # pprint it
        pprint.pprint(dictionary)

        return None
