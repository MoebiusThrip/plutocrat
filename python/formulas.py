# formulas.py to enable calculations on features

# import pretty print
import pprint


# class Formula
class Formula(list):
    """class Formula to keep track of feature manipulations.

    Inherits from:
        list
    """

    def __init__(self):
        """Initialize a formula instance.

        Arguments:
            None
        """

        return

    def formulate(self, parameters, function=None, names=None, addresses=None, attributes=None):
        """Formulate a new calculation.

        Arguments:
            parameters: list of str, incoming parameter designations.
            function: function object
            names: str or list of str, new name of outputs
            addresses: str or list of str, new addresses of outputs
            attributes: dict or list of dict, new attributes

        Returns:
            None

        Populates:
            self
        """

        # create default function object
        def identifying(tensor): return tensor
        function = function or identifying

        # place parameters in a list if not already
        if parameters == str(parameters):

            # place in list
            parameters = [parameters]

        # default names to list of names
        names = names or [None]
        if names == str(names):

            # place in list
            names = [names]

        # default addresses to list of Nones
        addresses = addresses or [None] * len(names)
        if addresses == str(addresses):

            # place in list of same length as names
            addresses = [addresses] * len(names)

        # default attributes to list of Nones
        attributes = attributes or [None] * len(names)

        # try to
        try:

            # trivially convert to a dict
            attributes = dict(attributes)

            # and insert into list
            attributes = [attributes]

        # otherwise
        except (ValueError, TypeError):

            # assume already a list
            pass

        # create ticket
        ticket = (parameters, function, names, addresses, attributes)

        # populate
        self.append(ticket)

        return None

    def view(self):
        """View all entries in the formula.

        Arguments:
            None

        Returns:
            None
        """

        # print spacer
        print('\n')

        # for each member
        for parameters, function, names, addresses, attributes in list(self):

            # construct record
            record = {'parameters': parameters, 'function': function, 'names': names}
            record.update({'addresses': addresses, 'attributes': attributes})

            # pretty print entry
            pprint.pprint(record)
            print('\n')

        return None