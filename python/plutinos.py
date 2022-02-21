# plutinos.py to represent elemental plutoccrat transactions

# import pprint for pretty printing
import pprint


# class Plutino to represent data plutinos
class Plutino(object):
    """Class Plutino to represent elemental transaction.

    Inherits from:
        None
    """

    def __init__(self, date, quantity, label, tag, account, text):
        """Initialize a plutino instance.

        Arguments:
            date: str
            quantity: float
            label: str, particular item identity
            tag: str, item category
            account: str, account
            text: str, item details
        """

        # set attributes
        self.date = date
        self.quantity = quantity
        self.label = label

        # set item attributes
        self.tag = tag
        self.account = account
        self.text = text

        return

    def __repr__(self):
        """Represent the plutino on screen.

        Arguments:
            None

        Returns:
            str
        """

        # create representation from attributes
        formats = (self.date, self.quantity, self.label, self.tag, self.account, self.text)
        representation = '< Plutino: {} {} {} {} {} {} >'.format(*formats)

        return representation

    def copy(self):
        """Copy the plutino.

        Argumenets:
            None

        Returns:
            plutino instance
        """

        # initialize new plutino from same attributes
        attributes = (self.date, self.quantity, self.label, self.tag, self.account, self.text)
        xerox = Plutino(*attributes)

        return xerox

    def dictate(self):
        """Copy attributes into a dict.

        Arguments:
            None

        Returns:
            dict
        """

        # get dictionary representation
        dictionary = self.__dict__

        return dictionary

    def view(self):
        """View the plutino.

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
