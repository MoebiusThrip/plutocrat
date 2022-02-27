# plutocrats.py to manage finances

# import modules
from cores import Core
from plutinos import Plutino

# import numpy
import numpy

# import sklearn
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
from sklearn.svm import OneClassSVM, SVC
from sklearn.decomposition import PCA

# import matplotlib for plots
from matplotlib import pyplot
from matplotlib import style as Style
from matplotlib import rcParams
Style.use('fast')
rcParams['axes.formatter.useoffset'] = False


# class Plutocrat to manage finances
class Plutocrat(Core):
    """class Plutocrat to manage finances.

    Inherits from:
        Core
    """

    def __init__(self):
        """Initialize the plutocrat instance.

        Arguments:
            None
        """

        return

    def __repr__(self):
        """Create on screen represeentation.

        Arguments:
            None

        Returns:
            str
        """

        # craete representation
        representation = ' < Plutocrat >'

        return representation

    def _mine(self, row, plutons):
        """Mine the statement entry and add to the records.

        Arguments:
            row: list of str
            plutons: dict of plutons

        Returns:
            None

        Populates:
            self
        """

        # set account to citi
        account = 'citi'

        # unpack row
        status, date, description, debit, credit = row

        # construct date from stamp
        month, day, year = date.split('/')
        date = '{}-{}-{}'.format(year, month, day)

        # construct quantity
        quantity = -(float(credit or 0.0) + float(debit or 0.0))

        # convert text to lowercase
        text = description.lower().strip()

        # find the longest label in keys so far
        labels = [label for label in plutons.keys() if label in text] + ['']
        labels.sort(key=lambda label: len(label), reverse=True)
        label = labels[0]

        # if a label is found
        if label:

            # also get the tag
            tag = plutons[label]

        # otherwise
        else:

            # present description
            self._print(text)

            # collect label and tag
            label = input('>>> label? ')
            tag = input('>>> tag? ')

            # add to plutons
            plutons[label] = tag

        # balance not given, so set to 0.0
        balance = 0.0

        # create plutino
        plutino = Plutino(date, quantity, tag, label, account, text, balance)
        self.append(plutino)

        return None

    def _supply(self, row, plutons):
        """Mine the statement entry and add to the records.

        Arguments:
            row: list of str
            plutons: dict of plutons

        Returns:
            None

        Populates:
            self
        """

        # set account to citi
        account = 'cafcu'

        # unpack row
        date, description, comments, check, amount, balance = row

        # construct date from stamp
        month, day, year = date.split('/')
        date = '{}-{}-{}'.format(year, month, day)

        # construct quantity
        quantity = float(amount.replace('$', '').replace(',', ''))

        # convert text to lowercase
        text = description.lower().replace('\n', ' ').strip()

        # find the longest label in keys so far
        labels = [label for label in plutons.keys() if label in text] + ['']
        labels.sort(key=lambda label: len(label), reverse=True)
        label = labels[0]

        # if a label is found
        if label:

            # also get the tag
            tag = plutons[label]

        # otherwise
        else:

            # present description
            self._print(text)

            # collect label and tag
            label = input('>>> label? ')
            tag = input('>>> tag? ')

            # add to plutons
            plutons[label] = tag

        # balance not given, so set to 0.0
        balance = float(balance.replace('$', '').replace(',', ''))

        # create plutino
        plutino = Plutino(date, quantity, tag, label, account, text, balance)
        self.append(plutino)

        return None

    def accrete(self):
        """Gather up records, adding classification.

        Arguments:
            None

        Returns:
            None
        """

        # load up plutons
        plutons = self._load('../output/plutons.json')

        # get credit card records
        paths = self._see('../statements')
        for path in paths:

            # load up the csv
            rows = self._tape(path)

            # go through each row, skipping headers
            for row in rows[1:]:

                # mine the row
                self._mine(row, plutons)

            # resave plutons
            self._dump(plutons, '../output/plutons.json')

        # get bank records
        paths = self._see('../banks')
        for path in paths:

            # load up the csv
            rows = self._tape(path)

            # go through each row, skipping headers
            for row in rows[1:]:

                # mine the row
                self._supply(row, plutons)

            # resave plutons
            self._dump(plutons, '../output/plutons.json')

        # sort by date
        self.sort(key=lambda plutino: plutino.date)

        return None

    def key(self):
        """Add an entry to 401(k) records.

        Arguments:
            None

        Returns:
            None
        """

        # load up keys
        path = '../kays/kays.json'
        kays = self._load(path)
        kays = kays or {'kays': []}

        # input details
        date = input('>>> date? ')
        account = input('>>> acount number? ')
        company = input('>>> company? ')
        job = input('>>> job? ')
        balance = input('>>> balance? ')

        # create record
        record = {'date': date, 'account': account, 'job': job, 'company': company, 'balance': balance}

        # append
        kays['kays'].append(record)

        # dump new kays
        self._dump(kays, path)

        return None


