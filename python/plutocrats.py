# plutocrats.py to manage finances

# import modules
from cores import Core
from plutinos import Plutino

# import numpy
import numpy

# import datetime
import datetime

# import sklearn
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
from sklearn.svm import OneClassSVM, SVC
from sklearn.decomposition import PCA

# import matplotlib for plots
from matplotlib import pyplot
from matplotlib import style as Style
from matplotlib import rcParams
import matplotlib.dates as Dates
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

    def _chart(self, plutinos, tag='all', averaging=14):
        """Plot the spending for a particular tag.

        Arguments:
            plutinos: list of plutino instances
            tag: str, tag for subset
            averaging: int, moving average window for slopes

        Returns:
            None
        """

        # sort by date
        plutinos.sort(key=lambda plutino: plutino.date)

        # accumulate by date
        accumulation = 0.0
        points = {}
        for plutino in plutinos:

            # add quantity to accumulation
            date = datetime.datetime.strptime(plutino.date, '%Y-%m-%d').timestamp() / (3600 * 24)
            accumulation += plutino.quantity
            points[date] = accumulation

        # begin graph
        pyplot.clf()
        pyplot.title(tag)

        # format xaxis as dates
        formatter = Dates.DateFormatter("%m-%d")
        pyplot.gca().xaxis.set_major_formatter(formatter)

        # arrange all points
        points = list(points.items())
        points.sort(key=lambda pair: pair[0])

        # plot line
        horizontals = [point[0] for point in points]
        verticals = [point[1] for point in points]
        pyplot.plot(horizontals, verticals, 'b--')

        # determine slopes
        slopes = []
        for point in points[1:]:

            # find index of point greater than increment
            subset = [pointii for pointii in points if pointii[0] < point[0] - averaging]

            # check for length
            if len(subset) > 0:

                # get latest
                latest = subset[-1]

                # calculate slope
                slope = (point[1] - latest[1]) / (point[0] - latest[0])

                # add to plot
                slopes.append((point[0], slope))

        # plot slopes
        secondary = pyplot.gca().twinx()
        horizontals = [point[0] for point in slopes]
        verticals = [point[1] for point in slopes]
        secondary.plot(horizontals, verticals, 'g--')

        # save plot
        pyplot.savefig('../plots/{}.png'.format(tag))
        pyplot.clf()

        return None

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

        return plutino

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

        return plutino

    def accrete(self):
        """Gather up records, adding classification.

        Arguments:
            None

        Returns:
            None
        """

        # begin plutinos
        plutinos = []

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
                plutinos.append(self._mine(row, plutons))

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
                plutinos.append(self._supply(row, plutons))

            # resave plutons
            self._dump(plutons, '../output/plutons.json')

        # remove duplicates
        skimmer = {(plutino.date, plutino.quantity, plutino.text): plutino for plutino in plutinos}
        plutinos = list(skimmer.values())
        plutinos.sort(key=lambda plutino: plutino.date)

        # add to instance
        [self.append(plutino) for plutino in plutinos]

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

    def pulverize(self):
        """Get reports for tags.

        Arguments:
            None

        Returns:
            None
        """

        # make graph of all
        self._chart(list(self))

        # find all tags
        tags = list(set([plutino.tag for plutino in self]))
        for tag in tags:

            # print tag
            print(tag)

            # make chart
            plutinos = [plutino for plutino in self if plutino.tag == tag]
            self._chart(plutinos, tag)

        return None

    def scan(self):
        """Summarize all categories.

        Arguments:
            None

        Returns:
            None
        """

        # load up plutons
        plutons = self._load('../output/plutons.json')

        # group according to tag
        plutons = list(plutons.items())
        tags = self._group(plutons, lambda pair: pair[1])
        tags = list(tags.items())
        tags.sort(key=lambda pair: pair[0])
        for tag, labels in tags:

            # print
            self._print('\n')
            self._print('{}:'.format(tag))

            # sort members
            labels.sort()
            for label in labels:

                # print label
                self._print('    {}'.format(label[0]))

        return None

    def spelunk(self, tag, label=None):
        """Gather up records, adding classification.

        Arguments:
            tag: str, particular tag to view
            label: str, particular label to view

        Returns:
            None
        """

        # get all plutons
        plutons = [pluton for pluton in self if pluton.tag == tag]

        # if a label if given
        if label:

            # subset to label
            plutons = [pluton for pluton in plutons if pluton.label == label]

        # sort by date
        plutons.sort(key=lambda pluton: pluton.date)

        # report
        self._tell(plutons)

        return None
