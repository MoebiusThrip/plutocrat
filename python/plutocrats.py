# plutocrats.py to manage finances

# import modules
from cores import Core
from features import Feature
from formulas import Formula
from hydras import Hydra
from squids import Squid
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

        # initalize squid
        self.squid = Squid('..')

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

    def _average(self, plutinos, window):
        """Create moving average based on a window of days.

        Arguments:
            plutinos: list of plutinos
            window: int, number of days

        Returns:
            list of floats
        """

        # begin average
        average = []

        # collect datetimes
        dates = [datetime.datetime.strptime(plutino.date, '%Y-%m-%d') for plutino in plutinos]
        quantities = [plutino.quantity for plutino in plutinos]

        # for each plutiono
        for quantity, date in zip(quantities, dates):

            # set adcumulation
            accumulation = 0.0

            # for each other plutino
            for quantityii, dateii in zip(quantities, dates):

                # get the delta
                delta = dateii - date
                if (delta > datetime.timedelta(days=-window)) & (delta <= datetime.timedelta(days=0)):

                    # add to accumulation
                    accumulation += quantityii / window

            # add acumulatino
            average.append(accumulation)

        return average

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

    def _draw(self, tag='all', averaging=14):
        """Plot the spending for a particular tag as hdf5 file.

        Arguments:
            plutinos: list of plutino instances
            tag: str, tag for subset
            averaging: int, moving average window for slopes

        Returns:
            None
        """

        # get all plutios by default
        plutinos = [plutino for plutino in self]
        if tag != 'all':

            # get relevant plutinos and sort by date
            plutinos = [plutino for plutino in self if plutino.tag == tag]

        # sort
        plutinos.sort(key=lambda plutino: plutino.date)

        # get times as abscissa
        dates = [datetime.datetime.strptime(plutino.date, '%Y-%m-%d').timestamp() for plutino in plutinos]
        dates = [date * 1000 for date in dates]

        # get expenses
        expenses = [plutino.quantity for plutino in plutinos]

        # compute week moving average
        week = self._average(plutinos, 7)
        month = self._average(plutinos, 31)

        # add auxiliaries
        auxiliaries = {}
        auxiliaries['cost'] = [plutino.quantity for plutino in plutinos]
        auxiliaries['label'] = [plutino.label for plutino in plutinos]
        auxiliaries['date'] = [plutino.date for plutino in plutinos]
        auxiliaries['text'] = [plutino.text for plutino in plutinos]
        auxiliaries['account'] = [plutino.account for plutino in plutinos]

        # make plot
        address = 'ores/{}_record.h5'.format(tag)
        title = '{} expenditures'.format(tag)
        ordinates = [expenses, week, month]
        self.squid.splatter('{}_expenses'.format(tag), ordinates, 'time', dates, address, title, auxiliaries)

        # if last date within range
        if (datetime.datetime.now() - datetime.datetime.fromtimestamp(dates[-1] / 1000)) < datetime.timedelta(days=30):

            # print tag and latest monthly average
            self._print('{}: {}'.format(tag, month[-1]))

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

    def _supply(self, row, plutons, mode=False):
        """Mine the statement entry and add to the records.

        Arguments:
            row: list of str
            plutons: dict of plutons
            mode: boolean, new exported style?

        Returns:
            None

        Populates:
            self
        """

        # set account to citi
        account = 'cafcu'

        # if not mode
        if not mode:

            # unpack row
            date, description, comments, check, amount, balance = row

        # if mode is true
        if mode:

            # unpack differently
            _, date, _, _, amount, check, _, description, _, _, balance, _, comments = row

            # replace hyphens
            comments = comments.replace('\u2013', '-')
            description = description.replace('\u2013', '-')

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

            # check for new format mode
            mode = False
            if 'ExportedTransactions' in path:

                # set mode to True
                mode = True

            # load up the csv
            rows = self._tape(path)

            # go through each row, skipping headers
            for row in rows[1:]:

                # mine the row
                plutinos.append(self._supply(row, plutons, mode=mode))

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

        # clear old plots
        self._clean('../plots')
        self._clean('../ores')

        # make graph of all
        self._chart(list(self))
        self._draw()

        # find all tags
        tags = list(set([plutino.tag for plutino in self]))
        for tag in tags:

            # print tag
            print(tag)

            # make hdf5 file
            self._draw(tag)

            # make chart
            plutinos = [plutino for plutino in self if plutino.tag == tag]
            self._chart(plutinos, tag)

        return None

    def repair(self, label, tag):
        """Recategorize a label.

        Arguments:
            label: str, the label
            tag: str, new tag

        Returns:
            None
        """

        # load up plutons
        plutons = self._load('../output/plutons.json')

        # add pluton
        plutons[label] = tag

        # resave plutons
        self._dump(plutons, '../output/plutons.json')

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
        """Print records for a tag.

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
