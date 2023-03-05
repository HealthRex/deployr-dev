"""
Utility functions
"""
from datetime import datetime
from datetime import timedelta
from html.parser import HTMLParser
from io import StringIO

def result_date_parse(datestring):
    """
    Tries to parse datetime from datestring, if fails just returns datestring
    """
    datestring = datestring.rstrip()
    formats = ['%m/%d/%Y %I:%M %p', '%m/%d/%Y']
    result_time = None
    for fmt in formats:
        try:
            return datetime.strptime(datestring, fmt)
        except ValueError as e:
            print(str(e))
            pass
    raise ValueError(f"no valid date format found for {datestring}")

def is_dst():
    """
    Checks whether utc current time is in dst for PT
    """

    utc_timestamp = datetime.utcnow()
    dst_utc = datetime(2022, 11, 6, 9, 0, 0) # this needs to be upated soon...
    return utc_timestamp < dst_utc


def bpa_datetime_parse(datestring):
    """
    Parses the datetime saved upon BPA trigger
    """
    try:
        order_time = datetime.strptime(datestring, '%a %b %d %H:%M:%S %Y')
        if is_dst():  # only a temp solution, will cause issues again when
            # we go back on DST.
            order_time -= timedelta(hours=7)
        else:
            order_time -= timedelta(hours=8)
    except:
        print("Error in bpa datetime parse")
    return order_time


class MLStripper(HTMLParser):
    def __init__(self):
        super().__init__()
        self.reset()
        self.strict = False
        self.convert_charrefs = True
        self.text = StringIO()

    def handle_data(self, d):
        self.text.write(d)

    def get_data(self):
        data = self.text.getvalue()
        data = (data
                .replace('\t', ' ')
                .replace('\n', ' ')
                .replace('\r', ' ')
                .replace('\xa0', ' ')
                )
        data = " ".join(data.split())
        return data
