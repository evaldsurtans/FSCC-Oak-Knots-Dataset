import datetime
import json


def json_converter(o):
    if isinstance(o, datetime.datetime):
        return o.__str__()


def json_dumps(o, indent=None):
    try:
        return json.dumps(o, default=json_converter, indent=indent)
    except:
        pass
    return '{error serialization}'

