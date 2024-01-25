# Copyright © 2023-2024 Apple Inc.

import dataclasses
import datetime
import enum
from functools import singledispatch
from typing import Mapping, Sequence, Union


@singledispatch
def encode(arg) -> Union[str, int, float, bool, Sequence, Mapping, None]:
    """Return a string representation to use for logging.

    This generic function is intended to be used as the `default`
    argument to a JSON encoder, dispatching on the type of `arg`.

    The representation is printed if `arg` has an unregistered type.

    As a special case, dataclasses are handled directly in the generic
    function, since they do not have a shared superclass.

    """

    if dataclasses.is_dataclass(arg):
        return arg.asdict()
    return repr(arg)


@encode.register(datetime.date)
def encode_date(arg: datetime.date) -> str:
    return arg.isoformat()


@encode.register(datetime.datetime)
def encode_datetime(arg: datetime.datetime) -> str:
    return arg.isoformat()


@encode.register(datetime.time)
def encode_time(arg: datetime.time) -> str:
    return arg.isoformat()


@encode.register(datetime.timedelta)
def encode_timedelta(arg: datetime.timedelta) -> str:
    return str(arg)


@encode.register(enum.Enum)
def encode_enum(arg: enum.Enum):
    """Return a dict containing the class, name, and value of an enum
    argument.  E.g.,

    ```
    In [1]: encode(PrivacyGuaranteeLocation.CENTRAL_PRIVACY)
    Out[1]: {'cls': 'PrivacyGuaranteeLocation', 'id': 'CENTRAL_PRIVACY',
             'val': 2}
    ```
    """
    return {
        'cls': type(arg).__name__,
        'id': arg.name,
        'val': arg.value,
    }
