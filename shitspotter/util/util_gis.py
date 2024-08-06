"""
GIS helpers
"""
import ubelt as ub
import math
import numpy as np
import warnings
from functools import cache

try:
    from line_profiler import profile
except ImportError:
    profile = ub.identity


@profile
def coco_estimate_sunlight(dset, image_ids=None):
    """
    Estimate the amount of sunlight in each image of a coco dataset based on
    its datetime and geo-location.

    Ignore:
        >>> # xdoctest: +SKIP
        >>> # xdoctest: +REQUIRES(module:suntime)
        >>> import shitspotter
        >>> dset = shitspotter.open_shit_coco()
        >>> image_ids = None
        >>> sunlight_values = estimate_image_sunlight(dset, image_ids)
        >>> # xdoctest: +REQUIRES(--show)
        >>> # xdoctest: +REQUIRES(module:kwplot)
        >>> import kwplot
        >>> import pandas as pd
        >>> sns = kwplot.autosns()
        >>> df = pd.DataFrame({'sunlight': sunlight_values})
        >>> fig = kwplot.figure()
        >>> ax = fig.gca()
        >>> sns.histplot(data=df, x='sunlight')
        >>> ax.set_title('Sunlight Histogram')
        >>> kwplot.show_if_requested()
    """
    from shitspotter.util.util_math import Rational
    try:
        import suntime  # NOQA
        import timezonefinder  # NOQA
        import pytz  # NOQA
    except ImportError:
        raise ImportError(ub.codeblock(
            '''
            Missing requirements, please:
            pip install sumtime timezonefinder pytz
            '''))
    else:
        sunlight_values = []
        images = dset.images(image_ids)
        for img in images.objs_iter:
            if 'geos_point' not in img:
                sunlight = np.nan
            else:
                geos_point = img['geos_point']
                if isinstance(geos_point, float) and math.isnan(geos_point):
                    sunlight = np.nan
                elif not isinstance(geos_point, dict):
                    warnings.warn(f'Warning: unknown geos_point format {geos_point!r} in {img!r}')
                    sunlight = np.nan
                else:
                    coords = geos_point['coordinates']
                    point = [Rational.coerce(x) for x in coords]
                    lon, lat = point
                    datetime = img['datetime']
                    sunlight = estimate_sunlight(lat, lon, datetime)
            sunlight_values.append(sunlight)
        sunlight_values = np.array(sunlight_values)
    return sunlight_values


@cache
def _timezone_finder():
    import timezonefinder
    tzfinder = timezonefinder.TimezoneFinder()
    return tzfinder


@cache
def _constants():
    import kwutil
    one_day = kwutil.timedelta.coerce('1 day')
    one_hour = kwutil.timedelta.coerce('1 hour')
    return one_day, one_hour


@profile
def estimate_sunlight(lat, lon, datetime):
    """
    Heuristic for estimating sunlight from geolocation and time.

    Example:
        >>> # xdoctest: +REQUIRES(module:suntime)
        >>> # xdoctest: +REQUIRES(module:timezonefinder)
        >>> # xdoctest: +REQUIRES(module:pytz)
        >>> from shitspotter.util.util_gis import *  # NOQA
        >>> lat, lon = 42, -73
        >>> datetime = '2024-07-03T10:40:34'
        >>> estimate_sunlight(lat, lon, datetime)
    """
    import suntime
    import pytz
    from scipy import interpolate
    import kwutil
    sun = suntime.Sun(lat, lon)
    tzfinder = _timezone_finder()
    timezone_str = tzfinder.timezone_at(lng=lon, lat=lat)
    timezone = pytz.timezone(timezone_str)
    datetime = kwutil.datetime.coerce(datetime, default_timezone=timezone)

    one_day, one_hour = _constants()
    yesterday = datetime - one_day
    tomorrow = datetime + one_day

    yesterday_sunset = sun.get_sunset_time(yesterday, timezone)
    today_sunrise = sun.get_sunrise_time(datetime, timezone)
    today_sunset = sun.get_sunset_time(today_sunrise, timezone)
    if today_sunset < today_sunrise:
        # hack to work around a bug in suntime
        # https://github.com/SatAgro/suntime/issues/30
        today_sunset = sun.get_sunset_time(today_sunrise + one_day, timezone)
    tomorrow_sunrise = sun.get_sunrise_time(tomorrow, timezone)

    # Hacky linear interpolation to guestimate the amount of light
    # based on sun times. Could do better if we had the elevation
    # and then we could get the sun angle in the sky. Linear
    # interpolation is probably the wrong assumption, fixme later.
    values = [
        {'time': yesterday_sunset + one_hour , 'light': 0.0},
        {'time': today_sunrise - one_hour    , 'light': 0.0},
        {'time': today_sunrise               , 'light': 0.5},
        {'time': today_sunrise + one_hour    , 'light': 1.0},

        {'time': today_sunset - one_hour     , 'light': 1.0},
        {'time': today_sunset                , 'light': 0.5},
        {'time': today_sunset + one_hour     , 'light': 0.0},
        {'time': tomorrow_sunrise - one_hour , 'light': 0.0},
    ]

    xs = [r['time'].timestamp() for r in values]
    ys = [r['light'] for r in values]
    interp = interpolate.interp1d(xs, ys, kind='linear')
    sunlight = interp(datetime.timestamp())

    if 0:
        # plot the heuristic curve for debugging
        import kwplot
        kwplot.autompl()
        test_xs = np.linspace(xs[0], xs[-1], 100)
        test_ys = interp(test_xs)
        kwplot.plt.plot(test_xs, test_ys)
        kwplot.plt.plot(xs, ys, 'o')
    return sunlight
