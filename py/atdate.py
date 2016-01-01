#!/usr/bin/python
# -*- coding: utf-8 -*-

#  Import numpy
import numpy as np
import datetime as dt

def get_date_range(sdate0, sdatef, incr=1, format='%Y%m%d',
                   leap_year=True, start_the_first = False):
# !!! start_the_first only works when leap_year = False
    daysInMonth = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    date0 = dt.datetime.strptime(sdate0, format).date()
    datef = dt.datetime.strptime(sdatef, format).date()
    day_delta = dt.timedelta(incr)
    
    date = date0
    rng_date = []
    if leap_year:
        while date <= datef:
            day = date.day
            month = date.month
            year = date.year
            # Do not use strftime in case of year < 1900
            rng_date.append('%04d%02d%02d' % (year, month, day))
            date += day_delta
    else:
        while date <= datef:
            day = date.day
            month = date.month
            year = date.year
            # Do not use strftime in case of year < 1900
            rng_date.append('%04d%02d%02d' % (year, month, day))
            # Increment
            day += incr
            while day > daysInMonth[month - 1]:
                day -= daysInMonth[month - 1]
                month += 1
                while (month > 12):
                    month -= 12
                    year += 1
                    if start_the_first:
                        day = 1
            date = dt.date(year, month, day)
    return rng_date
