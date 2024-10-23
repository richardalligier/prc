# PRC: code from https://github.com/akrherz/iem/blob/main/scripts/asos/iem_scraper_example2.py
# MIT License


"""
Example script that scrapes data from the IEM ASOS download service.

Requires: Python 3
"""

import datetime
import utils
from subprocess import call
import os

def download():
    start = datetime.datetime(2021, 12, 31)
    end = datetime.datetime(2023, 1, 1)
    step = datetime.timedelta(hours=24)
    config = utils.read_config()
    os.makedirs(config.METARs, exist_ok=True)
    now = start
    while now < end:
        sdate = now.strftime("year1=%Y&month1=%m&day1=%d&")
        edate = (now + step).strftime("year2=%Y&month2=%m&day2=%d&")
        print(f"Downloading: {now}")
        fname = os.path.join(config.METARs,f"{now:%Y%m%d}.csv")
        cmd = f'curl "https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py?data=all&report_type=3&{sdate}&{edate}&tz=UTC&format=comma&missing=empty&latlon=yes&elev=yes" > {fname}'
        print(cmd)
        call(cmd,shell=True)
        now += step

if __name__ == "__main__":
    download()
