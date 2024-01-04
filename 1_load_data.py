import sunkit_instruments.goes_xrs
import drms
from sunpy.time import TimeRange
import sunkit_instruments.goes_xrs
from datetime import datetime
from datetime import timedelta
import joblib
import ssl
ssl._create_default_https_context = ssl._create_unverified_context
import time
import pandas as pd
import os
################################################
t_start = '2010.05.01'
t_end = '2023.01.01'
remake_goes_data = False
remake_sharps_data = False

output_path = 'your/own/path'

if not os.path.exists(output_path):
    os.makedirs(output_path)
################################################
if remake_sharps_data:
    date_range = pd.date_range(start=t_start, end=t_end, freq='MS', inclusive='both')
    c = drms.Client()

    # these are the SHARPs parameters downloaded
    key = ['USFLUX, MEANGBT, MEANJZH, MEANPOT, SHRGT45, TOTUSJH, MEANGBH,'
           'MEANALP, MEANGAM, MEANGBZ, MEANJZD, TOTUSJZ, SAVNCPP, TOTPOT,'
           'MEANSHR, AREA_ACR, R_VALUE, ABSNJZH, T_REC, NOAA_AR, HARPNUM, CRVAL1, CRLN_OBS']

    keys_all = []
    # iterate through time range by month, downloading SHARPs data if possible
    for i in range(len(date_range)-1):

        t1 = date_range[i].strftime('%Y.%m.%d')
        t2 = date_range[i+1].strftime('%Y.%m.%d')

        print('Importing SHARPs data between {} -- {}\n'.format(t1, t2))

        try:
            # filter data quality
            keys_all.append(c.query('hmi.sharp_cea_720s[]['+t1+' - '+t2+'][? (abs(OBS_VR) < 3500) and (QUALITY<65536) ?]', key=key[0]))
        except:
            print('Something went wrong with the query, will try in 5 minutes\n')
            time.sleep(5*60)
            try:
                keys_all.append(c.query('hmi.sharp_cea_720s[][' + t1 +
                                        ' - ' + t2 + '][? (abs(OBS_VR) < 3500) and (QUALITY<65536) ?]',
                                        key=key[0]))
            except:
                print('Connection problem still persists, so I am skipping this month\n')
                continue

    keys_all = pd.concat(keys_all).reset_index(drop=True)
    keys_all['time'] = keys_all.T_REC.apply(lambda x: datetime.strptime(x, '%Y.%m.%d_%H:%M:%S_TAI'))

    joblib.dump(keys_all, output_path+'keys_all.pkl')
else:
    keys_all = joblib.load(output_path+'keys_all.pkl')
#######################################################################################################################

print('=======================================================================')
print('                      Importing GOES Flare Data                        ')
print('=======================================================================')

if remake_goes_data:
    time_range = TimeRange(t_start.replace('.','-'), t_end.replace('.','-'))
    # retrieve GOES events within time range
    goes_flare = sunkit_instruments.goes_xrs.get_goes_event_list(time_range)

    print('SHARPs data is at every 0, 12, 24, 36, 48 minutes of an hour, so I will round the peak time minutes to the closest one\n')

    # iterate over retrieved GOES events, saving to pandas dataframe
    df_goes = pd.DataFrame(columns=['start', 'peak', 'end', 'peak_adjusted', 'cls', 'noaa_ar'])
    for i, event in enumerate(goes_flare):
        df_goes.loc[i, 'start'] = event['start_time']
        df_goes.loc[i, 'peak'] = event['peak_time']
        df_goes.loc[i, 'end'] = event['end_time']
        df_goes.loc[i, 'peak_adjusted'] = pd.to_datetime(df_goes.peak[i].value) - \
                                          timedelta(minutes=pd.to_datetime(df_goes.peak[i].value).minute % 12)
        df_goes.loc[i, 'cls'] = event['goes_class']
        df_goes.loc[i, 'noaa_ar'] = event['noaa_active_region']

    df_goes = df_goes.loc[df_goes.noaa_ar != 0].reset_index(drop=True)

    joblib.dump(df_goes, output_path+'df_goes.pkl')

else:
    df_goes = joblib.load(output_path+'df_goes.pkl')


print('Grabbed all the GOES data; there are', df_goes.shape[0], 'events with NOAA AR numbers')

######################################################################################################

A_flr = []
B_flr = []
C_flr = []
M_flr = []
X_flr = []

# iterate over all retrieved GOES events to pair with SHARPs series and separate by class
for i in range(df_goes.shape[0]):
    print('=======================================================================')
    print('Importing data for {} class flare peaked at {} \n'.format(df_goes.cls.iloc[i], df_goes.peak_adjusted.iloc[i]))

    T_REC = df_goes.peak_adjusted.iloc[i]
    noaa = df_goes.noaa_ar.iloc[i]
    if len(str(noaa)) < 5:
        noaa = int(noaa + 1e4)

    # locate SHARPs data matching the event's NOAA number, extracting data before the flare peak
    slice = keys_all.loc[(keys_all.NOAA_AR == noaa) & (keys_all.time <= T_REC)].reset_index(drop=True)

    # if SHARPs data found, save, sorted by flare class
    if slice.empty:
        print('There is no SHARP data for NOAA number {} for flare {} on {}'.format(noaa, df_goes.cls.iloc[i], T_REC))
    else:
        if 'A' in df_goes.cls.iloc[i]:
            A_flr.append(slice)
            joblib.dump(A_flr, output_path+'A_flr.pkl')
            if len(A_flr) % 10 == 0:
                print('There are {} A class flares detected so far!\n'.format(len(A_flr)))
        elif 'B' in df_goes.cls.iloc[i]:
            B_flr.append(slice)
            joblib.dump(B_flr, output_path+'B_flr.pkl')
            if len(B_flr) % 10 == 0:
                print('There are {} B class flares detected so far!\n'.format(len(B_flr)))
        elif 'C' in df_goes.cls.iloc[i]:
            C_flr.append(slice)
            joblib.dump(C_flr, output_path+'C_flr.pkl')
            if len(C_flr) % 10 == 0:
                print('There are {} C class flares detected so far!\n'.format(len(C_flr)))
        elif 'M' in df_goes.cls.iloc[i]:
            M_flr.append(slice)
            joblib.dump(M_flr, output_path+'M_flr.pkl')
            if len(M_flr) % 10 == 0:
                print('There are {} M class flares detected so far!\n'.format(len(M_flr)))
        elif 'X' in df_goes.cls.iloc[i]:
            X_flr.append(slice)
            joblib.dump(X_flr, output_path+'X_flr.pkl')
            if len(X_flr) % 10 == 0:
                print('There are {} X class flares detected so far!\n'.format(len(X_flr)))

print('=======================================================================')
print('=======================================================================')

print('There are {} A class Flares in total\n'.format(len(A_flr)))
print('There are {} B class Flares in total\n'.format(len(B_flr)))
print('There are {} C class Flares in total\n'.format(len(C_flr)))
print('There are {} M class Flares in total\n'.format(len(M_flr)))
print('There are {} X class Flares in total\n'.format(len(X_flr)))

N_tot = int(len(A_flr) + len(B_flr) + len(C_flr) + len(M_flr) + len(X_flr))

print('=============================================================================')
print('=============       There are {} Flares detected in total        ============'.format(N_tot))
print('=============  between {} and {}    ===================='.format(t_start, t_end))
print('=============================================================================')


