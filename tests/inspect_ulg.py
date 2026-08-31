"""Quick inspection of ULog fields."""
from pyulog import ULog
import math

ulog = ULog('tests/data/uav_sead/18_01_59.ulg')
lpos = next(d for d in ulog.data_list if d.name == 'vehicle_local_position')
att  = next(d for d in ulog.data_list if d.name == 'vehicle_attitude')

yaw_vals = lpos.data['yaw']
print('Yaw min={:.2f} max={:.2f} rad  ({:.1f} to {:.1f} deg)'.format(
    min(yaw_vals), max(yaw_vals),
    math.degrees(min(yaw_vals)), math.degrees(max(yaw_vals))))

wind = next((d for d in ulog.data_list if d.name == 'wind_estimate'), None)
if wind:
    wn = wind.data['windspeed_north']
    we = wind.data['windspeed_east']
    print('Wind N: {:.3f} to {:.3f} m/s'.format(min(wn), max(wn)))
    print('Wind E: {:.3f} to {:.3f} m/s'.format(min(we), max(we)))

baro = next((d for d in ulog.data_list if d.name == 'sensor_baro'), None)
if baro:
    alt = baro.data['altitude']
    print('Baro alt: {:.1f} to {:.1f} m'.format(min(alt), max(alt)))

xv = lpos.data['x']
yv = lpos.data['y']
zv = lpos.data['z']
print('NED x: {:.3f} to {:.3f} m'.format(min(xv), max(xv)))
print('NED y: {:.3f} to {:.3f} m'.format(min(yv), max(yv)))
print('NED z: {:.3f} to {:.3f} m'.format(min(zv), max(zv)))
print('LocalPos rows:', len(lpos.data['timestamp']))
print('Attitude rows:', len(att.data['timestamp']))
