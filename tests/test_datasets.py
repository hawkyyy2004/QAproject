"""Data files become schema, ranges and sampled rows -- never one unit per row."""
import json
from pathlib import Path

import numpy
import pytest

from qaproject import datasets
from qaproject.parsers import extract
from qaproject.settings import settings
from qaproject.store import Store

# netCDF4 itself still reshapes the array it is handed (NumPy 2.5 deprecation).
pytestmark = pytest.mark.filterwarnings(
    'ignore:Setting the shape on a NumPy array:DeprecationWarning')


WOCE = """CTD,20100713SIOCCHDLIB
# Auto-generated Exchange CTD file
EXPOCODE = 06AQ144
STNNBR = 15
LATITUDE = -60.9967
LONGITUDE = -58.5067
THETA,CTDPRS,CTDTMP,CTDSAL
DEG C,DBAR,ITS-90,PSS-78
 -0.2721,     4.0, -0.2720, 33.9070
 -0.2862,     6.0, -0.2860, 33.9120
 -0.3012,     8.0, -0.3010, 33.9160
  1.7436,   796.0,  1.7890, 34.7000
END_DATA
"""


def test_woce_csv_names_units_and_ranges(tmp_path):
    path = tmp_path/'station.csv'
    path.write_text(WOCE, encoding='utf-8')
    items = datasets.csv_units(path)
    schema = next(i for i in items if i['evidence_kind'] == 'csv_schema')
    assert 'THETA(DEG C)' in schema['text'] and 'CTDPRS(DBAR)' in schema['text']
    assert 'STNNBR = 15' in schema['text']
    stats = next(i for i in items if i['evidence_kind'] == 'csv_stats')
    assert 'CTDPRS/DBAR: 4~796' in stats['text']
    rows = [i for i in items if i['evidence_kind'] == 'csv_rows']
    # Four data rows stay one small table; END_DATA is not a record.
    assert len(rows) == 1 and 'END_DATA' not in rows[0]['text']


def test_every_csv_unit_repeats_the_cast_identity(tmp_path):
    """A salinity question names the station; the unit it came from must prove it."""
    path = tmp_path/'station.csv'
    path.write_text(WOCE, encoding='utf-8')
    items = datasets.csv_units(path)
    for item in items[1:]:
        assert item['text'].startswith('站位背景：EXPOCODE=06AQ144')
        assert 'STNNBR=15' in item['text'].splitlines()[0]
    assert '数值范围' in items[1]['text']


def test_annotation_table_is_sampled_not_cut_into_prose(tmp_path):
    """A manifest of ids, labels and paths has no mostly-numeric record.

    Treating it as plain text used to cut one unit per 30 lines: a 95,000-row
    manifest became 3,171 units and 1,583 model calls for one file.
    """
    header = ('segment_uid,clip_uid,source_relpath,dataset,split,label,group,'
              'seg_start_sec,seg_end_sec,padded,npy_relpath')
    body = '\n'.join(
        f'{i:04x}_s0000.00,{i:04x},vessel/310.wav,vtuad,train,tug,vessel/310,'
        f'{i*3}.0,{i*3+3}.0,0,train/tug/{i}.npy' for i in range(120))
    path = tmp_path/'manifest.csv'
    path.write_text(header+'\n'+body+'\n', encoding='utf-8')
    items = datasets.csv_units(path)
    assert items[0]['text'].startswith('数据表：120 行 x 11 列')
    assert items[0]['evidence_kind']=='csv_schema'
    assert 'segment_uid' in items[0]['text'] and 'npy_relpath' in items[0]['text']
    assert all(item['evidence_kind']!='csv_text' for item in items)
    assert len(items)<=5


def test_short_table_splits_into_two_windows(tmp_path):
    # 30 rows is too small for the three-window layout but still holds two
    # depth regimes worth asking about separately.
    path = tmp_path/'cast.csv'
    body = '\n'.join(f'{i},{20-i*0.5:.1f}' for i in range(30))
    path.write_text('depth,temperature\n(m),(degC)\n'+body+'\n', encoding='utf-8')
    rows = [i for i in datasets.csv_units(path) if i['evidence_kind'] == 'csv_rows']
    assert [i['locator'] for i in rows] == ['table-01', 'table-02']
    assert '第 1' in rows[0]['text'] and '第 16' in rows[1]['text']


def test_excel_style_csv_keeps_header_without_inventing_units(tmp_path):
    path = tmp_path/'plain.csv'
    path.write_text('depth,temperature\n0,12.5\n10,11.9\n20,11.2\n', encoding='utf-8')
    items = datasets.csv_units(path)
    schema = items[0]['text']
    assert 'depth' in schema and 'temperature' in schema
    assert '(temperature)' not in schema
    assert '数值范围' in items[1]['text']


def test_csv_latin1_degree_sign_is_not_read_as_chinese(tmp_path):
    # 0xB0 followed by "C" is a valid gb18030 sequence, so a decoder that always
    # tries Chinese first turns "°C" into a nonsense character.
    path = tmp_path/'deg.csv'
    path.write_bytes('depth,temperature\r\n(m),\u00b0C\r\n0,12.5\r\n10,11.9\r\n'.encode('cp1252'))
    schema = datasets.csv_units(path)[0]['text']
    assert 'depth((m))' in schema and 'temperature(\u00b0C)' in schema


def test_csv_gb18030_names_survive(tmp_path):
    path = tmp_path/'cn.csv'
    path.write_bytes('深度,温度\n(m),摄氏度\n0,12.5\n10,11.9\n'.encode('gb18030'))
    schema = datasets.csv_units(path)[0]['text']
    assert '深度' in schema and '温度' in schema


def netcdf_file(path):
    netCDF4 = pytest.importorskip('netCDF4')
    dataset = netCDF4.Dataset(str(path), 'w')
    dataset.createDimension('time', 2)
    dataset.createDimension('depth', 3)
    dataset.createDimension('lat', 4)
    dataset.title = 'Test climatology'
    time = dataset.createVariable('time', 'f8', ('time',))
    time.units = 'days since 2000-01-01'
    time[:] = [0, 30]
    depth = dataset.createVariable('depth', 'f4', ('depth',))
    depth.units = 'meters'
    depth[:] = [0, 100, 1000]
    lat = dataset.createVariable('lat', 'f4', ('lat',))
    lat.units = 'degrees_north'
    lat[:] = [-60, -20, 20, 60]
    temp = dataset.createVariable('temperature', 'f4', ('time', 'depth', 'lat'), fill_value=-9.0)
    temp.units = 'degC'
    temp.long_name = 'sea water temperature'
    temp[:] = numpy.asarray([[[20, 18, 16, 12], [12, 11, 10, 8], [4, 3, 2, 2]],
                             [[21, 19, 17, 13], [13, 12, 11, 9], [5, 4, 3, 3]]], dtype='f4')
    salinity = dataset.createVariable('salinity', 'f4', ('time', 'depth', 'lat'), fill_value=-9.0)
    salinity.units = 'PSU'
    salinity[:] = numpy.asarray([[[35, 35.1, 35.2, 34.9], [34.8, 34.8, 34.9, 34.7], [34.7, 34.7, 34.7, 34.7]],
                                 [[35, 35.1, 35.2, 34.9], [34.8, 34.8, 34.9, 34.7], [34.7, 34.7, 34.7, 34.7]]],
                                dtype='f4')
    dataset.close()
    return path


def test_netcdf_reports_schema_and_annotates_depth(tmp_path):
    path = netcdf_file(tmp_path/'oa.nc')
    items = datasets.netcdf_units(path)
    schema = next(i for i in items if i['locator'] == 'nc-schema')
    assert 'Test climatology' in schema['text']
    assert 'temperature[degC]' in schema['text']
    assert '2000-01-01' in schema['text']
    temperature = next(i for i in items if i['locator'] == 'var-temperature')
    # time is the leading axis and is pinned to one date, so rows are depth levels.
    assert 'time=2000-01-31' in temperature['text']
    assert '1(100)' in temperature['text']
    assert '列 0(-60)' in temperature['text']


def test_netcdf_variable_without_data_is_not_given_a_unit(tmp_path):
    netCDF4 = pytest.importorskip('netCDF4')
    path = tmp_path/'empty.nc'
    dataset = netCDF4.Dataset(str(path), 'w')
    dataset.createDimension('pressure', 4)
    variable = dataset.createVariable('silicate', 'f4', ('pressure',), fill_value=-9.0)
    variable[:] = [-9.0, -9.0, -9.0, -9.0]
    dataset.close()
    items = datasets.netcdf_units(path)
    assert [i['locator'] for i in items] == ['nc-schema']


@pytest.mark.parametrize('name,payload', [
    ('cast.csv', WOCE.encode('utf-8')),
    ('cast.tsv', 'depth\ttemp\n0\t12.5\n10\t11.9\n20\t11.2\n'.encode('utf-8')),
])
def test_extract_routes_data_files(tmp_path, name, payload):
    store = Store(tmp_path/'data')
    doc_id, _ = store.import_file(name, payload, 'live')
    units, pages = extract(store, store.document(doc_id), settings())
    kinds = {unit.get('evidence_kind') for unit in units}
    assert kinds & {'csv_schema', 'csv_rows'}
    assert pages == 0
    assert all(unit['image'] == '' for unit in units)


def test_extract_routes_netcdf(tmp_path):
    store = Store(tmp_path/'nc')
    path = netcdf_file(tmp_path/'oa.nc')
    doc_id, _ = store.import_file('oa.nc', path.read_bytes(), 'live')
    units, _ = extract(store, store.document(doc_id), settings())
    assert {u.get('evidence_kind') for u in units} == {'netcdf_schema', 'netcdf_variable'}
    assert any('degC' in unit['text'] for unit in units)
