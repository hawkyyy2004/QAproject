"""Turn data files into evidence a language model can reason about.

A netCDF file or a 40,000-row CSV is not text: sending it whole is impossible
and sending only its first rows teaches nothing. Both readers here produce
compact, self-describing excerpts -- dimensions, variables with units, ranges
and a sampled slice -- so a question can be about what the data says instead of
what the file is called.

Nothing here loads a whole array: slices and decimation only, so a 176 MB
climatology costs the same as a 20 KB station file.
"""
import csv
import datetime as dt
import io
import math
import re
from collections import Counter
from pathlib import Path

MAX_UNIT_CHARS = 3500
NUMBER = re.compile(r'^[+-]?(\d+\.?\d*|\.\d+)([eE][+-]?\d+)?$')
METADATA_LINE = re.compile(r'^([A-Za-z_][\w .\-]{0,40})\s*=\s*(.+)$')

# Identity of a cast or profile. Every unit repeats it so that a question
# naming the station can be checked against the unit it was generated from;
# without it the reviewer rejects the question as unverifiable scope.
CONTEXT_KEYS = ('EXPOCODE','SECT','STNNBR','CASTNO','DATE','TIME',
                'LATITUDE','LONGITUDE','DEPTH')
SKIP_VARIABLE = re.compile(r'(_qc$|_quality_flag$|_bnds$|^crs$|^time_bnds$)', re.I)
SCIENCE_HINT = ('temperature', 'salinity', 'oxygen', 'nitrate', 'phosphate', 'silicate',
                'chlorophyll', 'density', 'carbon', 'alkalinity', 'ph', 'depth', 'pressure',
                'velocity', 'ssha', 'sst', 'anomaly', 'flux')


def _clean(value):
    return re.sub(r'\s+', ' ', str(value)).strip()


def _number(text):
    text = text.strip()
    if not NUMBER.match(text):
        return None
    try:
        value = float(text)
    except ValueError:
        return None
    # Sentinel codes used across oceanographic archives; they are not measurements.
    return None if value in (-9.0, -99.0, -999.0, -9999.0) else value


def _fmt(value):
    if value is None:
        return ''
    if abs(value) >= 100000 or (value and abs(value) < 0.001):
        return f'{value:.4g}'
    return f'{value:.4f}'.rstrip('0').rstrip('.')


def _decode(raw):
    try:
        return raw.decode('utf-8-sig')
    except UnicodeDecodeError:
        pass
    try:
        text = raw.decode('gb18030')
    except UnicodeDecodeError:
        return raw.decode('cp1252', 'replace')
    # gb18030 also "succeeds" on stray Latin-1 bytes such as a lone degree sign
    # ("°C" becomes one Chinese character), so trust it only when Chinese is a
    # real share of the text rather than a single accidental match.
    if text and len(re.findall(r'[\u4e00-\u9fff]', text)) / len(text) >= 0.05:
        return text
    return raw.decode('cp1252', 'replace')


def _delimiter(sample):
    counts = {d: sample.count(d) for d in (',', ';', '\t', '|')}
    best = max(counts, key=counts.get)
    return best if counts[best] else ','


def _unit_like(row):
    """WOCE exports put units under the names: 'DEG C', 'PSS-78', 'UMOL/KG'."""
    # A whole cell in parentheses -- '(m)', '(deg C)' -- is a unit, not a name.
    return sum(1 for cell in row
               if re.search(r'[ /°^]|^[A-Z]{2,}-\d|^\(.*\)$', cell.strip()))


def _split_table(lines, delimiter):
    """Find the header (and unit row) above the first all-numeric record.

    Two-row headers are ambiguous: an Excel export has names directly above the
    data, a WOCE export has units there and names one row higher. Whichever of
    the two candidate rows looks more like units is the unit row.
    """
    reader = list(csv.reader(lines, delimiter=delimiter))
    start = next((i for i, row in enumerate(reader)
                  if row and sum(_number(x) is not None for x in row) >= max(1, len(row) // 2)), None)
    if start is None:
        # Annotation tables hold ids, labels and paths, so no record is mostly
        # numeric. Treating them as prose cuts one unit per 30 lines: a 95,000
        # row manifest became 3,171 units and 1,583 model calls. The most common
        # field count still identifies the header of a real table.
        widths = Counter(len(row) for row in reader if len(row) > 1).most_common(1)
        if not widths or widths[0][1] < 2:
            return None, [], [], []
        width = widths[0][0]
        start = next((i for i, row in enumerate(reader) if len(row) == width), None)
        if start is None:
            return None, [], [], []
        return reader[start], [], [line for line in lines[:start] if line.strip()], reader[start + 1:]
    near = reader[start - 1] if start >= 1 else []
    far = reader[start - 2] if start >= 2 else []
    header, units = near, []
    if far and len(far) == len(near) and not any(_number(x) is not None for x in far):
        if _unit_like(near) > _unit_like(far):
            header, units = far, near
        elif _unit_like(far) > _unit_like(near):
            header, units = near, far
    first = start - 1 if not units else start - 2
    metadata = [line for line in lines[:max(0, first)] if line.strip()]
    return header, units, metadata, reader[start:]


def csv_units(path, *, max_rows=200000, sample=12, chars=MAX_UNIT_CHARS):
    """One metadata unit plus sampled table units; never one unit per row."""
    text = _decode(Path(path).read_bytes())
    lines = text.splitlines()
    delimiter = _delimiter('\n'.join(lines[:40]))
    header, units, metadata, rows = _split_table(lines, delimiter)
    items = []
    if not rows:
        # Not a table at all: fall back to the plain text the file actually holds.
        body = [line for line in lines if line.strip() and not line.lstrip().startswith('#')]
        if not body:
            raise ValueError('CSV 中没有可读取的数据行')
        for index in range(0, len(body), 30):
            chunk = body[index:index+30]
            items.append({'text': '\n'.join(chunk), 'locator': f'record-{index+1:04d}',
                          'evidence_kind': 'csv_text', 'seg': index//30+1})
        return items
    rows = [row for row in rows if row]
    if not rows:
        raise ValueError('CSV 中没有可读取的数据行')
    # Trailing markers such as END_DATA and half-filled notes are not records.
    # A row is a record when it carries most of the header's columns; requiring
    # numeric cells instead would drop every annotation table.
    threshold = max(2, max(len(row) for row in rows) // 2)
    rows = [row for row in rows if len(row) >= threshold]
    width = max(len(row) for row in rows)
    header = (header + ['']*width)[:width]
    names = [_clean(x) or f'column_{i+1}' for i, x in enumerate(header)]
    units = (units + ['']*width)[:width]
    columns = [dict(name=names[i], unit=_clean(units[i]) if i < len(units) else '') for i in range(width)]
    stats = [dict(column=columns[i]['name'], unit=columns[i]['unit'], n=0, lo=None, hi=None, total=0.0)
             for i in range(width)]
    for row in rows[:max_rows]:
        for i in range(min(width, len(row))):
            value = _number(row[i])
            if value is None:
                continue
            s = stats[i]
            s['n'] += 1
            s['total'] += value
            s['lo'] = value if s['lo'] is None else min(s['lo'], value)
            s['hi'] = value if s['hi'] is None else max(s['hi'], value)
    head = [f'数据表：{len(rows)} 行 x {width} 列，分隔符“{delimiter}”']
    kept = [line for line in metadata if not line.lstrip().startswith('#')]
    facts = {}
    for line in kept[:18]:
        match = METADATA_LINE.match(line.strip())
        if match:
            key, value = match.group(1).strip(), match.group(2).strip()[:120]
            head.append(f'{key} = {value}')
            facts.setdefault(key.upper(), value)
    head.append('列：' + '；'.join(
        f'{c["name"]}' + (f'({c["unit"]})' if c['unit'] else '') for c in columns))
    items.append({'text': '\n'.join(head)[:chars], 'locator': 'table-schema',
                  'evidence_kind': 'csv_schema', 'seg': 1, 'columns': columns})
    context = '；'.join(f'{k}={facts[k]}' for k in CONTEXT_KEYS
                        if facts.get(k) and facts[k].lower() != 'none')
    prefix = f'站位背景：{context}\n' if context else ''

    def unit_text(body):
        return prefix + body[:max(0, chars - len(prefix))]

    ranges = [f'{s["column"]}{"/"+s["unit"] if s["unit"] else ""}: '
              f'{_fmt(s["lo"])}~{_fmt(s["hi"])}，均值 {_fmt(s["total"]/s["n"]) if s["n"] else "—"}'
              for s in stats if s['n']]
    if ranges:
        items.append({'text': unit_text('数值范围（全表统计）\n' + '\n'.join(ranges)),
                      'locator': 'table-stats', 'evidence_kind': 'csv_stats', 'seg': 2})
    # Evenly spaced windows, so middle and end of the file are represented too.
    windows = 3 if len(rows) > sample * 4 else 2 if len(rows) >= sample * 2 else 1
    size = len(rows) // windows
    for w in range(windows):
        begin = w * size
        chunk = rows[begin:begin+size]
        if not chunk:
            continue
        step = max(1, len(chunk)//sample)
        picked = [(begin+i, chunk[i]) for i in range(0, len(chunk), step)][:sample]
        table = ['\t'.join(columns[i]['name'] + (f'/{columns[i]["unit"]}' if columns[i]['unit'] else '')
                           for i in range(width))]
        table += ['\t'.join(_fmt(_number(x)) or x.strip() for x in row[:width]) for _, row in picked]
        label = f'第 {begin+1}–{begin+len(chunk)} 行' if windows > 1 else '全表'
        items.append({'text': unit_text(f'数据表摘录（{label}，每 {step} 行取一行）\n' + '\n'.join(table)),
                      'locator': f'table-{w+1:02d}', 'evidence_kind': 'csv_rows', 'seg': 3+w})
    return items


def _attribute(dataset, names, limit=600):
    for name in names:
        value = getattr(dataset, name, None)
        if isinstance(value, str) and value.strip():
            return _clean(value)[:limit]
    return ''


def _time_text(variable):
    try:
        units = getattr(variable, 'units', '')
        values = variable[:]
        if not len(values):
            return ''
        if 'since' in str(units):
            import netCDF4
            first, last = netCDF4.num2date(values[0], units), netCDF4.num2date(values[-1], units)
            return f'{first:%Y-%m-%d %H:%M} 至 {last:%Y-%m-%d %H:%M}'
        return f'{_fmt(float(values[0]))} 至 {_fmt(float(values[-1]))} {units}'.strip()
    except Exception:
        return ''


def _axis_summary(variable, name):
    text = _time_text(variable) if name.lower() == 'time' else ''
    if text:
        return text
    try:
        values = variable[:]
        if not len(values):
            return ''
        return f'{_fmt(float(values.min()))} 至 {_fmt(float(values.max()))} {getattr(variable, "units", "")}'.strip()
    except Exception:
        return ''


def _labels(dataset, dimension):
    """Coordinate values for a dimension, when the file provides them."""
    if dataset is None or dimension not in dataset.variables:
        return None
    import numpy
    try:
        variable = dataset.variables[dimension]
        values = numpy.ma.filled(numpy.ma.asarray(variable[:], dtype=float), numpy.nan)
    except Exception:
        return None
    if values.ndim != 1:
        return None
    units = str(getattr(variable, 'units', ''))
    if 'since' in units:
        # A time axis in "days since 1900" says nothing to a reader; show dates.
        try:
            import netCDF4
            return [f'{netCDF4.num2date(v, units, only_use_cftime_datetimes=False):%Y-%m-%d}' for v in values]
        except Exception:
            return None
    return [float(v) for v in values]


def _sample_variable(variable, name, dataset=None, limit=60):
    """Decimated 1-D or first-2-D slice; never the whole array.

    Row and column indices are annotated with the real coordinate value
    (depth 1500 m, latitude 30 N), otherwise "row 42" means nothing.
    """
    try:
        variable.set_auto_mask(True)
        shape = variable.shape
        # Centre of any leading axes: a gridded field was previously sampled at
        # its first latitude, i.e. always the pole.
        index = tuple(s // 2 for s in shape[:-2]) if len(shape) > 2 else ()
        data = variable[index] if index else variable[:]
        import numpy
        array = numpy.ma.filled(numpy.ma.asarray(data, dtype=float), numpy.nan)
    except Exception:
        return ''
    # The slice keeps the last two dimensions; annotate those, not the ones cut off.
    row_axis = _labels(dataset, variable.dimensions[-2] if len(variable.shape) > 1 else variable.dimensions[-1]) \
        if variable.shape else None
    col_axis = _labels(dataset, variable.dimensions[-1]) if len(variable.shape) > 1 else None

    def tag(index, axis):
        if axis is None or index >= len(axis):
            return str(index)
        value = axis[index]
        if isinstance(value, str):
            return f'{index}({value})'
        if not math.isfinite(value):
            return str(index)
        return f'{index}({_fmt(float(value))})'

    array = numpy.atleast_1d(array)
    array = array.reshape(-1) if array.ndim == 1 else array
    rows = []
    if array.ndim == 1:
        step = max(1, len(array) // limit)
        rows = [f'{tag(i*step, row_axis)}: {_fmt(float(v)) if math.isfinite(v) else "—"}'
                for i, v in enumerate(array[::step][:limit])]
    else:
        height = min(len(array), 24)
        step_row = max(1, len(array) // height)
        # Columns span the whole secondary axis, so a level-ordered axis shows
        # surface through bottom instead of only its first few entries.
        width = min(array.shape[1], 8)
        cols = [round(i * (array.shape[1] - 1) / max(1, width - 1)) for i in range(width)] \
            if array.shape[1] > 1 else [0]
        if col_axis is not None:
            rows.append('列 ' + '\t'.join(tag(c, col_axis) for c in cols))
        for i in range(0, len(array), step_row):
            cells = [_fmt(float(array[i][c])) if math.isfinite(array[i][c]) else '—' for c in cols]
            rows.append(f'{tag(i, row_axis)}: ' + '\t'.join(cells))
            if len(rows) >= height:
                break
    return '\n'.join(rows)


def _fixed_axes(dataset, variable):
    """Say which plane the slice came from; without it a map has no depth."""
    shape, dims = variable.shape, variable.dimensions
    if len(shape) <= 2:
        return ''
    parts = []
    for dim, size in zip(dims[:-2], shape[:-2]):
        index = size // 2
        axis = _labels(dataset, dim)
        if axis is not None and index < len(axis):
            value = axis[index]
            shown = value if isinstance(value, str) else _fmt(float(value)) if math.isfinite(value) else ''
            parts.append(f'{dim}={shown}' if shown else f'{dim} 第 {index} 个')
        else:
            parts.append(f'{dim} 第 {index} 个')
    return '取样平面（其余维取中间层）：' + '，'.join(parts)


def netcdf_units(path, *, max_variables=8, chars=MAX_UNIT_CHARS):
    """Schema unit plus one sampled unit per science variable."""
    import netCDF4
    try:
        dataset = netCDF4.Dataset(str(path), 'r')
    except Exception as error:
        raise ValueError('无法读取该 netCDF 文件，请确认是有效的 .nc/.nc4。') from error
    try:
        attributes = {name: getattr(dataset, name) for name in ('title', 'summary', 'description',
                                                                'institution', 'source', 'references',
                                                                'comment', 'history')
                      if isinstance(getattr(dataset, name, None), str)}
        dims = {name: len(dim) for name, dim in dataset.dimensions.items()}
        variables = list(dataset.variables.items())
        science = [(name, var) for name, var in variables
                   if name.lower() not in {'time', 'lat', 'latitude', 'lon', 'longitude', 'depth',
                                           'pressure', 'z', 'lev', 'level', 'crs'}
                   and not SKIP_VARIABLE.search(name)
                   and getattr(var, 'dtype', None) is not None
                   and var.dtype.kind in 'fiu']
        def rank(item):
            name = item[0].lower()
            hits = [index for index, hint in enumerate(SCIENCE_HINT) if hint in name]
            return (min(hits) if hits else len(SCIENCE_HINT), len(item[1].shape) > 2, name)
        science.sort(key=rank)
        axes = {}
        for name in ('time', 'lat', 'latitude', 'lon', 'longitude', 'depth', 'pressure', 'z'):
            if name in dataset.variables:
                summary = _axis_summary(dataset.variables[name], name)
                if summary:
                    axes[name] = summary
        head = [f'netCDF 数据文件：{Path(path).name}']
        for key in ('title', 'summary', 'institution', 'source'):
            if attributes.get(key):
                head.append(f'{key}: {_clean(attributes[key])[:300]}')
        head.append('维度：' + '，'.join(f'{k}={v}' for k, v in dims.items())[:400])
        if axes:
            head.append('坐标范围：' + '；'.join(f'{k} {v}' for k, v in axes.items())[:400])
        head.append(f'变量（共 {len(variables)} 个）：' + '；'.join(
            f'{name}' + (f'[{_clean(getattr(var, "units", ""))}]' if getattr(var, 'units', '') else '')
            + (f' {getattr(var, "long_name", "")}' if getattr(var, 'long_name', '') else '')
            for name, var in variables[:30])[:900])
        items = [{'text': '\n'.join(head)[:chars], 'locator': 'nc-schema',
                  'evidence_kind': 'netcdf_schema', 'seg': 1}]
        kept = 0
        for name, variable in science:
            if kept >= max_variables:
                break
            rows = _sample_variable(variable, name, dataset)
            # A variable that is all fill values in the sample teaches nothing;
            # move on to the next candidate instead of spending a unit on it.
            lines = rows.splitlines()
            # Ignore the leading row index; only real cell values count.
            cells = '\n'.join(line.split(':', 1)[1] for line in lines if ':' in line)
            values = re.findall(r'-?\d+\.?\d*(?:e[+-]?\d+)?', cells)
            if not lines or len(values) < max(4, len(lines) // 6):
                continue
            kept += 1
            label = f'{name}' + (f' / {getattr(variable, "long_name", "")}' if getattr(variable, 'long_name', '') else '')
            units = _clean(getattr(variable, 'units', ''))
            body = [f'变量 {label}' + (f'（单位 {units}）' if units else ''),
                    f'维度 {" ".join(f"{d}={s}" for d, s in zip(variable.dimensions, variable.shape))}',
                    _fixed_axes(dataset, variable), '取样值（索引: 数值）', rows]
            body = [line for line in body if line]
            items.append({'text': '\n'.join(body)[:chars], 'locator': f'var-{name}',
                          'evidence_kind': 'netcdf_variable', 'seg': kept+1, 'variable': name})
        return items
    finally:
        dataset.close()
