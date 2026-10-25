"""Report power provenance; never infer measured watts from inference latency."""
import json
import math
import re
from pathlib import Path
import os
import time

REPORTS = Path(__file__).resolve().parent / 'reports'


class CpuEnergyMeter:
    """Windows Energy Meter CPU-package telemetry; no third-party driver needed.

    Native counter help specifies picowatt-hours for Energy and milliseconds
    for Time. Measure deltas, not the Power counter's previous sampling interval.
    """
    def __init__(self):
        self.query = None
        self.reason = None
        self.first = None

    def _check(self, status, operation):
        if status:
            raise OSError(f'{operation} failed (PDH 0x{status & 0xffffffff:08X})')

    def _read(self):
        self._check(self.pdh.PdhCollectQueryData(self.query), 'Read energy counters')
        values = []
        for counter in self.counters:
            raw = self.RawCounter()
            counter_type = self.ct.c_uint32()
            self._check(self.pdh.PdhGetRawCounterValue(counter, self.ct.byref(counter_type),
                                                      self.ct.byref(raw)), 'Read raw energy value')
            if raw.status not in (0, 1):
                raise OSError(f'Invalid energy counter status 0x{raw.status:08X}')
            values.append(int(raw.first))
        return {'energy_pwh': values[0], 'meter_time_ms': values[1],
                'monotonic_ns': time.perf_counter_ns()}

    def start(self):
        try:
            if os.name != 'nt':
                raise OSError('Windows Energy Meter is unavailable on this operating system')
            import ctypes as ct
            from ctypes import wintypes
            self.ct = ct
            class RawCounter(ct.Structure):
                _fields_ = [('status', ct.c_uint32), ('timestamp', wintypes.FILETIME),
                            ('first', ct.c_longlong), ('second', ct.c_longlong), ('multi', ct.c_uint32)]
            self.RawCounter = RawCounter
            self.pdh = ct.WinDLL('pdh.dll')
            self.pdh.PdhOpenQueryW.argtypes = [ct.c_wchar_p, ct.c_size_t, ct.POINTER(ct.c_void_p)]
            self.pdh.PdhAddEnglishCounterW.argtypes = [ct.c_void_p, ct.c_wchar_p, ct.c_size_t,
                                                      ct.POINTER(ct.c_void_p)]
            self.pdh.PdhCollectQueryData.argtypes = [ct.c_void_p]
            self.pdh.PdhGetRawCounterValue.argtypes = [ct.c_void_p, ct.POINTER(ct.c_uint32),
                                                      ct.POINTER(RawCounter)]
            self.pdh.PdhCloseQuery.argtypes = [ct.c_void_p]
            for name in ['PdhOpenQueryW', 'PdhAddEnglishCounterW', 'PdhCollectQueryData',
                         'PdhGetRawCounterValue', 'PdhCloseQuery']:
                getattr(self.pdh, name).restype = ct.c_long
            self.query = ct.c_void_p()
            self._check(self.pdh.PdhOpenQueryW(None, 0, ct.byref(self.query)), 'Open energy query')
            self.counters = []
            for name in ('Energy', 'Time'):
                counter = ct.c_void_p()
                path = '\\Energy Meter(RAPL_Package0_PKG)\\' + name
                self._check(self.pdh.PdhAddEnglishCounterW(self.query, path, 0, ct.byref(counter)),
                            f'Open CPU-package {name} counter')
                self.counters.append(counter)
            self.first = self._read()
        except (OSError, AttributeError) as exc:
            self.reason = str(exc)
            self.close()
        return self

    @staticmethod
    def calculate(first, last, count, scope):
        energy = (last['energy_pwh'] - first['energy_pwh']) * 3.6e-9
        seconds = (last['meter_time_ms'] - first['meter_time_ms']) / 1000
        wall = (last['monotonic_ns'] - first['monotonic_ns']) / 1e9
        if energy <= 0 or seconds <= 0 or not math.isfinite(energy):
            raise ValueError('Energy counter did not advance or reset; no usable measurement')
        if seconds < 0.1:
            raise ValueError('Measurement shorter than 0.1 s; use a repeated historical energy benchmark')
        if wall <= 0 or abs(seconds - wall) > max(0.1, wall * .2):
            raise ValueError('Energy meter and benchmark clocks disagree; measurement rejected')
        return dict(status='measured', source='Windows Energy Meter / RAPL_Package0_PKG',
                    boundary='CPU package, including other system activity; not whole-PC or process-only energy',
                    workload_scope=scope, elapsed_seconds=seconds, wall_elapsed_seconds=wall,
                    energy_j=energy, average_power_w=energy / seconds, completed_decisions=count,
                    energy_per_decision_j=energy / count if count > 0 else None,
                    efficiency_decisions_per_j=count / energy if count > 0 else None,
                    start_counter=first, end_counter=last,
                    raw_energy_unit='picowatt-hour', joules_per_raw_energy_unit=3.6e-9,
                    raw_time_unit='millisecond',
                    qualification='Hardware-reported RAPL telemetry; not an external calibrated power-meter reading')

    def stop(self, count, scope):
        try:
            if self.reason or self.first is None:
                raise ValueError(self.reason or 'Energy measurement was not started')
            return self.calculate(self.first, self._read(), count, scope)
        except (OSError, ValueError) as exc:
            return dict(status='unavailable', reason=str(exc), completed_decisions=count, workload_scope=scope)
        finally:
            self.close()

    def close(self):
        if self.query:
            self.pdh.PdhCloseQuery(self.query)
            self.query = None


def start_cpu_energy(args=None):
    return CpuEnergyMeter().start() if getattr(args, 'average_power_w', None) is None else None


def save_cpu_energy(path, measurement):
    if measurement is not None:
        Path(path).write_text(json.dumps(measurement, indent=2), encoding='utf-8')


def positive_watts(value):
    import argparse
    try:
        watts = float(value)
    except (TypeError, ValueError):
        raise argparse.ArgumentTypeError('Power must be a finite positive number in watts')
    if not math.isfinite(watts) or watts <= 0:
        raise argparse.ArgumentTypeError('Power must be a finite positive number in watts')
    return watts


def add_power_arguments(parser):
    parser.add_argument('--average-power-w', type=positive_watts,
                        help='Externally obtained average watts over the same reported run interval')
    parser.add_argument('--power-basis', choices=['measured', 'estimated'],
                        help='Required with --average-power-w; describes the supplied power value')
    parser.add_argument('--power-source', help='Required with --average-power-w; meter/tool and hardware boundary')


def validate_power_arguments(parser, args):
    supplied = (args.average_power_w is not None, args.power_basis is not None,
                args.power_source is not None)
    if any(supplied) and (not all(supplied) or not args.power_source.strip()):
        parser.error('Supply --average-power-w, --power-basis and --power-source together')


def power_summary(backend, count, elapsed_seconds, args=None, reports=REPORTS, measurement=None):
    lines = ['POWER / ENERGY EFFICIENCY']
    watts = getattr(args, 'average_power_w', None)
    if watts is not None:
        watts = positive_watts(watts)
        basis = args.power_basis.upper()
        lines += [f'{basis} average power (externally supplied): {watts:.6f} W',
                  f'Source / hardware boundary: {args.power_source}',
                  'Power must cover the same interval below; no automatic power sensor was used.',
                  f'Power interval: {elapsed_seconds:.6f} s; completed decisions: {count}']
        if elapsed_seconds > 0 and count > 0:
            energy = watts * elapsed_seconds
            lines += [f'{basis} run energy: {energy:.6f} J',
                      f'{basis} energy per decision: {energy / count:.9f} J',
                      f'{basis} power efficiency: {count / energy:.6f} decisions/s/W (= decisions/J)']
        else:
            lines.append('Per-decision energy / efficiency: unavailable (no completed decisions or elapsed time)')
    elif backend == 'cpu' and measurement and measurement.get('status') == 'measured':
        lines += [f'CPU package average power (RAPL telemetry): {measurement["average_power_w"]:.6f} W',
                  f'CPU package energy: {measurement["energy_j"]:.6f} J',
                  f'Measured energy interval: {measurement["elapsed_seconds"]:.3f} s',
                  f'Completed decisions in energy interval: {measurement["completed_decisions"]}',
                  f'Workload: {measurement["workload_scope"]}']
        if measurement['completed_decisions'] > 0:
            lines += [f'Energy per decision: {measurement["energy_per_decision_j"] * 1000:.6f} mJ',
                      f'CPU power efficiency: {measurement["efficiency_decisions_per_j"]:.6f} decisions/s/W (= decisions/J)']
        else:
            lines.append('Per-decision energy / efficiency: unavailable (no completed decisions)')
        lines += [measurement['boundary'], measurement['qualification'],
                  'Not directly comparable to FPGA on-chip power from a different workload.']
    else:
        lines += [f'{backend.upper()} measured run power: unavailable (no matching power measurement)',
                  f'{backend.upper()} measured run energy / efficiency: unavailable',
                  'Elapsed time and CPU utilization cannot establish watts.']
        if backend == 'cpu' and measurement:
            lines.append(f'CPU sensor status: {measurement.get("reason", "unavailable")}')
    if backend == 'fpga':
        try:
            text = (reports / 'power_uart_activity.rpt').read_text(encoding='utf-8')
            match = re.search(r'\|\s*Total On-Chip Power \(W\)\s*\|\s*([\d.]+)', text)
            if match is None:
                raise ValueError('Total on-chip power missing')
            estimate = positive_watts(match.group(1))
            if watts is None:
                lines += ['', 'FPGA LIVE-RUN ENERGY ESTIMATE',
                          f'Estimated FPGA on-chip average power: {estimate:.3f} W',
                          'Source: saved Vivado UART-activity report; constant-power assumption over this run',
                          'Scope: FPGA on-chip only; excludes host and whole-board power',
                          f'Live run duration: {elapsed_seconds:.6f} s; completed decisions: {count}']
                if elapsed_seconds > 0:
                    live_energy = estimate * elapsed_seconds
                    lines.append(f'Estimated live-run energy (includes feed waits): {live_energy:.6f} J')
                    if count > 0:
                        lines += [f'Estimated live energy per decision: {live_energy / count:.6f} J',
                                  f'Estimated live energy efficiency: {count / live_energy:.9f} decisions/J (= decisions/s/W)']
                    else:
                        lines.append('Estimated live energy efficiency: unavailable (no completed decisions)')
                lines.append('Estimate only: not a live board-power sensor reading; actual idle/activity power may differ.')
            summary = json.loads((reports / 'performance_summary.json').read_text(encoding='utf-8'))
            transactions = summary['uart_simulation']
            duration = sum(float(row['total_ns']) for row in transactions) / len(transactions) / 1e9
            if not math.isfinite(duration) or duration <= 0:
                raise ValueError('Invalid reference UART duration')
            energy = estimate * duration
            lines += ['', 'SAVED FPGA UART BENCHMARK ESTIMATE (separate from this run)',
                      'Source: Vivado UART-activity report + simulated UART transactions',
                      'Scope: FPGA on-chip only; excludes host and whole-board power',
                      f'Estimated reference power: {estimate:.3f} W',
                      f'Reference transaction duration: {duration * 1000:.6f} ms',
                      f'Estimated reference energy: {energy * 1000:.6f} mJ/inference',
                      f'Estimated reference efficiency: {1 / energy:.3f} inferences/s/W',
                      'Not measured board power, live-run efficiency, or proof of CPU-FPGA energy savings.']
        except (OSError, ValueError, KeyError, TypeError, ZeroDivisionError) as exc:
            lines.append(f'Saved FPGA benchmark power estimate: unavailable ({exc})')
    return lines
