# -*- coding: utf-8 -*-
"""

Пути задаются переменными окружения (значения по умолчанию соответствуют Google Colab + Google Drive):
    SEPSIS_RAW_DIR  — папка с исходными таблицами MIMIC-IV и производными таблицами mimic-code
                      (*.csv.gz или *.csv; по умолчанию /content/drive/MyDrive)
    SEPSIS_WORK_DIR — рабочая папка для промежуточных данных, моделей, результатов и журналов
                      (по умолчанию $SEPSIS_RAW_DIR/sepsis_v3)

Пример запуска в Colab:
    from google.colab import drive; drive.mount('/content/drive')
    !python /content/sepsis-mimic4-temporal/src/stage0_sepsis3.py

Каждый этап:
  * читает только исходные таблицы (SEPSIS_RAW_DIR) и выходы предыдущих этапов (SEPSIS_WORK_DIR/data);
  * записывает результаты в data/, models/, results/, JSON-лог в logs/ и полный результат запуска в run_logs/;
  * останавливается с ошибкой, если не пройдена контрольная проверка (Stage.check).
"""
import os
import sys
import json
import time
import platform
from datetime import datetime

DRIVE = os.environ.get('SEPSIS_RAW_DIR', '/content/drive/MyDrive')
ROOT = os.environ.get('SEPSIS_WORK_DIR', f'{DRIVE}/sepsis_v3')
DATA = f'{ROOT}/data'
LOGS = f'{ROOT}/logs'
RUN_LOGS = f'{ROOT}/run_logs'
SEED = 20260926

for _d in (DATA, LOGS, RUN_LOGS):
    os.makedirs(_d, exist_ok=True)


def raw(name):
    """Путь к исходной таблице в SEPSIS_RAW_DIR: пробует .csv.gz, .csv, как есть."""
    for cand in (f'{DRIVE}/{name}.csv.gz', f'{DRIVE}/{name}.csv', f'{DRIVE}/{name}'):
        if os.path.isfile(cand):
            return cand
    raise FileNotFoundError(f'Не найдена исходная таблица: {name} (искали в {DRIVE})')


def is_gzip(path):
    with open(path, 'rb') as f:
        return f.read(2) == b'\x1f\x8b'


def read_csv_any(path, **kw):
    """pandas.read_csv со сжатием, определённым по содержимому файла, а не по расширению."""
    import pandas as pd
    return pd.read_csv(path, compression='gzip' if is_gzip(path) else None, low_memory=False, **kw)


class _Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, s):
        for st in self.streams:
            st.write(s)
            st.flush()

    def flush(self):
        for st in self.streams:
            st.flush()


class Stage:
    """Контекст этапа: дублирует вывод в run_logs, собирает метрики в JSON-лог, проверяет контрольные условия.

    with Stage('stage0_sepsis3') as st:
        st.log('n_episodes', n)
        st.check(n == n_ref, 'число эпизодов совпадает с эталоном')
    """

    def __init__(self, name):
        self.name = name
        self.t0 = time.time()
        stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.run_log_path = f'{RUN_LOGS}/{name}_{stamp}.log'
        self.json_path = f'{LOGS}/{name}.json'
        self.metrics = {}
        self.checks = []

    def __enter__(self):
        self._fh = open(self.run_log_path, 'w', encoding='utf-8')
        self._stdout, self._stderr = sys.stdout, sys.stderr
        sys.stdout = _Tee(self._stdout, self._fh)
        sys.stderr = _Tee(self._stderr, self._fh)
        print(f'# {self.name} | запуск {datetime.now().isoformat(timespec="seconds")}')
        print(f'# python {platform.python_version()} | {self._versions()}')
        return self

    def _versions(self):
        out = []
        for mod in ('pandas', 'numpy', 'pyarrow', 'duckdb', 'lightgbm', 'sklearn', 'optuna', 'scipy'):
            try:
                out.append(f'{mod} {__import__(mod).__version__}')
            except Exception:
                pass
        return ', '.join(out)

    def section(self, title):
        print('\n' + '=' * 90 + f'\n{title}  [{(time.time() - self.t0) / 60:.1f} мин]\n' + '=' * 90)

    def log(self, key, value):
        """Сохраняет метрику в JSON-лог (и печатает её)."""
        self.metrics[key] = value
        print(f'  {key}: {value}')

    def check(self, condition, description):
        """Контрольная проверка: при невыполнении этап останавливается с ошибкой."""
        ok = bool(condition)
        self.checks.append({'check': description, 'passed': ok})
        print(f'  [{"OK" if ok else "ОШИБКА"}] {description}')
        if not ok:
            raise RuntimeError(f'Контрольная проверка не пройдена: {description}')

    def __exit__(self, exc_type, exc, tb):
        status = 'ok' if exc_type is None else f'error: {exc_type.__name__}: {exc}'
        payload = {'stage': self.name, 'finished': datetime.now().isoformat(timespec='seconds'),
                   'minutes': round((time.time() - self.t0) / 60, 2), 'status': status,
                   'versions': self._versions(), 'seed': SEED,
                   'metrics': self.metrics, 'checks': self.checks, 'run_log': self.run_log_path}
        with open(self.json_path, 'w', encoding='utf-8') as f:
            json.dump(payload, f, ensure_ascii=False, indent=2, default=str)
        print(f'\n# статус: {status} | {payload["minutes"]} мин | лог: {self.json_path}')
        sys.stdout, sys.stderr = self._stdout, self._stderr
        self._fh.close()
        return False
