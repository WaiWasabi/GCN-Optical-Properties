from datetime import datetime
from pathlib import Path
from os.path import join
import pickle


def logfile(obj, name):
    date = datetime.now()
    path = join(date.strftime('%Y'), date.strftime('%B'), date.strftime('%d'))
    Path(path).mkdir(parents=True, exist_ok=True)
    filename = name + '_' + date.strftime('%X').replace(':', '-')
    with open(join(path, filename), 'wb') as file:
        pickle.dump(obj, file)
