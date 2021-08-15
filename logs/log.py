from datetime import datetime
from pathlib import Path
from os.path import join
import pickle


def logfile(obj, name):
    date = datetime.now()
    path = join('logs', date.strftime('%Y'), date.strftime('%B'), date.strftime('%d'))
    Path(path).mkdir(parents=True, exist_ok=True)
    filename = name + '_' + date.strftime('%X').replace(':', '-')
    file = open(join(path, filename), 'wb')
    pickle.dump(obj, file, protocol=pickle.HIGHEST_PROTOCOL)
    file.close()
