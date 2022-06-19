
import pandas as pd
import numpy as np

import os
os.environ['DEBUG'] = 'True'
os.environ["RMBASE_FILE_PYTHON"] = "/Users/fred/work/cfglocal/built/base.py"
print(os.environ['RMBASE_FILE_PYTHON'])
import logging
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
logging.debug("test")



from organizer import Organizer
from base.timer import Timer
t = Timer()
t.start()
org = Organizer()
org.init_transformers()

t.stop(org.data_source.df_grouped.shape[0])
data_source = org.data_source

org.train_predictors()


for c in data_source.df_grouped.columns:
  if c.startswith('sqft'):
    print(c)



# simple steps
from organizer import Organizer
from base.timer import Timer
org = Organizer()
org.init_transformers()
org.train_predictors()