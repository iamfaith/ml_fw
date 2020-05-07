from __future__ import absolute_import
from __future__ import division
from __future__ import print_function  # py2 use py3 print


import sys, os
abs_path = os.path.abspath('./')
sys.path.append(abs_path)
from init.load import init_cfg
cfg = init_cfg(os.path.join(abs_path + '/config', 'example.policy.py'))


def rename_project(name):
    cfg['project'] = name
    

from datetime import datetime

rename_project(str(datetime.now()))

# print(cfg)
# os.mkdir(str(datetime.now()))