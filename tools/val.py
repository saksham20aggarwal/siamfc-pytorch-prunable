from __future__ import absolute_import
import sys
# sys.path.append('/kaggle/working/siamfc-pytorch-prunable')
import os
from got10k.experiments import *
# from got10k.datasets import *

from siamfc import TrackerSiamFC


if __name__ == '__main__':
    net_path = '/kaggle/input/notebook221b0fd605/siamfc-pytorch-prunable/pretrained/siamfc_alexnet_e25.pth'
    tracker = TrackerSiamFC(net_path=net_path)

    root_dir = os.path.expanduser('/kaggle/input/got10k')
    e = ExperimentGOT10k(root_dir, subset='val', result_dir='results/', report_dir='reports/')
    e.run(tracker)
    e.report([tracker.name])
