import sys
from .DataProcess.DataTransfer import DataTransfer
from .Workflow import LFM
from .Workflow import BPR
from .Workflow import FM
from .Workflow import KNN



def choose():
    arg = sys.argv[1]
    if arg == 'lfm':
        LFM()
    elif arg == 'fm':
        FM()
    elif arg == 'bpr':
        BPR()
    elif arg == 'knn':
        KNN()
    sys.exit()

if __name__ == '__main__':
    # Transfer().process()
    choose()

