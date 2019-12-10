
from collections import namedtuple
import os
import sys
import fileinput
import re
abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)

# with open("env.sh", 'w+') as o:
#     for i in range(1,57,1):
#         o.write("sbatch stencil_%d.job\n" % (i))

with fileinput.input(files=('file1.txt', 'file2.txt'), inplace=True) as f:
    for line in f:
        print( re.sub(y, z, line) )