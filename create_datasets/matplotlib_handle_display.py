#!/usr/bin/env python3.5
"""Import matplotlib and use Agg backend if $DISPLAY not detected.

This should be imported before importing any other matplotlib module (i.e. pyplot).
"""

import os
import matplotlib as mpl
if os.name == 'nt':  # 'nt' refers to Windows
    display_ok = os.system('py -c "import matplotlib.pyplot as plt;plt.figure()"')  # Windows
else:
    display_ok = os.system('python3 -c "import matplotlib.pyplot as plt;plt.figure()"')  # Linux
# This allows mpl to run with no DISPLAY defined
if display_ok != 0:
    print("$DISPLAY not detected, matplotlib set to use 'Agg' backend")
    mpl.use('Agg')
