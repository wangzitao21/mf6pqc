import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from modflow_models import transport_model

import os
import numpy as np
import traceback
import phreeqcrm
import modflowapi
import sys

# 确保能导入 mf6pqc 和 transport_model
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)
from mf6pqc.mf6pqc import mf6pqc
from modflow_models import transport_model


head, concentration_data = transport_model(
    sim_ws='./simulation/mycase',
    species_list=["Ca", "Mg", "Cl"],
    perlen=1095,
    nstp=120,
    initial_conc=np.ones(120000) * 0.05,
    bc=[0.1, 0.1, 0.1],
    porosity=0.30,
    K11=1.0,
    initial_density=1000.0,
    initial_head=100.0
)

plt.imshow(head.reshape(-1, 100, 400)[-1])
plt.show()