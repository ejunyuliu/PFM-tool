import numpy as np

from PFM import data, spang
from PFM.microscope import multi

spang1 = spang.Spang(f=np.zeros((100, 100, 100, 15), dtype=np.float32))
data1 = data.Data(g=np.zeros((100, 100, 100, 4, 2), dtype=np.float32))

diSPIM = multi.MultiMicroscope(spang1, data1)

diSPIM.calc_H()
