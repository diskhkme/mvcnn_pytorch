import numpy as np
from tools.augmentation import RandomRemoval

img = np.random.randn(3,3,3)

aug = RandomRemoval()
aug(img)