"""Run with a Python that has scikit-image; cache reusable HOG features."""
from pathlib import Path
import json
import time
import numpy as np
import skimage
from skimage.feature import hog

root = Path(__file__).resolve().parent
data = np.load(root/'data/mnist.npz')
result = {}
timings = {}
for split in ('train', 'test'):
    t = time.perf_counter()
    result[split] = np.asarray([hog(img, orientations=9, pixels_per_cell=(7, 7),
        cells_per_block=(2, 2), block_norm='L2-Hys', feature_vector=True)
        for img in data['x_'+split]], dtype=np.float32)
    timings[split+'_s'] = time.perf_counter()-t
np.savez_compressed(root/'data/hog.npz', **result)
(root/'results/hog_environment.json').write_text(json.dumps({
    'skimage':skimage.__version__, 'numpy':np.__version__, 'timings':timings,
    'parameters':{'orientations':9,'pixels_per_cell':[7,7], 'cells_per_block':[2,2],
                  'block_norm':'L2-Hys','dimensions':324}}, indent=2))
print('HOG ready', timings, flush=True)
