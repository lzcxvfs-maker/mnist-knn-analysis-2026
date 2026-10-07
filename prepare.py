"""Convert the existing course IDX archive to canonical mnist.npz without resampling."""
from pathlib import Path
import hashlib
import json
import struct
import zipfile
import numpy as np

root = Path(__file__).resolve().parent
(root/'data').mkdir(exist_ok=True)
(root/'results').mkdir(exist_ok=True)
source = root.parent/'mnist_knn/KNN.zip'
names = ['train-images.idx3-ubyte','train-labels.idx1-ubyte',
         't10k-images.idx3-ubyte','t10k-labels.idx1-ubyte']
keys = ['x_train','y_train','x_test','y_test']
arrays, hashes = {}, {}
with zipfile.ZipFile(source) as z:
    for key, name in zip(keys, names):
        blob = z.read(name)
        hashes[name] = hashlib.sha256(blob).hexdigest()
        magic, n = struct.unpack('>II',blob[:8])
        if magic == 2051:
            h,w = struct.unpack('>II',blob[8:16])
            assert (h,w)==(28,28) and len(blob)==16+n*h*w
            arrays[key] = np.frombuffer(blob,np.uint8,offset=16).reshape(n,h,w)
        else:
            assert magic==2049 and len(blob)==8+n
            arrays[key] = np.frombuffer(blob,np.uint8,offset=8)
np.savez_compressed(root/'data/mnist.npz', **arrays)
manifest = {'source':'Existing local course KNN.zip; original remote provenance is recorded in the old README but not independently reverified.',
    'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
    'members_sha256':hashes, 'shapes':{k:list(v.shape) for k,v in arrays.items()},
    'npz_sha256':hashlib.sha256((root/'data/mnist.npz').read_bytes()).hexdigest(),
    'note':'Converted IDX to NPZ, not a separately supplied teacher mnist.npz.'}
(root/'results/data_manifest.json').write_text(json.dumps(manifest,indent=2))
print(json.dumps(manifest,indent=2))
