"""Reproducible validation scan and locked, one-configuration final evaluation."""
from pathlib import Path
import csv
import hashlib
import json
import platform
import sys
import time
import numpy as np
import torch
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from core import PCA, neighbors, vote, metrics

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'results'
KS=(1,3,5,7,9)


def save(name,value):
    (OUT/name).write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8')


def compact(m):
    return {k:m[k] for k in ['top1','top3','top5','macro_f1']}


def main():
    torch.set_num_threads(8)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.set_float32_matmul_precision('highest')
    from qa import run
    run()
    data=np.load(ROOT/'data/mnist.npz')
    x=data['x_train'].reshape(-1,784).astype(np.float32)/255
    y=data['y_train'].astype(int)
    # Reproducible 12k/2k stratified subsets of TRAINING data only.
    tr,rest=train_test_split(np.arange(len(y)),train_size=12000,stratify=y,random_state=42)
    va,_=train_test_split(rest,train_size=2000,stratify=y[rest],random_state=43)
    assert not np.intersect1d(tr,va).size
    np.savez_compressed(OUT/'split_indices.npz',train=tr,validation=va)
    save('environment.json',{'python':sys.version,'numpy':np.__version__,'torch':torch.__version__,
        'sklearn':sklearn.__version__,'platform':platform.platform(),
        'gpu':torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        'cuda':torch.version.cuda,'tf32':False,'seed':[42,43],
        'validation_train_count':len(tr),'validation_count':len(va),
        'train_classes':np.bincount(y[tr],minlength=10).tolist(),
        'validation_classes':np.bincount(y[va],minlength=10).tolist()})
    print('Fitting validation PCA (center + SVD)',flush=True)
    t=time.perf_counter(); pca=PCA(100).fit(x[tr]); fit_s=time.perf_counter()-t
    t=time.perf_counter(); a=pca.transform(x[tr]); b=pca.transform(x[va]); transform_s=time.perf_counter()-t
    save('pca_validation.json',{'fit_s':fit_s,'transform_s':transform_s,'cumulative_variance':np.cumsum(pca.ratio_).tolist()})
    hog=np.load(ROOT/'data/hog.npz')
    features={'raw':(x[tr],x[va]),'pca20':(a[:,:20],b[:,:20]),
              'pca50':(a[:,:50],b[:,:50]),'pca100':(a,b),'hog':(hog['train'][tr],hog['train'][va])}
    rows=[]
    for feature,(xt,xv) in features.items():
        for metric in ['l2','l1']:
            t=time.perf_counter(); ix,ds=neighbors(xt,xv,9,metric,'torch',64); search_s=time.perf_counter()-t
            for weights in ['uniform','distance']:
                for k in KS:
                    t=time.perf_counter(); pred,rank,_=vote(y[tr][ix],ds,k,metric,weights); vote_s=time.perf_counter()-t
                    m=metrics(y[va],pred,rank)
                    rows.append({'feature':feature,'dimensions':xt.shape[1],'metric':metric,'weights':weights,'k':k,
                        **compact(m),'neighbor_s':search_s,'vote_s':vote_s})
            print(feature,metric,'done',round(search_s,3),'s',flush=True)
            save('validation.json',rows)
    # Validation top1 only; explicit deterministic tie-breaker favors fewer dimensions,
    # smaller k, uniform weighting, then lexicographic feature/distance.
    best=sorted(rows,key=lambda r:(-r['top1'],r['dimensions'],r['k'],0 if r['weights']=='uniform' else 1,r['feature'],r['metric']))[0]
    save('chosen.json',{'chosen':best,'rule':'validation top1 -> min dimensions -> min k -> uniform -> feature/metric alpha'})
    with open(OUT/'validation.csv','w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0].keys()))
        writer.writeheader(); writer.writerows(rows)
    # Real baseline check against sklearn on exactly the same MNIST validation set.
    baseline=[]
    for weights in ['uniform','distance']:
        for k in KS:
            t=time.perf_counter()
            sk=KNeighborsClassifier(n_neighbors=k,metric='euclidean',weights=weights,algorithm='brute').fit(x[tr],y[tr])
            sk_pred=sk.predict(x[va]); sk_s=time.perf_counter()-t
            own_pred,_,_=vote(y[tr][features['raw'][1].shape[0] and neighbors(x[tr],x[va],k,'l2','torch',64)[0]],
                              neighbors(x[tr],x[va],k,'l2','torch',64)[1],k,'l2',weights)
            # Recompute with our unified neighbor search for identical comparison.
            ix,ds=neighbors(x[tr],x[va],k,'l2','torch',64)
            own_pred,_,_=vote(y[tr][ix],ds,k,'l2',weights)
            agree=float(np.mean(own_pred==sk_pred))
            baseline.append({'k':k,'weights':weights,'agreement':agree,'sklearn_s':sk_s})
    save('baseline.json',baseline)
    # One-shot final evaluation on full 10k test set with the locked configuration only.
    cfg=best
    print('Final evaluation with locked configuration:',cfg['feature'],cfg['metric'],cfg['weights'],cfg['k'],flush=True)
    if cfg['feature']=='raw':
        xtrain,xtest=x,data['x_test'].reshape(-1,784).astype(np.float32)/255
    elif cfg['feature'].startswith('pca'):
        pdim=int(cfg['feature'][3:])
        fpca=PCA(pdim).fit(x)
        xtrain,xtest=fpca.transform(x),fpca.transform(data['x_test'].reshape(-1,784).astype(np.float32)/255)
    elif cfg['feature']=='hog':
        xtrain,xtest=hog['train'],hog['test']
    ytest=data['y_test'].astype(int)
    t=time.perf_counter(); ix,ds=neighbors(xtrain,xtest,cfg['k'],cfg['metric'],'torch',64); search_s=time.perf_counter()-t
    t=time.perf_counter(); pred,rank,scores=vote(y[xtrain][ix] if hasattr(y,'__getitem__') else y[ix],ds,cfg['k'],cfg['metric'],cfg['weights'])
    # Correct indexing for training labels:
    pred,rank,scores=vote(y[ix],ds,cfg['k'],cfg['metric'],cfg['weights']); vote_s=time.perf_counter()-t
    fin=metrics(ytest,pred,rank)
    save('final.json',{'config':cfg,'train_count':len(x),'test_count':len(ytest),'metrics':fin,
                       'preparation_s':0.0,'neighbor_s':search_s,'vote_s':vote_s,
                       'note':'HOG extraction time is recorded separately in hog_environment.json; preparation here is cache loading/selection only.',
                       'topn_tie_rule':'score descending, class ID ascending; unvoted zero-score classes can enter Top3/Top5. These are not calibrated probabilities.'})
    np.savez_compressed(OUT/'predictions.npz',truth=ytest,prediction=pred,rank=rank,scores=scores,neighbors=ix,distances=ds)
    print('Done. Test top1:',fin['top1'],'top3:',fin['top3'],'top5:',fin['top5'],flush=True)

if __name__=='__main__':
    main()
