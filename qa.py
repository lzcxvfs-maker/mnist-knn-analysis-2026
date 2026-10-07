"""Meaningful correctness checks against independent implementations."""
import json
import time
from pathlib import Path
import numpy as np
from sklearn.decomposition import PCA as SKPCA
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from core import PCA, MyKNN, distances, select_neighbors, neighbors, vote, metrics


def run():
    rng = np.random.default_rng(123)
    x = rng.normal(size=(240, 24)).astype(np.float32)
    q = rng.normal(size=(40, 24)).astype(np.float32)
    y = rng.integers(0, 10, len(x))
    checks = []
    for metric, skmetric in [('l2','euclidean'), ('l1','manhattan')]:
        explicit = np.asarray([[(np.sum((a-b)**2) if metric=='l2' else np.sum(abs(a-b)))
                               for b in x] for a in q])
        np.testing.assert_allclose(distances(q,x,metric), explicit, atol=2e-5, rtol=2e-6)
        ni, nd = neighbors(x,q,metric=metric)
        gi, gd = neighbors(x,q,metric=metric,engine='torch')
        np.testing.assert_array_equal(ni,gi)
        np.testing.assert_allclose(nd,gd,atol=2e-5,rtol=2e-6)
        for weights in ['uniform','distance']:
            model = KNeighborsClassifier(n_neighbors=5,metric=skmetric,weights=weights,algorithm='brute').fit(x,y)
            pred,rank,scores = vote(y[ni],nd,5,metric,weights)
            np.testing.assert_array_equal(pred, model.predict(q))
            expected = np.zeros((len(q),10))
            expected[:,model.classes_] = model.predict_proba(q)
            np.testing.assert_allclose(scores/scores.sum(1)[:,None], expected,atol=2e-6)
            np.testing.assert_allclose(MyKNN(5,metric,weights).fit(x,y).predict_proba(q),expected,atol=2e-6)
            np.testing.assert_array_equal(metrics(y[:len(q)],pred,rank)['cm'],confusion_matrix(y[:len(q)],pred,labels=range(10)))
            checks.append(metric+'_'+weights+'_sklearn_prediction_and_probability')
        checks.append(metric+'_distance_and_cpu_gpu_neighbors')
    own = PCA(8).fit(x)
    ref = SKPCA(8,svd_solver='full').fit(x.astype(np.float64))
    np.testing.assert_allclose(own.basis_@own.basis_.T,ref.components_.T@ref.components_,atol=1e-10)
    np.testing.assert_allclose(own.ratio_,ref.explained_variance_ratio_,atol=1e-12)
    before = own.mean_.copy()
    own.transform(q+100)
    np.testing.assert_array_equal(before,own.mean_)
    checks.append('pca_svd_subspace_variance_and_training_only_mean')
    ix,ds = select_neighbors(np.asarray([[1.,0.,1.,1.],[0.,0.,0.,0.]]),2)
    np.testing.assert_array_equal(ix,[[1,0],[0,1]])
    p,r,s = vote(np.asarray([[2,1,2,1],[7,2,1,0]]),np.asarray([[1.,1.,2.,2.],[0.,0.,1.,4.]]),4,weights='distance')
    assert p.tolist()==[1,2] and s[1,7]==s[1,2] and s[1,1]==0
    truth=np.asarray([1,7])
    m=metrics(truth,p,r)
    assert m['top1']<=m['top3']<=m['top5']
    checks.extend(['kth_distance_boundary_ties','class_vote_ties','zero_distance_weights','topn_monotonicity'])
    # Same CPU data, three repeats; these are distance-kernel timings, not GPU speedups.
    a=q[:32]; b=x
    times={}
    answers={}
    for method in ['nested_loop','vectorized']:
        runs=[]
        for _ in range(3):
            t=time.perf_counter()
            result=(np.asarray([[np.sum((i-j)**2) for j in b] for i in a])
                    if method=='nested_loop' else distances(a,b))
            runs.append(time.perf_counter()-t)
        times[method]={'seconds':runs,'median_s':float(np.median(runs))}
        answers[method]=result
    np.testing.assert_allclose(answers['nested_loop'],answers['vectorized'],atol=2e-5,rtol=2e-6)
    output={'passed':True,'checks':checks,'distance_benchmark':times,
            'benchmark_shape':{'query':32,'train':240,'dimensions':24},
            'note':'Small controlled kernel microbenchmark; not full MNIST runtime.'}
    (Path(__file__).parent/'results/qa.json').write_text(json.dumps(output,indent=2))
    print('QA passed:',len(checks),flush=True)
    return output

if __name__=='__main__':
    run()
