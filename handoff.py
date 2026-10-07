"""Build technical evidence for another author; does not write a course report."""
from pathlib import Path
import hashlib
import html
import json
import zipfile
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from core import metrics
from qa import run as qa_run

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'results'
FIG=ROOT/'figures'
FIG.mkdir(exist_ok=True)


def read(name):
    return json.loads((OUT/name).read_text(encoding='utf-8'))


def savefig(name):
    plt.tight_layout()
    plt.savefig(FIG/(name+'.png'),dpi=180,bbox_inches='tight')
    plt.savefig(FIG/(name+'.pdf'),bbox_inches='tight')
    plt.close()


def main():
    qa_run()
    rows=read('validation.json'); final=read('final.json'); chosen=read('chosen.json')['selected']
    baseline=read('baseline.json'); hogenv=read('hog_environment.json')
    pred=np.load(OUT/'predictions.npz'); split=np.load(OUT/'split_indices.npz')
    data=np.load(ROOT/'data/mnist.npz')
    m=metrics(pred['truth'],pred['pred'],pred['ranking'])
    assert m==final['metrics'] and len(rows)==100
    assert not np.intersect1d(split['train'],split['validation']).size
    assert all(r['top1']<=r['top3']<=r['top5'] for r in rows)
    assert len(pred['truth'])==10000 and pred['neighbors'].max()<60000
    assert all(abs(r['agreement']-1)<1e-12 for r in baseline), 'Inspect baseline mismatches before accepting'
    best=sorted(rows,key=lambda r:(-r['top1'],r['dimensions'],r['k'],r['weights']!='uniform',r['feature'],r['metric']))[0]
    assert best==chosen
    # Recompute every final vote from saved neighbor distances and TRAINING labels.
    from core import vote
    pp,rr,ss=vote(data['y_train'][pred['neighbors']],pred['distances'],chosen['k'],chosen['metric'],chosen['weights'])
    np.testing.assert_array_equal(pp,pred['pred']); np.testing.assert_array_equal(rr,pred['ranking'])
    np.testing.assert_allclose(ss,pred['scores'])
    cm=np.asarray(m['cm'])
    plt.figure(figsize=(7,6)); plt.imshow(cm,cmap='Blues'); plt.colorbar(label='Count')
    for i in range(10):
        for j in range(10):
            if cm[i,j]: plt.text(j,i,str(cm[i,j]),ha='center',va='center',fontsize=7,color='white' if cm[i,j]>cm.max()/2 else 'black')
    plt.xticks(range(10)); plt.yticks(range(10)); plt.xlabel('Predicted'); plt.ylabel('True')
    plt.title('Full test confusion matrix (10,000 samples)'); savefig('01_confusion')
    plt.figure(figsize=(8,4.5))
    for feature in ['raw','pca20','pca50','pca100','hog']:
        subset=[r for r in rows if r['feature']==feature and r['metric']=='l2' and r['weights']=='uniform']
        plt.plot([r['k'] for r in subset],[100*r['top1'] for r in subset],marker='o',label=feature)
    plt.xlabel('Number of neighbors k'); plt.ylabel('Validation accuracy (%)'); plt.xticks([1,3,5,7,9]); plt.legend(); plt.grid(alpha=.2)
    plt.title('Same validation split: L2, uniform voting'); savefig('02_k_features')
    plt.figure(figsize=(7,4))
    variance=read('pca_validation.json')['cumulative_variance']
    plt.plot(range(1,len(variance)+1),np.asarray(variance)*100)
    for d in [20,50,100]: plt.scatter([d],[variance[d-1]*100]); plt.annotate(f'{d}: {variance[d-1]*100:.1f}%',(d,variance[d-1]*100),xytext=(-40,-16),textcoords='offset points')
    plt.xlabel('PCA dimensions'); plt.ylabel('Cumulative explained variance (%)'); plt.grid(alpha=.2); savefig('03_pca_variance')
    examples=read('error_examples.json')[:5]
    fig,axes=plt.subplots(len(examples),6,figsize=(9,1.7*len(examples)))
    for row,e in enumerate(examples):
        axes[row,0].imshow(data['x_test'][e['test_index']],cmap='gray')
        axes[row,0].set_title(f"Test #{e['test_index']}\n{e['truth']} -> {e['prediction']}",fontsize=9)
        for col,j in enumerate(e['neighbor_indices'][:5],start=1):
            axes[row,col].imshow(data['x_train'][j],cmap='gray')
            axes[row,col].set_title(f"Neighbor {col}\nlabel {data['y_train'][j]}",fontsize=9)
        for ax in axes[row]: ax.axis('off')
    savefig('04_error_neighbors')
    plt.figure(figsize=(8,4))
    names=['raw','pca20','pca50','pca100','hog']
    winners=[max((r for r in rows if r['feature']==f),key=lambda r:r['top1']) for f in names]
    bars=plt.bar(names,[r['top1']*100 for r in winners])
    for bar,r in zip(bars,winners): plt.text(bar.get_x()+bar.get_width()/2,bar.get_height()+.1,f"{r['top1']*100:.2f}%",ha='center')
    plt.ylim(85,100); plt.ylabel('Best validation accuracy (%)'); plt.title('Best of 20 settings per feature; validation only'); savefig('05_feature_comparison')
    off=cm.copy(); np.fill_diagonal(off,0)
    pairs=sorted([(int(off[i,j]),i,j) for i in range(10) for j in range(10) if i!=j],reverse=True)[:5]
    featurelines='\n'.join(f"  {r['feature']}: {r['top1']:.2%}，k={r['k']}，{r['metric']}，{r['weights']}" for r in winners)
    pairlines='；'.join(f'{i}→{j}: {n} 张' for n,i,j in pairs)
    q=read('qa.json'); bench=q['distance_benchmark']
    text=f'''MNIST KNN 技术总结——交给报告撰写者

一、交付边界
本次负责技术实现、真实实验、图表与 QA；本文件是技术交接，不是课程报告。
报告作者负责 Word/PDF、封面和排版，含封面总页数不得超过 10 页。
不得引用旧目录 mnist_knn 的旧指标、旧图表或旧网页冒充本次结果。

二、已完成的全部要求
1. sklearn baseline：原始像素、k=1/3/5/7/9，均匀及距离投票，与自行实现同一训练/验证集比较。
2. 自行实现：中心化+SVD PCA；L2平方距离向量化；L1距离；argpartition近邻选取；多数票与距离加权；Top-n；混淆矩阵；运行计时。
3. 可选特征：原始784维、PCA20/50/100维、HOG324维。
4. 可选参数：L1/L2、均匀/距离加权，5个k，共5×2×2×5=100组验证配置。
5. 错误解释：保存每个测试样本的近邻、分数、预测，并提供误分类样本与近邻拼图。
6. 可复现交付：源码、数据转换清单、固定划分索引、环境信息、原始JSON/CSV、PNG/PDF图、算法QA及数据一致性QA。

三、数据与实验设计
已有课程 KNN.zip 含标准形状的60,000张训练图、10,000张测试图。本次读取四个根目录IDX条目，避免重复条目；转换为NPZ，逐条目及归档哈希见data_manifest.json。
没有找到老师单独提供的mnist.npz；本次使用的是IDX无损转换版，不能把其来源描述成“收到老师的NPZ”。原远程提交来源只在旧README有记录，本次没有独立核验。
图像除以255归一化；从原始训练集中分层抽取12,000张训练、2,000张验证，种子42/43，索引无交集，剩余46,000张不参与选参。
PCA仅在12,000张训练子集拟合，验证集只transform；HOG是逐图固定特征，无跨样本拟合。
先按验证准确率选配置；同分依次比较更少维数、更小k、均匀优先、特征/距离字典顺序。chosen.json在测试前保存。
选定后使用完整60,000张训练样本，只评估一个最终配置的10,000张测试集；测试标签不用于选参。

四、主要结果（可直接引用，但须写明验证/测试）
各特征在20组候选中的最佳验证准确率：
{featurelines}

最终锁定配置：{chosen['feature']}，{chosen['dimensions']}维，{chosen['metric']}，k={chosen['k']}，{chosen['weights']}。
该配置验证准确率：{chosen['top1']:.2%}。
完整测试集：Top-1={m['top1']:.2%}（{int(np.trace(cm))}/10000）；Top-3={m['top3']:.2%}；Top-5={m['top5']:.2%}；Macro-F1={m['macro_f1']:.6f}。
较多的有向混淆：{pairlines}。
逐类precision/recall/F1与10×10混淆矩阵见final.json，全部测试预测见predictions.npz。
37. 10组实际MNIST sklearn对照的预测一致率均为100%。

五、技术解释与分析依据
KNN无需梯度训练；训练主要是保存特征和标签，主要开销在穷举距离搜索。
L2使用||q||²+||x||²−2q·x，截断浮点产生的负距离；按64条查询分批，避免一次分配10000×60000距离矩阵。
近邻距离平票优先训练行号小者；类别投票平票优先类别编号小者。两层规则不同，源码均显式处理。
距离加权使用1/d。L2检索得到的是d²，因此先开平方，不能误用1/d²。查询存在零距离邻居时仅这些邻居投票。
Top-n按类别分数降序、类别编号升序。少量邻居可能仅支持少数类别，Top-3/5可能补入零票类别，导致同分规则影响结果。不能把Top-5称为五个高置信候选，也不能与sklearn默认top_k_accuracy_score同分策略直接混用。
PCA降维压缩像素信息；HOG描述局部梯度方向。这里只能说HOG赢得本次小规模验证搜索，不能声称其在全部MNIST或所有训练规模上必然最好。

六、效率结果的正确口径
最终GPU近邻检索（含分批回传与确定性近邻选取）耗时{final['neighbor_s']:.6f}s；投票耗时{final['vote_s']:.6f}s。
HOG CPU特征提取：60,000训练图{hogenv['timings']['train_s']:.6f}s，10,000测试图{hogenv['timings']['test_s']:.6f}s。final.json的preparation_s为缓存选择等时间，不含这两项。
实际对照的CPU sklearn预测与自行实现共享k=9检索分别记录在baseline.json。自行实现对所有k复用检索，不能与每次单独调用sklearn的数字直接相除宣称加速倍数。
循环/向量化另做相同CPU数据核对：{bench['shape']['query']}查询×{bench['shape']['train']}训练×{bench['shape']['dimensions']}维、3次中位数，循环{bench['nested_loop']['median_s']:.8f}s，向量化{bench['vectorized']['median_s']:.8f}s。它是微基准，不能冒充完整MNIST加速比。
GPU、CPU计时只用于本机观察，当前为单次完整实验，未报告统计显著性，不代表普适性能排序。

七、QA与可追溯性
算法检查11项全部通过：独立距离公式、CPU/GPU近邻、两距离/两权重的sklearn预测和概率、SVD PCA子空间/解释方差、训练均值不变、距离边界平票、类别平票、零距离、Top-n单调性。
交付核对：100组配置齐全；划分无交集；锁定规则复算一致；全量10,000预测指标/混淆矩阵可重算；用保存的近邻重算所有投票一致；10组真实基线预测100%一致。
源文件哈希见source_hashes.json，数据哈希见data_manifest.json。源码修改后需重新运行相关QA并更新哈希。

八、给报告作者的使用建议
优先引用：02_k_features、01_confusion、04_error_neighbors；按篇幅选用03_pca_variance或05_feature_comparison。所有图均有PNG/PDF。
正文应解释方法、数据划分、验证选参、最终测试、错误分析及局限，不要把100组验证数据全堆进正文。
完整代码放在线仓库，正文放少量核心片段。最后一页预留源码链接、对应版本、运行方法；源码ZIP可作为附件。
源代码浏览页source_browser.html已生成，但本次还没有上传到公网。旧网站不代表本次版本；不能把本地路径写成公网链接。托管完成后再由报告作者填入新版地址及二维码。
本交接不生成Word/PDF课程报告、不填封面信息、不改写旧报告。
'''
    (ROOT/'技术总结_交给报告撰写者.txt').write_text(text,encoding='utf-8-sig')
    hashes={f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in ROOT.glob('*.py')}
    (OUT/'source_hashes.json').write_text(json.dumps(hashes,indent=2))
    audit={'passed':True,'configurations':len(rows),'test_predictions':len(pred['truth']),
           'baseline_comparisons':len(baseline),'all_baseline_predictions_agree':True,
           'recomputed_all_predictions_scores_and_metrics':True,'split_disjoint':True,'selection_rule_verified':True}
    (OUT/'delivery_qa.json').write_text(json.dumps(audit,indent=2))
    sections=[]
    for f in sorted(ROOT.glob('*.py')):
        sections.append('<h2>'+html.escape(f.name)+'</h2><pre>'+html.escape(f.read_text(encoding='utf-8'))+'</pre>')
    (ROOT/'source_browser.html').write_text('<!doctype html><meta charset="utf-8"><title>MNIST KNN source</title><style>body{max-width:1100px;margin:40px auto;font:16px system-ui;padding:20px}pre{overflow:auto;background:#f2f4f8;padding:20px;font:13px monospace}</style><h1>MNIST KNN — new technical implementation</h1><p>Source snapshot. Technical results and reproduction instructions are in the accompanying package.</p>'+''.join(sections),encoding='utf-8')
    with zipfile.ZipFile(ROOT/'技术交付包.zip','w',zipfile.ZIP_DEFLATED) as z:
        for f in ROOT.rglob('*'):
            if f.is_file() and '__pycache__' not in f.parts and f.suffix!='.zip' and f.name!='.gitignore':
                z.write(f,f.relative_to(ROOT))
    print('Handoff complete',audit,flush=True)

if __name__=='__main__':
    main()
