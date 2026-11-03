import pandas as pd, numpy as np, json
m=pd.read_pickle('master_v4_pseudonymised.pkl')
e=m[m.eligible][['child_id','month_index','puskesmas','label']].copy()
MI=lambda y,mo:(y-2022)*12+mo-1
out={}; rows=[]
def pairs(h):
    a=e.rename(columns={'month_index':'t','label':'status_t'}); b=e[['child_id','month_index','label']].rename(columns={'month_index':'th','label':'y'})
    a['th']=a.t+h; p=a.merge(b,on=['child_id','th']); return p
def block(p,h,design):
    if design=='primary':
        dev=(p.th>=MI(2023,1))&(p.th<=MI(2024,12)); val=(p.t>=MI(2025,1))&(p.th<=MI(2025,12)); test=p.t>=MI(2026,1)
    else: # h12 secondary
        dev=(p.th<=MI(2024,12))&(p.th>=MI(2023,1)); val=pd.Series(False,index=p.index); test=(p.t>=MI(2025,1))
    b=np.select([dev,val,test],['development','threshold_learning','test'],'excluded')
    p=p.assign(block=b); p.loc[(p.th<MI(2023,1)),'block']='target_in_2022'
    return p
summ=[]
for h,des in [(1,'primary'),(3,'primary'),(6,'primary'),(12,'secondary')]:
    p=block(pairs(h),h,des)
    for bl,g in p.groupby('block'):
        vc=g.y.value_counts(normalize=True)
        summ.append(dict(h=h,design=des,block=bl,pairs=len(g),children=g.child_id.nunique(),
            normal=round(100*vc.get('normal',0),1),stunted=round(100*vc.get('stunted',0),1),severe=round(100*vc.get('severely stunted',0),1),
            n_severe=int((g.y=='severely stunted').sum()),status_change=round(100*(g.status_t!=g.y).mean(),1),
            anchor_months=f"{g.t.min()}..{g.t.max()}"))
    dev=set(p[p.block=='development'].child_id); te=p[p.block=='test']
    out[h]={'test_children_seen_in_development':int(te.child_id.isin(dev).groupby(te.child_id).first().sum()),'test_children':int(te.child_id.nunique())}
    if des=='primary':
        out[h]['test_by_puskesmas']=te.groupby('puskesmas').agg(pairs=('y','size'),children=('child_id','nunique'),severe=('y',lambda s:(s=='severely stunted').sum())).to_dict('index')
s=pd.DataFrame(summ); pd.set_option('display.width',250); print(s.to_string(index=False))
for h in out: print(h, {k:v for k,v in out[h].items() if k!='test_by_puskesmas'})
print(pd.DataFrame(out[6]['test_by_puskesmas']).T.to_string())
print(pd.DataFrame(out[1]['test_by_puskesmas']).T[['pairs','severe']].T.to_string())
# attrition test period h=1,3,6: eligible anchors with no target
for h in (1,3,6):
    a=e[(e.month_index>=MI(2026,1))&(e.month_index+h<=MI(2026,9))]
    has=set(zip(e.child_id,e.month_index)); a=a.assign(has=[(c,t+h) in has for c,t in zip(a.child_id,a.month_index)])
    print('attrition h',h,'anchors',len(a),'with target %.1f%%'%(100*a.has.mean()),'stunted share with/without target %.1f / %.1f'%(100*a[a.has].label.ne('normal').mean(),100*a[~a.has].label.ne('normal').mean()))
s.to_csv('stage2_pair_summary.csv',index=False)
