"""Stage 1: district master dataset (pseudonymised, no PII). Decisions approved 2026-10-16:
A3 exclude children with suspected double registration (same name+DOB+sex+village, active in overlapping months);
C6 child with two birth dates: modal date if difference <=30 days, else exclude child;
F8 consecutive-month height drop >1.5 cm flagged, values unchanged."""
import sys, json, hashlib, secrets
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, sys.argv[3])
from who_haz_copy import load_tables, haz_who
RAW=Path(sys.argv[1]); OUT=Path(sys.argv[2]); WHO=sys.argv[4]
k=pd.read_pickle(RAW/'kab_raw_all.pkl')
audit={'n_files':int(k.source_file.nunique()),'n_raw_rows':len(k),'months':f"{k.file_year.min()}-{k.file_month[k.file_year.idxmin()]:02d}..2026-09"}
k['month_index']=(k.file_year-2022)*12+k.file_month-1
k['calendar_month']=k.file_year.astype(str)+'-'+k.file_month.astype(str).str.zfill(2)
assert not k.duplicated(['link_id','month_index']).any()
# A3
c=k.groupby('link_id').agg(nm=('Nama','first'),dob=('Tgl Lahir',lambda s:s.mode().iloc[0]),jk=('JK','first'),desa=('Desa/Kel','first'),m0=('month_index','min'),m1=('month_index','max')).reset_index()
c['key']=c.nm.str.upper().str.replace(r'\s+',' ',regex=True).str.strip()+'|'+c.dob+'|'+c.jk+'|'+c.desa
excl=set()
for _,g in c[c.duplicated('key',keep=False)].groupby('key'):
    g=g.sort_values('m0')
    if (g.m0.iloc[1:].values<=g.m1.iloc[:-1].values).any(): excl|=set(g.link_id)
audit['A3_children_excluded_double_registration']=len(excl)
audit['A3_rows_excluded']=int(k.link_id.isin(excl).sum())
k=k[~k.link_id.isin(excl)].copy()
# C6
dob=pd.to_datetime(k['Tgl Lahir'],errors='coerce'); k['dob']=dob
nd=k.groupby('link_id').dob.nunique(); c6=[]
for i in nd[nd>1].index:
    v=k.loc[k.link_id==i,'dob']; span=(v.max()-v.min()).days
    if span<=30: k.loc[k.link_id==i,'dob']=v.mode().iloc[0]; c6.append({'action':'modal date','span_days':span})
    else: k=k[k.link_id!=i]; c6.append({'action':'excluded','span_days':span})
audit['C6_children_two_birth_dates']=c6
# pseudonym
salt=secrets.token_hex(32); (OUT/'private'/'salt_v4.txt').write_text(salt)
k['child_id']=[hashlib.sha256((salt+x).encode()).hexdigest()[:16] for x in k.link_id]
# numerics
num=lambda s: pd.to_numeric(s.replace({'-':np.nan,'':np.nan}),errors='coerce')
k['sex_male']=(k.JK=='L').astype(int)
k['birth_weight_kg']=num(k['BB Lahir']); k['birth_length_cm']=num(k['TB Lahir'])
k['weight_kg']=num(k.Berat); k['height_cm']=num(k.Tinggi); k['muac_cm']=num(k.LiLA)
k['measure_position']=k['Cara Ukur'].replace({'-':np.nan})
k['waz']=num(k['ZS BB/U']); k['haz']=num(k['ZS TB/U']); k['whz']=num(k['ZS BB/TB'])
k['age_days']=(pd.to_datetime(k['Tanggal Pengukuran'])-k.dob).dt.days
k['age_months']=k.age_days/30.4375
T=load_tables(WHO); k['haz_who']=haz_who(k.age_days.values,k.JK.values,k.height_cm.values,k['Cara Ukur'].values,T)
lab={'Normal':'normal','Pendek':'stunted','Sangat Pendek':'severely stunted','Tinggi':'tall','Outlier':'system_outlier','-':'missing'}
k['tbu_category']=k['TB/U'].map(lab).fillna('missing')
# validity (same rules as v3; out-of-range values set missing)
rng={'birth_weight_kg':(0.5,6),'birth_length_cm':(30,60),'weight_kg':(1,40),'height_cm':(40,125),'muac_cm':(7,30),'waz':(-6,5),'whz':(-5,5)}
setmiss={}
for col,(lo,hi) in rng.items():
    bad=k[col].notna()&~k[col].between(lo,hi); setmiss[col]=int(bad.sum()); k.loc[bad,col]=np.nan
k['muac_observed']=k.muac_cm.notna().astype(int)
audit['values_set_missing_by_validity_rule']=setmiss
# eligibility of an observation as anchor/target
reasons={'system_outlier_or_missing_category':~k.tbu_category.isin(['normal','stunted','severely stunted','tall']),
 'tall_haz_gt_3':k.tbu_category.eq('tall'),
 'haz_missing_or_implausible':~k.haz.between(-6,6),
 'measurement_before_birth':k.age_days<0,
 'age_gt_60_months':k.age_days>1856,
 'height_missing_or_implausible':k.height_cm.isna()}
k['eligible']=True; er={}
for r,m in reasons.items():
    er[r]=int((m&k.eligible).sum()); k.loc[m,'eligible']=False
audit['excluded_observation_reasons_sequential']=er
k['label']=np.where(k.eligible,k.tbu_category,None)
# F8 flag
k=k.sort_values(['child_id','month_index'])
g=k.groupby('child_id'); dL=g.height_cm.diff(); dm=g.month_index.diff()
k['flag_height_drop_gt1_5cm']=((dm==1)&(dL<-1.5)).astype(int)
# WHO agreement
e=k.eligible
whocat=np.select([k.haz_who<-3,k.haz_who<-2,k.haz_who<=3],['severely stunted','stunted','normal'],'tall')
audit['who_category_agreement_by_year']={int(y):round(float(v),4) for y,v in (whocat==k.tbu_category)[e].groupby(k.file_year[e]).mean().items()}
# programme fields (raw codes, not yet predictors)
k['vita_t']=(k['Jml Vit A']=='1').astype(int)
for src,dst in [('KPSP','kpsp_t'),('KIA','kia_t'),('Kelas Ibu Balita','kelas_ibu_t'),('MBG','mbg_t'),('Naik Berat Badan','weight_gain_code_t')]:
    k[dst]=k[src].replace({'':'-'})
k['puskesmas']=k.Pukesmas; k['kecamatan']=k.Kec
keep=['child_id','calendar_month','month_index','puskesmas','kecamatan','Desa/Kel','Posyandu','sex_male','birth_weight_kg','birth_length_cm',
 'age_days','age_months','weight_kg','height_cm','measure_position','muac_cm','muac_observed','waz','haz','whz','haz_who','tbu_category','eligible','label',
 'flag_height_drop_gt1_5cm','vita_t','kpsp_t','kia_t','kelas_ibu_t','mbg_t','weight_gain_code_t']
m=k[keep].rename(columns={'Desa/Kel':'village','Posyandu':'posyandu'}).reset_index(drop=True)
m.to_pickle(OUT/'master_v4_pseudonymised.pkl')
audit.update({'n_rows_master':len(m),'n_children':int(m.child_id.nunique()),'eligible_observations':int(m.eligible.sum()),
 'label_counts':m.label.value_counts().to_dict(),'puskesmas':int(m.puskesmas.nunique()),'villages':int(m.village.nunique()),'posyandu':int(m.posyandu.nunique()),
 'height_drop_flag_rate_by_year':{int(y):round(float(v),4) for y,v in m.flag_height_drop_gt1_5cm.groupby(m.calendar_month.str[:4].astype(int)).mean().items()}})
json.dump(audit,open(OUT/'master_v4_audit.json','w'),indent=1)
# summaries
m['year']=m.calendar_month.str[:4]
s=m[m.eligible].assign(st=lambda x:x.label.isin(['stunted','severely stunted'])).groupby(['year','puskesmas']).agg(children=('child_id','nunique'),observations=('child_id','size'),stunted_pct=('st','mean'))
s['stunted_pct']=(s.stunted_pct*100).round(1); s.to_csv(OUT/'summary_by_year_puskesmas.csv')
print(json.dumps(audit,indent=1))
