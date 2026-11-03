import re, sys, json
from pathlib import Path
import pandas as pd
from lxml import html as lxml_html
from concurrent.futures import ProcessPoolExecutor
RAW=Path(sys.argv[1]); OUT=Path(sys.argv[2])
MONTHS={'januari':1,'februari':2,'maret':3,'april':4,'mei':5,'juni':6,'juli':7,'agustus':8,'september':9,'oktober':10,'november':11,'desember':12}
def parse(p):
    text=p.read_bytes().decode('utf-8',errors='replace')
    text=re.sub(r"<!--.*?-->","",text,flags=re.S)
    doc=lxml_html.fromstring(text); tables=doc.xpath('//table')
    t=tables[0]
    headers=[th.text_content().strip() for th in t.xpath('.//thead//th')]
    rows=[];bad=0;linkok=0
    for tr in t.xpath('.//tbody/tr'):
        v=[td.text_content().replace('\xa0',' ').strip() for td in tr.xpath('./td')]
        if len(v)!=len(headers): bad+=1; continue
        r=dict(zip(headers,v))
        links=' '.join(tr.xpath('.//a/@href')+tr.xpath('.//a/@onclick'))
        ids=set(re.findall(r"/Balita/(?:perkembangan|ukur)/([^./]+)\.html",links))|set(re.findall(r"view_penerima\('([^']+)'\)",links))
        r['_id_link_consistent']= ids=={r.get('NIK','').strip()}
        rows.append(r)
    cap=t.xpath('.//caption'); 
    f=pd.DataFrame(rows)
    y=int(p.parent.name); m=MONTHS[p.stem.split('. ')[1].strip().lower()]
    f['file_year']=y; f['file_month']=m; f['source_file']=f'{y}/{p.name}'
    meta=dict(file=f'{y}/{p.name}',n_tables=len(tables),headers=headers,rows=len(f),malformed=bad,caption=cap[0].text_content().strip() if cap else '')
    return f,meta
files=sorted(p for p in RAW.glob('*/*.xls'))
with ProcessPoolExecutor(4) as ex: res=list(ex.map(parse,files))
df=pd.concat([r[0] for r in res],ignore_index=True)
df.to_pickle(OUT/'kab_raw_all.pkl')
json.dump([r[1] for r in res],open(OUT/'file_meta.json','w'),indent=1,ensure_ascii=False)
print(len(files),'files',len(df),'rows')
