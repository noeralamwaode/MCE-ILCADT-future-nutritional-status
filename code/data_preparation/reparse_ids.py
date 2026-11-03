import re, sys
from pathlib import Path
import pandas as pd
from lxml import html as lxml_html
from concurrent.futures import ProcessPoolExecutor
RAW=Path(sys.argv[1])
MONTHS={'januari':1,'februari':2,'maret':3,'april':4,'mei':5,'juni':6,'juli':7,'agustus':8,'september':9,'oktober':10,'november':11,'desember':12}
def parse(p):
    text=p.read_bytes().decode('utf-8',errors='replace'); text=re.sub(r"<!--.*?-->","",text,flags=re.S)
    t=lxml_html.fromstring(text).xpath('//table')[0]; out=[]
    for tr in t.xpath('.//tbody/tr'):
        links=' '.join(tr.xpath('.//a/@href')+tr.xpath('.//a/@onclick'))
        ids=set(re.findall(r"/Balita/(?:perkembangan|ukur)/([^./]+)\.html",links))|set(re.findall(r"view_penerima\('([^']+)'\)",links))
        out.append('|'.join(sorted(ids)))
    return out
files=sorted(RAW.glob('*/*.xls'))
with ProcessPoolExecutor(2) as ex: res=list(ex.map(parse,files))
d=pd.read_pickle('kab_raw_all.pkl')
d['link_id']=[x for r in res for x in r]
d.to_pickle('kab_raw_all.pkl'); print('ok')
