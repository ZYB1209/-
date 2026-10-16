"""Reviewed, local literature knowledge. No PDF text is executed as instructions."""
import json, math, re
from pathlib import Path
from html import escape
from reporting import readable
PATH=Path(__file__).parent/'knowledge/domain_knowledge.json'

def load_knowledge():
    data=json.loads(PATH.read_text(encoding='utf-8'))
    if not data.get('cards') or not data.get('sources'):
        raise ValueError('专家知识库为空，请检查知识文件。')
    return readable(data)

def lookup_knowledge(query,limit=4):
    if not isinstance(query,str) or not query.strip() or len(query)>600:
        raise ValueError('知识检索需要1–600字的问题。')
    if type(limit) is not int or not 1<=limit<=6:
        raise ValueError('知识检索条数应为1–6。')
    data=load_knowledge(); q=query.strip().lower()
    q=q.replace('振动','抖动').replace('晃动','抖动').replace('股噪','股纹').replace('信噪比','snr')
    grams=set(re.findall(r'[\u4e00-\u9fff]{2}',q))
    # Overlapping bigrams provide a fallback for Chinese words not in the card tags.
    grams|={q[i:i+2] for i in range(len(q)-1) if re.fullmatch(r'[\u4e00-\u9fff]{2}',q[i:i+2])}
    ranked=[]
    for c in data['cards']:
        tags=[t.lower() for t in c['tags']]
        score=sum(3+min(len(t),6) for t in tags if t in q)
        content=(c['title']+' '+c['body']).lower()
        for g in grams:
            if g in content:
                frequency=sum(g in (x['title']+' '+x['body']) for x in data['cards'])
                score+=math.log(1+len(data['cards'])/frequency)*.3
        if c['id'].lower() in q: score+=100
        if score>0: ranked.append((score,c))
    ranked.sort(key=lambda x:(-x[0],x[1]['id']))
    selected=[c for _,c in ranked[:limit]]
    ids={r['source'] for c in selected for r in c['references']}
    return dict(knowledge_items=selected,sources=[s for s in data['sources'] if s['id'] in ids],
                query=query,knowledge_version=data['version'],
                note='资料只提供专家背景；不是本条数据的检测结果。' if selected else '没有找到相关知识；不要据此推断本条数据。')

def background():
    data=load_knowledge()
    ids={'K02','K04','K18','K19','K24','K26','K31','K33','K36','K39','K40','K41','K42','K43','K45','K48','K49','K50','K51'}
    return dict(version=data['version'],method_names=data.get('method_names',{}),source_ids=[s['id'] for s in data['sources']],
                core_facts=[dict(id=c['id'],fact=c['body'],limitation=c['limitations'])
                            for c in data['cards'] if c['id'] in ids])

def knowledge_question(question):
    """Conservative knowledge intent for the optional offline rule backend."""
    if re.search(r'(?<![A-Za-z0-9])D\d{1,3}(?![A-Za-z0-9])',question,re.I):return False
    if any(t in question for t in ['检测整条','检测这条','扫描','比较两种检测工具','评估检测']):return False
    if re.search(r'(用|使用|运行|执行).*(TFO|tfo|金字塔|目标特征).*检测',question):return False
    return any(t in question for t in ['根据论文','知识库','原理','为什么抖动','股纹是什么','2021','2023','论文参数']) or ('检测' not in question and any(t in question for t in ['TFO','金字塔','SSR','2024','2025','深度原理','截面定位原理','内部缺陷量化']))

def reference_label(ref):
    locator='第'+','.join(map(str,ref['sections']))+'节' if ref.get('sections') else 'PDF第'+','.join(map(str,ref.get('pdf_pages',[])))+'页'
    return ref['source']+' '+locator+'，'+ref['locator']

def knowledge_html():
    data=load_knowledge()
    parts=['<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">',
           '<title>钢丝绳漏磁专家知识</title><style>body{font-family:Microsoft YaHei,sans-serif;max-width:960px;margin:40px auto;padding:0 20px;color:#18313d;background:#f0f4f5;line-height:1.85}article{background:white;padding:22px;margin:18px 0;border-radius:12px}h2{font-size:20px}small{color:#61717a}a{color:#146f6b}</style>',
           '<a href="/">返回智能体</a><h1>钢丝绳漏磁专家知识</h1>',
           f'<p>已接入 {len(data["cards"])} 条知识卡，来自{sum(s.get("source_type")!="method_note" for s in data["sources"])}篇论文与{sum(s.get("source_type")=="method_note" for s in data["sources"])}份方法说明。知识用于解释和复核；工具实现与适配状态单独记录。</p>',
           '<p><a href="/钢丝绳漏磁专家知识.md">打开/保存完整知识说明</a></p>']
    for s in data['sources']:
        parts.append(f'<p><b>{escape(s["id"])}</b>：{escape(s["title"])}（{s["publication_year"]}），DOI：{escape(s["doi"])}。</p>')
    for c in data['cards']:
        refs='；'.join(reference_label(r) for r in c['references'])
        parts.append(f'<article id="{escape(c["id"])}"><h2>{escape(c["id"])} {escape(c["title"])}</h2><p>{escape(c["body"])}</p><p><b>适用限制：</b>{escape(c["limitations"])}</p><p><b>实现状态：</b>{escape(c["implementation_status"])}</p><small>出处：{escape(refs)}</small></article>')
    return ''.join(parts).encode('utf-8')
