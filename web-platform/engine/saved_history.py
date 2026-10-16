"""Disk-backed result history and safe reconstruction of detection context."""
import json,re
from pathlib import Path

def load_saved(root,identifier):
    if not isinstance(identifier,str) or not re.fullmatch(r'[a-f0-9]{16}',identifier):raise ValueError('保存结果编号无效')
    path=Path(root)/'sessions'/(identifier+'.json')
    if not path.is_file():raise ValueError('保存的结果不存在')
    data=json.loads(path.read_text(encoding='utf-8'))
    if not isinstance(data,dict) or not isinstance(data.get('result'),dict):raise ValueError('保存的结果格式无效')
    data['saved_result']='/runs/sessions/'+identifier+'.json';data['history_id']=identifier
    return data

def list_saved(root,known_records=None,record=None):
    folder=Path(root)/'sessions'
    if not folder.is_dir():return []
    items=[]
    for path in sorted(folder.glob('*.json'),key=lambda p:p.stat().st_mtime_ns,reverse=True):
        if not re.fullmatch(r'[a-f0-9]{16}',path.stem):continue
        try:data=load_saved(root,path.stem)
        except (ValueError,OSError):continue
        if known_records is not None and data.get('record') not in known_records:continue
        if record and data.get('record')!=record:continue
        r=data['result'];items.append(dict(id=path.stem,record=data.get('record'),created_at=data.get('created_at'),
            question=data.get('question',''),session=data.get('session'),mode=r.get('mode'),
            candidates=len(r['candidates']) if 'candidates' in r else None,
            has_review=bool(r.get('manual_review')),saved_result=data['saved_result']))
    return items[:100]

def hydrate(agent,root,identifier,session,record):
    saved=load_saved(root,identifier)
    if saved.get('session')!=session or saved.get('record')!=record:raise ValueError('恢复结果与当前对话或记录不一致')
    candidates=[saved]
    if 'candidates' not in saved['result'] and 'fast' not in saved['result']:
        for item in list_saved(root,record=record):
            if item['session']==session and item['id']!=identifier:candidates.append(load_saved(root,item['id']))
    for data in candidates:
        r=data['result'];prediction=r if 'candidates' in r else r.get('balanced')
        if prediction is not None:
            agent.memory[record]=prediction
            if prediction.get('conditions'):agent.condition_memory[record]=prediction['conditions']
            break
    if hasattr(agent,'dialogue'):
        agent.dialogue[record]=[dict(role='user',content=saved.get('question','')),dict(role='assistant',content=saved.get('answer',''))]
    return saved
