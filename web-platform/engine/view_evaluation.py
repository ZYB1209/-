"""Self-contained HTML timeline: actual labels vs frozen predictions, no API calls."""
import argparse,html,json
from pathlib import Path

def percentage(x):return '未定义' if x is None else f'{x:.2%}'

def render(folder):
    report=json.loads((folder/'report.json').read_text('utf-8'));summary=report['summary'];key='center' if report['settings'].get('primary_criterion')=='center' else '0.5';primary=summary['events'][key]
    blocks=[]
    for row in report['results']:
        n=row['samples'];x=lambda point:70+point*1020/max(n-1,1)
        matched={m['prediction_index'] for m in row['events'][key]['matches']}
        shapes=['<svg viewBox="0 0 1140 145" role="img" aria-label="真实标注与检测区间对照">',
            '<text x="0" y="45">真实标注</text><text x="0" y="100">检测候选</text>',
            '<line x1="70" y1="55" x2="1090" y2="55" stroke="#ddd"/><line x1="70" y1="110" x2="1090" y2="110" stroke="#ddd"/>']
        for kind,intervals,y in [('GT',row['truth'],24),('D',row['predictions'],79)]:
            for i,(a,b) in enumerate(intervals):
                color='#138572' if kind=='GT' else '#326fce' if i in matched else '#d86d36'
                title=html.escape(f'{kind}{i+1}: [{a},{b}] / {a/500:.3f}–{b/500:.3f}名义秒')
                shapes.append(f'<rect x="{x(a):.2f}" y="{y}" width="{max((b-a+1)*1020/n,1):.2f}" height="25" fill="{color}"><title>{title}</title></rect>')
        for tick in range(6):
            point=int(tick*(n-1)/5);shapes.append(f'<text x="{x(point):.2f}" y="140" text-anchor="middle">{point}</text>')
        shapes.append('</svg>');e=row['events'][key]
        blocks.append(f'<section><h2>{html.escape(row["record"])}</h2><p>GT={row["ground_truth_count"]}；候选={row["prediction_count"]}；TP={e["tp"]} / FP={e["fp"]} / FN={e["fn"]}；P={percentage(e["precision"])} / R={percentage(e["recall"])} / F1={percentage(e["f1"])}。</p>'+''.join(shapes)+'</section>')
    page='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>智能体与真实缺陷标注对照</title>
<style>body{font:15px 'Microsoft YaHei',sans-serif;color:#18313d;background:#f2f6f7;max-width:1200px;margin:30px auto;padding:20px}section,header{background:white;border-radius:12px;padding:22px;margin:18px 0}h1{font-size:26px}h2{font-size:18px}p{line-height:1.8}svg{width:100%;font-size:12px}.warn{color:#956121}</style>'''
    page+=f'<header><h1>智能体与真实缺陷标注对照</h1><p>完成 {report["completed"]}/{report["requested"]} 条；精确率 {percentage(primary["precision"])}，召回率 {percentage(primary["recall"])}，F1 {percentage(primary["f1"])}。</p><p>绿色：GT；蓝色：按报告主标准成功匹配候选；橙色：未匹配候选。主标准：{report["settings"].get("primary_criterion","iou")}；横轴为采样点，鼠标悬停显示区间。</p><p class="warn">检出标准采用中心落在标注内、一对一匹配；历史IoU报告仍沿用原标准。重复报警也计误报。正常点多时普通准确率会偏高。本次为开发集验证，非独立测试成绩。</p></header>'
    page+=''.join(blocks)+'</html>';target=folder/'comparison.html';target.write_text(page,encoding='utf-8');return target

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('folder',type=Path);args=parser.parse_args();print(render(args.folder))
