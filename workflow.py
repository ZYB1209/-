"""Explicit coarse-to-local workflow, reused by both rule and model planners.
No annotations are available to this execution engine.
"""
from __future__ import annotations
import json,time,uuid
from pathlib import Path

NODES={
 'condition_analysis':('信号工况与噪声识别','量化低阶背景、环周覆盖、斜纹脊与共现'),
 'adaptive_filter':('自适应滤波','模型选分段路线→论文工程算法→响应保留检查→必要时回退'),
 'adaptive_detection':('处理后检测','浮点包络阈值分段→时间/可选长度映射'),
 'audit':('数据检查','确认三轴通道、采样率、来源与坐标条件'),
 'signal_image':('三轴信号成图','构造完整分辨率浮点图矩阵'),
 'temporal_channel':('时间检测：通道响应工具','提取通道局部响应并定位时间候选'),
 'temporal_energy':('时间检测：补充能量工具','从另一种响应提取方式产生补充候选'),
 'candidate_merge':('候选融合','合并区间、保留工具来源'),
 'boundary_refine':('局部定位：范围复核','局部提取、三轴证据、边界提议与保守选择'),
 'coordinate_mapping':('坐标映射','采样点到时间；有实测映射才给长度坐标'),
 'render_image':('证据图生成','显示候选并记录图像坐标映射'),
 'structured_report':('结构化报告','保存位置、范围、状态与执行日志'),
}

class DetectionWorkflow:
    def run(self,record,mode='fast',region=None,trace=None):
        import core as tools
        if mode not in ('fast','balanced'): raise ValueError('未知检测组合')
        if region is not None:
            if len(region)!=2 or any(type(v) is not int or v<0 for v in region) or region[0]>region[1]:
                raise ValueError('采样点区间无效')
        context={'record':record,'mode':mode,'region':region,'detections':[]}
        log=[]; run_id=uuid.uuid4().hex[:12]
        def execute(name,fn):
            began=time.time()
            node=dict(id=name,title=NODES[name][0],status='running')
            try:
                summary=fn()
                node.update(status='completed',elapsed_s=round(time.time()-began,3),output=summary)
            except Exception as exc:
                node.update(status='failed',error=type(exc).__name__)
                log.append(node)
                (tools.RUNS/('failed_workflow_'+run_id+'.json')).write_text(json.dumps(dict(run_id=run_id,record=record,nodes=log),ensure_ascii=False,indent=2),encoding='utf-8')
                raise
            log.append(node)
            if trace is not None:
                trace.append(dict(tool=name,record=record,arguments={'mode':mode,'scope_samples':region},elapsed_s=node['elapsed_s']))
        def audit():
            context['audit']=tools.audit(record)
            if region is not None and region[1]>=context['audit']['samples']: raise ValueError('范围超过记录采样点数')
            return context['audit']
        def image():
            matrix=tools.image_tensor(record)
            return dict(shape=list(matrix.shape),representation='三轴×采样点×32通道浮点图',encoder_trained=False)
        def detect(name):
            result=getattr(tools,name)(record);context['detections'].append(result)
            return dict(candidate_count=len(result['intervals']),threshold=result['threshold'])
        def merge():
            context['merged']=tools.union_candidates(context['detections'])
            return dict(candidate_count=len(context['merged']),merge_rule='重叠或间隙≤25点，保留来源')
        def local():
            context['candidates']=[tools.refine(record,c) for c in context['merged']]
            return dict(candidate_count=len(context['candidates']),accepted_refinements=sum(c['refinement_used'] for c in context['candidates']),
                        policy='宽响应保留；短候选满足边界条件才采纳细化；不计算真实断口长度')
        def coordinate():
            position=tools.load(record)[3]
            for i,c in enumerate(context['candidates'],1):
                c.update(id=f'D{i:02d}',start_s=round(c['refined_start']/500,4),end_s=round(c['refined_end']/500,4),
                         center_sample=(c['refined_start']+c['refined_end'])//2)
                if position is not None:
                    p=position[c['refined_start']:c['refined_end']+1]
                    c['position_m']=[float(p.min()),float(p.max())]
            if region is not None:
                context['candidates']=[c for c in context['candidates'] if c['refined_end']>=region[0] and c['refined_start']<=region[1]]
            return dict(has_length_mapping=position is not None,output_units=['samples','nominal_seconds']+(['meters'] if position is not None else []),
                        scope_candidates=len(context['candidates']))
        def evidence():
            # Scoped query retains whole-record calibration and stable global candidate IDs.
            context['image']=tools.render_image(record,*(region or [0,None]),candidates=context['candidates'],name=mode+('_region' if region else '')+'_overview')
            return dict(image=context['image'],pixel_mapping_saved=True)
        def report():
            context['result']=dict(record=record,mode=mode,audit=context['audit'],candidates=context['candidates'],params=tools.PARAMS,
                tools=[r['tool'] for r in context['detections']],thresholds={r['tool']:r['threshold'] for r in context['detections']},
                scope_samples=region,stage='untrained prototype',score_kind='响应强度；不表示缺陷概率',image=context['image'])
            return dict(candidate_count=len(context['candidates']),result_file=mode+'_result.json')
        execute('audit',audit);execute('signal_image',image)
        execute('temporal_channel',lambda:detect('temporal_channel'))
        if mode=='balanced':execute('temporal_energy',lambda:detect('temporal_energy'))
        else:log.append(dict(id='temporal_energy',title=NODES['temporal_energy'][0],status='skipped',reason='当前选择通道响应主流程'))
        execute('candidate_merge',merge);execute('boundary_refine',local)
        execute('coordinate_mapping',coordinate);execute('render_image',evidence);execute('structured_report',report)
        result=context['result']
        result['workflow']=dict(run_id=run_id,version='0.3',status='completed',nodes=log,
            decisions=dict(extra_detector=mode=='balanced',length_mapping=context['audit']['has_position'],
                           scope='whole_record' if region is None else 'region_overlap',
                           local_refinement='conditional',annotations_used=False))
        folder=tools.RUNS/record;folder.mkdir(exist_ok=True)
        (folder/(mode+'_result.json')).write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
        (folder/('workflow_'+run_id+'.json')).write_text(json.dumps(result['workflow'],ensure_ascii=False,indent=2),encoding='utf-8')
        return result

def workflow_catalog():
    return dict(version='2.0',nodes={k:dict(title=v[0],purpose=v[1]) for k,v in NODES.items()},
      routes={
        'adaptive':{'tool':'adaptive_detect','arguments':{'strategy':'auto'},'purpose':'inspect_conditions→模型选分段路线→论文算法滤波→保留检查/回退→检测'},
        'full_scan':{'tool':'scan','arguments':{'mode':'fast'},'purpose':'整条检测'},
        'region_scan':{'tool':'scan','arguments':{'mode':'fast','region':[2500,5000]},'purpose':'指定5–10秒的范围检测'},
        'supplementary_scan':{'tool':'scan','arguments':{'mode':'balanced'},'purpose':'增加补充检测工具，结果不保证更准确'},
        'explain_candidate':{'tool':'explain','arguments':{'candidate_id':'D01'},'purpose':'已有结果复用→局部证据图→解释'},
        'compare_tools':{'tool':'compare','arguments':{},'purpose':'运行单/双工具组合并比较'},
        'evaluate':{'tool':'evaluate','arguments':{'mode':'fast'},'purpose':'用户明确要求时单独读取标注评估'},
      },contract='模型选择任务工具与参数；执行器落实工具组合和条件分支。工具结果回传模型生成解释。')

if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--record',default='s1a0-1');parser.add_argument('--mode',default='fast',choices=['fast','balanced'])
    args=parser.parse_args();result=DetectionWorkflow().run(args.record,args.mode)
    print('Completed workflow:',result['workflow']['run_id'],'Candidates:',len(result['candidates']))
