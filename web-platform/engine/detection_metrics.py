"""Detection-only scoring: midpoint-in-label, maximum-cardinality one-to-one."""
def center_metrics(pred,truth):
    from core import interval_iou
    from evaluate_agent import ratios
    edges=[[j for j,g in enumerate(truth) if 2*g[0]<=sum(p)<=2*g[1]] for p in pred]
    owners={}
    def visit(i,seen):
        for j in sorted(edges[i],key=lambda j:(abs(sum(pred[i])-sum(truth[j])),j)):
            if j in seen:continue
            seen.add(j)
            if j not in owners or visit(owners[j],seen):owners[j]=i;return True
        return False
    for i in range(len(pred)):visit(i,set())
    pairs=sorted((i,j) for j,i in owners.items());used_p={i for i,j in pairs};used_g={j for i,j in pairs}
    matches=[dict(prediction_index=i,truth_index=j,predicted=list(pred[i]),ground_truth=list(truth[j]),center_sample=sum(pred[i])/2,
        iou=interval_iou(pred[i],truth[j]),center_error_samples=abs(sum(pred[i])-sum(truth[j]))/2) for i,j in pairs]
    tp=len(pairs);fp=len(pred)-tp;fn=len(truth)-tp
    return dict(criterion='center_in_label_one_to_one',tp=tp,fp=fp,fn=fn,**ratios(tp,fp,fn),matches=matches,
        false_positives=[dict(index=i,interval=list(p),reason='duplicate_detection' if edges[i] else 'center_outside_labels',best_iou=max((interval_iou(p,g) for g in truth),default=0)) for i,p in enumerate(pred) if i not in used_p],
        false_negatives=[dict(index=j,interval=list(g),best_iou=max((interval_iou(p,g) for p in pred),default=0)) for j,g in enumerate(truth) if j not in used_g])
