# 本地数据格式（数据不随代码发布）

目录结构：

```text
dataset/MFL_TriAxial_AllAngles_Dataset/
  manifest.csv
  data/...
  annotations/...
  positions/...  # 可选实测位置
```

manifest 字段：`record_id,source_type,source_collection,group,data_path,annotation_path,row_count,data_sha256,annotation_sha256`。路径相对上述数据集根目录，SHA-256 为对应文件的校验值。`source_type=physical_acquisition` 代表实际采集，评估默认选择这一类记录。该字段本身不能证明缺陷或钢绳相互独立。

信号 CSV 需要 `mag_1x` 至 `mag_32x`、`mag_1y` 至 `mag_32y`、`mag_1z` 至 `mag_32z` 共 96 个有限数值列。可包含 `time(0.1ms)`。当前实现采用固定名义 500 Hz，其他采样率需先修改并验证时间与滤波参数，不能直接套用。

真实标注 CSV 必需字段：`label,start_row_idx,end_row_idx`；异常标记为 `anomaly`。索引暂按 0 起始、首尾都包含；所有区间必须在原始记录内。评估采用未标异常点为正常的假设，使用前应确认标注完整性。

可选实测位置表：`positions/<record_id>.csv`，字段 `sample_index,position_m`。采样点升序、覆盖首尾、至少两个锚点，位置值有限；在锚点之间线性插值。应使用同步编码器或可靠位移测量，不能猜测扫描速度。

仓库的 `.gitignore` 排除了全部数据集内容，仅保留本说明。复制数据后请使用 `git status --ignored` 确认没有把数据加入提交。
