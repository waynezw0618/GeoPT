# STAR-CCM+ -> GeoPT 数据管线（船舶水动力）

这个目录提供了你要的三段式流程：

1. **STAR-CCM+ Java 宏**导出压力/速度/几何等字段为 CSV。
2. **Python 转换脚本**把 CSV 转成 GeoPT 可读的 `x_i.npy / y_i.npy / cond_i.npy`。
3. **Linux Shell 批处理**遍历 `.sim` 案例并自动完成导出+转换。

---

## 1) STAR-CCM+ Java 宏

文件：`ExportGeoPTFields.java`

### 你需要在 STAR-CCM+ 里先准备
- 一个体网格采样表：`GeoPT_Volume_Table`
- 一个船体表面采样表：`GeoPT_Surface_Table`

表里至少要有这些列（列名可通过 `--colmap` 重映射）：
- 位置：`Position[0], Position[1], Position[2]`
- 速度：`Velocity[0], Velocity[1], Velocity[2]`
- 压力：`Pressure`
- 法向（推荐）：`Normal[0], Normal[1], Normal[2]`

### 执行方式
在 STAR-CCM+ 命令行里对单个案例执行：

```bash
starccm+ -batch /path/to/ExportGeoPTFields.java /path/to/case.sim
```

导出结果默认在 `.sim` 同目录的 `geopt_exports/` 下：
- `<case_name>_volume.csv`
- `<case_name>_surface.csv`

---

## 2) Python 转换脚本

文件：`starccm_csv_to_geopt.py`

### 依赖
```bash
pip install numpy pandas scikit-learn
```

### 单案例转换
```bash
python tools/starccm_geopt/starccm_csv_to_geopt.py \
  --volume_csv /data/geopt_exports/case_001_volume.csv \
  --surface_csv /data/geopt_exports/case_001_surface.csv \
  --outdir /data/geopt_npys \
  --case_id 1 \
  --cond "Fn=0.26,heel_deg=0.0,yaw_deg=3.0" \
  --compute_sdf
```

输出：
- `x_1.npy`：`(N,7)` -> `[x,y,z,sdf_or_0,nx,ny,nz]`
- `y_1.npy`：`(N,4)` -> `[p,ux,uy,uz]`
- `cond_1.npy`：`(C,)` 条件向量（可放 Froude 数、漂角、纵倾等）

> 注意：转换脚本默认应用了与 GeoPT DTCHull 脚本一致的坐标重排（前向映射到 `-X`、Y/Z 互换）和中心化步骤，便于接入 GeoPT。

---

## 3) Linux 批处理脚本

文件：`batch_starccm_to_geopt.sh`

### 批量执行
```bash
bash tools/starccm_geopt/batch_starccm_to_geopt.sh \
  /opt/Siemens/STAR-CCM+18.06.006/starccm+ \
  /data/star_cases \
  /data/geopt_npys \
  /workspace/GeoPT/tools/starccm_geopt/ExportGeoPTFields.java \
  4
```

参数说明：
1. `starccm+` 可执行程序路径
2. `.sim` 文件目录（脚本会遍历一级目录下所有 `.sim`）
3. GeoPT 输出目录（npy）
4. Java 宏路径
5. 并行数（可选，默认 1）

---

## 与 GeoPT 训练脚本对接建议

- 你的数据目录里应有成对文件：`x_*.npy`, `y_*.npy`, `cond_*.npy`
- 若用现有 `DTCHull` loader，建议先抽样检查每个 case 点数是否足够（默认会随机取 80k 点）。
- 如果你要专门做“船舶水动力”新任务，推荐新建一个 data loader（结构可参考 `data_provider/data_loader.py` 中 `DTCHull` 类）。

