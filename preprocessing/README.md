# H5 解析

## L1000 数据解析

### obs 以矩阵方式存放实验信息

```
['cell_id', 'det_plate', 'det_well', 'lincs_phase', 'pert_dose', 'pert_dose_unit', 'pert_id', 'pert_iname', 'pert_mfc_id', 'pert_time', 'pert_time_unit', 'pert_type', 'rna_plate', 'rna_well']
```

数据样式

```
cell_id                               A375     # 细胞系 
det_plate         REP.A001_A375_24H_X1_B22
det_well                               A03
lincs_phase                              2     # lincs 计划阶段
pert_dose                           -666.0     # 药物剂量 
pert_dose_unit                        -666     # 药物剂量单位
pert_id                               DMSO     # 药物 id
pert_iname                            DMSO     # 药物名称
pert_mfc_id                           -666     # brd_id + 药物生产批次
pert_time                             24.0     # 药物处理时间
pert_time_unit                           h     # 药物处理时间单位
pert_type                      ctl_vehicle     # 药物类型
rna_plate                              nan
rna_well                               nan
```

brd_id 药物标识

[pert_mfc_id](https://clue.io/connectopedia/what_is_a_brd_id)

[字段解释](https://clue.io/connectopedia/glossary)


### uns 类似于 python 的字典, 可以存多种类型的数据

### var (Dataframe) 存储基因图谱对应的 gene 的名称及相关信息

同一个细胞系（同一种药物处理，同一种药物剂量处理）在不同的检测板和检测孔测序的

### 数据处理

treat 使用 pert_dose = 10, pert_dose_unit = um 的数据

control 使用 pert_iname = DMSO 的数据

```bash
python lincs.py
python lincs_smiles.py
python lincs_split.py
```

## sciplex3 

### obs

```
dose       # 药物剂量，以 nm 为单位，实验的 treat 使用的是 10uM = 10000nm，control 都是 0 nm

g1s_score  # 是一个用于评估细胞周期阶段的指标，它是基于一组与G1期和S期相关的基因的表达水平计算的。一般来说，g1s_score越高，表示细胞越接近DNA合成期（S期）。

g2m_score  # 是另一个用于评估细胞周期阶段的指标，它是基于一组与G2期和M期相关的基因的表达水平计算的。一般来说，g2m_score越高，表示细胞越接近有丝分裂期（M期）


```

### 生成数据

train: 93962

valid: 9478

test: 3883

```bash
python chemcpa_sciplex.py

python chemcpa_de_gene.py

python chemcpa_sciplex_integrated.py
```