from pathlib import Path
import csv, json, math, statistics, hashlib, html, shutil
from collections import Counter
from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.section import WD_SECTION_START, WD_ORIENT
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_CELL_VERTICAL_ALIGNMENT

ROOT=Path('/home/f/Documents/FlagTree')
OUT=ROOT/'output/f06-20261105'
DATA=OUT/'data'
DATA.mkdir(parents=True,exist_ok=True)
def read(p): return json.loads((ROOT/p).read_text())
PBASE='work/2026-11-03/f06-p800-e2e-opt'
RBASE='work/2026-11-02/f06-large-matmul-4090'
SBASE='work/2026-11-04/f06-p800-special-opt'
MBASE='work/2026-11-05/f06-p800-parallel-mul'
ps=read(PBASE+'/summary.json'); pm=read(PBASE+'/measurements.json')
rs=read(RBASE+'/summary.json')
with (ROOT/RBASE/'cases.csv').open() as f: rr=list(csv.DictReader(f))
sp=read(SBASE+'/summary.json'); mm=read(MBASE+'/comparison.json')
fs=read('work/2026-11-03/f06-p800-fallback-root-cause/summary.json')
fa=read('work/2026-11-03/f06-p800-fallback-analysis/all-cases.json')
fallback={r['ordinal']:r for r in fa}
names={
 'matmul_bias':'MatMul + Bias','matmul_bias_relu':'MatMul + Bias + ReLU',
 'matmul_bias_gelu_exact':'MatMul + Bias + GELU exact','matmul_bias_gelu_tanh':'MatMul + Bias + GELU tanh',
 'matmul_bias_silu':'MatMul + Bias + SiLU','add_softmax':'Add + Softmax',
 'add_layernorm':'Add + LayerNorm 无仿射','add_layernorm_weight':'Add + LayerNorm 仅 weight',
 'add_layernorm_bias':'Add + LayerNorm 仅 bias','add_layernorm_affine':'Add + LayerNorm 仿射',
 'mul_sum':'Mul + Sum','mul_sum_keepdim':'Mul + Sum keepdim'}
codes={k:f'T{i+1:02}' for i,k in enumerate(names)}
dtypes={'float32':'FP32','float16':'FP16','bfloat16':'BF16'}
routes={'sdnn':'SDNN','cluster':'cluster','eager':'原生 E','inductor':'原生 I',
 'sdnn_cluster_at_1':'S→C mm后','sdnn_cluster_at_2':'S→C bias后'}
def num(v):
 try: return float(v) if v not in ('',None) else None
 except (ValueError,TypeError): return None
def fmt(v,n=3): return '-' if num(v) is None else f'{float(v):.{n}f}'
def pair(g,e,n=3): return fmt(g,n)+' / '+fmt(e,n)
def gm(xs): return math.exp(statistics.mean(math.log(x) for x in xs)) if xs else None
def csvout(name,rows):
 keys=list(dict.fromkeys(k for r in rows for k in r))
 with (DATA/name).open('w',encoding='utf-8-sig',newline='') as f:
  w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows({k:json.dumps(v,ensure_ascii=False) if isinstance(v,(dict,list)) else v for k,v in r.items()} for r in rows)

# All source measurements remain at their original precision in the evidence package.
prows=[]
for r in pm:
 c=r['case']; out={k:c.get(k) for k in ['case_id','kind','dtype','M','N','K','tolerance']}
 out.update(ordinal=r['ordinal'],size='small' if c['input_builder']=='historical_small' else 'large',
   selected=r['decision']['selected'],fallback=r['decision']['fallback'],status=r['status'],
   fallback_category=fallback[r['ordinal']]['category'],physical_device=r['physical_device'])
 for path in ['eager','inductor','auto']:
  for metric in ['gpu_us','e2e_us']:out[path+'_'+metric]=r['routes'].get(path,{}).get(metric)
 for metric in ['gpu','e2e']:
  for base in ['eager','inductor']:out[metric+'_vs_'+base]=r.get('speedup_'+metric+'_vs_'+base)
 out['inductor_failure']=r.get('inductor_failure',fallback[r['ordinal']].get('baseline_inductor_failure'))
 prows.append(out)
csvout('P800_216_cases.csv',prows)
shutil.copyfile(ROOT/RBASE/'cases.csv',DATA/'RTX4090_216_cases.csv')
csvout('P800_MulSum_36_cases_all_routes.csv',mm)
csvout('P800_fallback_216_classification.csv',fa)
srows=[]
for c in sp['cases']:
 for route,r in c['routes'].items():
  srows.append(dict(ordinal=c['ordinal'],kind=c['kind'],dtype=c['dtype'],scale=c['scale'],shape=c['shape'],
    route=route,status=r.get('status'),reason=r.get('reason'),ordinary_finite_pass=r.get('ordinary_finite_pass'),
    special_passed=r.get('special_passed'),gpu_us=r.get('gpu_us'),e2e_us=r.get('e2e_us'),
    gpu_vs_eager=r.get('ratio',{}).get('eager',{}).get('gpu'),e2e_vs_eager=r.get('ratio',{}).get('eager',{}).get('e2e'),
    gpu_vs_inductor=r.get('ratio',{}).get('inductor',{}).get('gpu'),e2e_vs_inductor=r.get('ratio',{}).get('inductor',{}).get('e2e'),
    failed_patterns=r.get('failed_patterns'),classification_failures=r.get('classification_failures'),
    probe_finite_failures=r.get('probe_finite_failures'),e2e_pass=r.get('e2e_pass_vs_baseline'),kernels=r.get('kernels')))
csvout('P800_special_216_cases_all_routes.csv',srows)
sources=[PBASE+'/summary.json',PBASE+'/measurements.json',PBASE+'/final-audit.json',PBASE+'/remote-cleanup.json',
 RBASE+'/summary.json',RBASE+'/cases.csv',RBASE+'/focus-summary.json',RBASE+'/validation.json',
 SBASE+'/summary.json',SBASE+'/report-validation.json',SBASE+'/cleanup-receipt.json',
 MBASE+'/comparison.json',MBASE+'/summary.json',MBASE+'/verification.json',MBASE+'/cleanup-receipt.json',
 'work/2026-11-03/f06-p800-fallback-root-cause/summary.json',
 'work/2026-11-03/f06-p800-fallback-analysis/all-cases.json',
 'work/2026-11-03/f06-p800-e2e-diagnosis/diagnosis-summary.json',
 'work/2026-11-03/f06-p800-counter-research/analysis.json',
 'work/2026-11-03/f06-p800-bf16-formats/summary.json',
 'work/2026-11-05/f06-p800-mulsum-plans/STATE.json']
complete=read(MBASE+'/remote-final/complete.json')
sources += [MBASE+'/remote-final/'+record['path']+'/result.json' for record in complete.values()]
manifest=[]
for rel in sources:
 p=ROOT/rel; dest=DATA/'sources'/rel;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,dest)
 manifest.append(dict(source=rel,bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
oldpdf=Path('/home/f/Documents/xwechat_files/wxid_c4hhsflf60v622_f674/temp/RWTemp/2026-11/970db33da575f7aa4577c155856e56d4/F06本周工作报告_20261020.pdf')
oldout=DATA/'sources/historical/F06本周工作报告_20261020.pdf';oldout.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(oldpdf,oldout)
manifest.append(dict(source='historical/F06本周工作报告_20261020.pdf',bytes=oldpdf.stat().st_size,sha256=hashlib.sha256(oldpdf.read_bytes()).hexdigest()))
(DATA/'source_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2))

doc=Document()
sec=doc.sections[0]; sec.page_width=Inches(8.5);sec.page_height=Inches(11)
sec.top_margin=sec.bottom_margin=Inches(.65);sec.left_margin=sec.right_margin=Inches(.65)
sec.header_distance=sec.footer_distance=Inches(.25)
for n in ['Normal','Title','Subtitle','Heading 1','Heading 2']:
 s=doc.styles[n];s.font.name='Noto Sans CJK SC';s.font.color.rgb=RGBColor(0,0,0)
 s._element.get_or_add_rPr().get_or_add_rFonts().set(qn('w:eastAsia'),'Noto Sans CJK SC')
 s.paragraph_format.space_after=Pt(6);s.paragraph_format.line_spacing=Pt(15)
 for b in list(s._element.iter(qn('w:pBdr'))):b.getparent().remove(b)
 snap=OxmlElement('w:snapToGrid');snap.set(qn('w:val'),'0');s._element.get_or_add_pPr().append(snap)
doc.styles['Normal'].font.size=Pt(10.5)
doc.styles['Title'].font.size=Pt(23)
doc.styles['Heading 1'].font.size=Pt(17)
doc.styles['Heading 2'].font.size=Pt(12)
doc.styles['Subtitle'].font.size=Pt(10)
doc.styles['Title'].paragraph_format.line_spacing=Pt(30)
doc.styles['Heading 1'].paragraph_format.line_spacing=Pt(23)
doc.styles['Heading 1'].paragraph_format.space_before=Pt(4)
doc.styles['Heading 1'].paragraph_format.space_after=Pt(10)
doc.styles['Heading 2'].paragraph_format.line_spacing=Pt(17)
doc.styles['Heading 2'].paragraph_format.space_before=Pt(9)
doc.styles['Heading 2'].paragraph_format.space_after=Pt(5)
doc.core_properties.title='F06 融合测试结果与方法问题报告'
doc.core_properties.subject='P800 与 RTX 4090 阶段测试汇总'
doc.core_properties.author='F06 项目'
foot=sec.footer.paragraphs[0];foot.alignment=WD_ALIGN_PARAGRAPH.CENTER
run=foot.add_run('F06  ·  2026-11-05  ·  ');run.font.size=Pt(8)
field=OxmlElement('w:fldSimple');field.set(qn('w:instr'),'PAGE');foot._p.append(field)

def p(text,small=False):
 pp=doc.add_paragraph(text)
 if small:
  pp.paragraph_format.line_spacing=Pt(12.5)
  for r in pp.runs:r.font.size=Pt(9)
 return pp
def heading(text,new=False):
 if new:doc.add_page_break()
 doc.add_heading(text,level=1)
def sub(text):doc.add_heading(text,level=2)
def table(headers,rows,widths=None,font=9,pad=65):
 t=doc.add_table(rows=1,cols=len(headers));t.alignment=WD_TABLE_ALIGNMENT.CENTER;t.autofit=False
 if widths:
  for col,w in zip(t.columns,widths):col.width=Inches(w)
 props=t._tbl.tblPr
 borders=OxmlElement('w:tblBorders')
 for edge in ['top','left','bottom','right','insideH','insideV']:
  e=OxmlElement('w:'+edge);e.set(qn('w:val'),'single');e.set(qn('w:sz'),'4');e.set(qn('w:color'),'D9D9D9');borders.append(e)
 props.append(borders)
 repeat=OxmlElement('w:tblHeader');t.rows[0]._tr.get_or_add_trPr().append(repeat)
 for i,values in enumerate([headers]+list(rows)):
  cells=t.rows[0].cells if i==0 else t.add_row().cells
  tr=t.rows[i]._tr.get_or_add_trPr();tr.append(OxmlElement('w:cantSplit'))
  for j,(cell,value) in enumerate(zip(cells,values)):
   if widths:cell.width=Inches(widths[j])
   cell.vertical_alignment=WD_CELL_VERTICAL_ALIGNMENT.CENTER
   cp=cell._tc.get_or_add_tcPr();m=OxmlElement('w:tcMar')
   for e in ['top','bottom','left','right']:
    x=OxmlElement('w:'+e);x.set(qn('w:w'),str(pad));x.set(qn('w:type'),'dxa');m.append(x)
   cp.append(m)
   shade=OxmlElement('w:shd');shade.set(qn('w:fill'),'DFEAF3' if i==0 else ('F6F8FA' if i%2==0 else 'FFFFFF'));cp.append(shade)
   pp=cell.paragraphs[0];pp.paragraph_format.space_after=Pt(0);pp.paragraph_format.space_before=Pt(0);pp.paragraph_format.line_spacing=Pt(font*1.35)
   snap=OxmlElement('w:snapToGrid');snap.set(qn('w:val'),'0');pp._p.get_or_add_pPr().append(snap)
   pp.alignment=WD_ALIGN_PARAGRAPH.LEFT if j==0 else WD_ALIGN_PARAGRAPH.CENTER
   run=pp.add_run(str(value));run.font.size=Pt(font);run.bold=i==0;run.font.color.rgb=RGBColor(0,0,0)
 spacer=doc.add_paragraph();spacer.paragraph_format.space_after=Pt(3);spacer.paragraph_format.line_spacing=Pt(1);spacer.add_run().font.size=Pt(1)
 return t

# Main report. Each page has one coherent purpose; complete case tables follow.
doc.add_paragraph('F06 融合测试结果与方法问题报告',style='Title')
doc.add_paragraph('阶段汇总截至 2026 年 11 月 5 日',style='Subtitle')
p('F06 已在昆仑芯 P800 与 NVIDIA RTX 4090 两个平台开展验证，历史适配口径为 13 个融合模板变体，覆盖 FP32、FP16、BF16。本报告汇总已完成的公平计时实验、当前实现方法和主要问题，并附两平台逐例结果。这里的“模板数量”不表示完整模型数量。')
sub('主要结果')
p('P800 端到端优化后的自动路径完成 216 例测试，按当轮检查全部通过；其中 47 例使用资源融合或切分，169 例回退到基线。整体相对 eager 的 GPU / 端到端加速比分别为 1.131× / 1.129×。这些数字评价的是含回退的自动选择系统，不能全部计为融合收益。[S1]')
p('RTX 4090 完成 216 例两轮测试，F06 通过 213 例。通过样本相对 eager 的 GPU / 端到端加速比为 2.269× / 1.067×；相对有效 Inductor 对照为 0.956× / 1.962×。大矩阵 MatMul+Bias+Activation 的 36 例优化后全部通过，GPU 与优化 Inductor 基本持平。[S2]')
p('最新 P800 Mul+Sum 单乘并行修复完成 36 例：28 例通过普通有限值检查并计时，4 例通过本轮完整扩展验收。并行化显著降低串行修复开销，但大矩阵 SDNN 分块仍远慢于 eager 与 Inductor，不能替换默认路径。[S4]')
table(['实验批次','覆盖','有效性与限制'],[
 ['历史模板适配','2 平台 × 117 配置','13 模板 × 3 精度 × 3 小尺寸；旧计时口径'],
 ['P800 自动路径','216 配置','12 类 × 3 精度 × 6 尺寸；含 169 例回退'],
 ['RTX 4090 公平复测','216 配置 × 2 轮','F06 213 例通过；Inductor 155 例通过'],
 ['P800 特殊值优化','216 配置','实验模式；逐候选状态见完整数据'],
 ['P800 单乘并行修复','36 配置','仅物理卡 7，串行；完整验收通过 4 例']],[1.55,1.5,4.15],9.5)
sub('当前结论')
p('主要瓶颈已从单纯的 Python 调用包装，进一步定位到 P800 的资源选择、过细分块、标量特殊值修复和同步。复用 Inductor 并不保证资源、分块和生成代码与原生基线一致。当前尚未证明某一种 SDNN 或 cluster 方案在全部尺寸上最优；大矩阵应允许多内核，并以完整调用实测决定是否采用。')
p('各批次的检查集合、设备与源码快照不同，报告分别统计，不跨批次拼接倍率。正在准备的“整矩阵乘法 + 整行归约”新方案尚无纳入本报告的正式 P800 全量结果。',True)

heading('覆盖范围和精度标准',True)
p('历史 13 模板与近期 216 例测试不是同一清单。216 = 12 类 × 3 精度 × 6 尺寸；它增加 LayerNorm 仅 weight / 仅 bias 两类细分，但未覆盖历史的 Add+GELU exact、Add+GELU tanh、Add+SiLU。上述三类不能据此宣称已完成最新的大矩阵公平复测。[S0–S2]')
coverage=[]
for k,v in names.items():coverage.append([codes[k],v,'是' if k not in ['add_layernorm_weight','add_layernorm_bias'] else '未单列','是'])
coverage += [['H11','Add + GELU exact','是','待补测'],['H12','Add + GELU tanh','是','待补测'],['H13','Add + SiLU','是','待补测']]
table(['编号','融合组合','历史 13 模板','近期 216 例'],coverage,[.5,3.75,1.5,1.45],9,pad=45)
p('小矩阵 (M,N,K)：(16,128,64)、(32,256,64)、(64,1024,64)。大矩阵：(128,1024,128)、(512,1024,1024)、(2048,2048,2048)。非 MatMul 组合只使用 M、N；LayerNorm eps=1e-3。',True)
table(['数据类型','atol','rtol','适用范围'],[
 ['FP32','3e-4','3e-4','全部'],['FP16','3e-3','3e-3','全部'],
 ['BF16','3e-2','3e-3','历史小矩阵及非 MatMul'],['BF16','3e-3','3e-3','大矩阵 MatMul 系列']],[1,1,1,4.2],9,pad=45)
p('精度阈值逐例沿用原记录，没有统一放宽 BF16 大 MatMul 阈值。普通有限值、NaN/Inf 分类与传播，以及后续极值/次正规数压力测试分列报告；历史通过不等于后续更广检查全部通过。',True)

heading('当前方法与公平计时',True)
sub('复用 Inductor 的执行流程')
p('静态 FX 图匹配与语义守卫 → 枚举资源和合法切分边界 → 调用 Inductor 编译、分解、调度与自动调优 → 检查精度及真实生成内核 → 捕获前测量 GPU 和普通调用端到端 → 固定计划执行。小矩阵资源候选要求实际单内核；大矩阵允许 SDNN 完整路径，或在 MatMul / Bias 后切分到 cluster。')
p('资源分配采用候选实测，不按固定比例分摊计算。MatMul 优先考察 SDNN，逐元素后处理和归约同时考察 cluster；是否切分取决于整条调用的实测收益，必须计入中间数据读写与同步。')
p('自动选择先拒绝错误候选，再要求候选相对稳定基线在两轮普通调用中均至少降低 3% 端到端耗时，最后按 GPU 时间择优。没有合格资源候选时保留 eager 或 Inductor。回退是一项选择结果，不是融合已执行的证明；设备执行错误必须单列，不能静默算作成功。')
p('项目增加的是模板资格、资源计划、语义舍入/特殊值修复、推理入口和验收逻辑。Inductor 仍负责原生代码生成与搜索；资源选择或 IR 修复会改变最终执行，因而“复用 Inductor”与“具有原生 Inductor 性能”不能等同。')
table(['项目','P800 自动路径','P800 最新专项','RTX 4090'],[
 ['GPU 测量','3 次捕获，42 样本','3 次捕获，18 样本','两轮；7 组 × 20 次重放'],
 ['Graph 调用数','同例各路径相同','同例各路径相同','通常 64；最大 MatMul 为 8'],
 ['普通调用 E2E','2 轮 × 128 次','2 轮 × 128 次','每路径每轮 31 次'],
 ['基线','eager 与 max-autotune-no-cudagraphs','同左','同左'],
 ['稳态计时排除','编译、调优、预热、捕获、输入构造与 H2D/D2H','同左','同左']],[1.1,2.05,2,2.05],9)
p('GPU 时间来自外层 CUDA Graph 重放与设备事件，包含内核内部搬运、同步及切分中间读写。端到端是预置设备输入上的普通同步调用，包含主机提交、输出分配、设备执行及完成等待；它不是 Graph 重放的主机耗时。两者不能相减后直接解释为“纯 CPU 时间”。')
p('加速比 = 基线耗时 ÷ 候选耗时；大于 1 表示候选更快。耗时减少百分比 = (1 − 1/加速比) × 100%。汇总对逐例倍率取几何平均，并标明共同有效样本数。接近 1 的差异按基本持平解释。')
p('P800：OAM，驱动 5.0.21.47，PyTorch 2.9.0+cu128，FlagTree/Triton 3.6.0，实际 XPUDriver arch=3；CUDA API 为 SDK 兼容接口。RTX 4090：PyTorch 2.11.0+cu130，Triton 3.6.0，驱动 615.71.10。数值不作跨平台绝对性能排名。',True)

heading('P800 全组合自动选择结果',True)
p('本页为端到端入口优化后的固定批次 [S1]。216 例当轮验收通过；47 例资源方案、166 例 eager 回退、3 例 Inductor 回退。相比 eager，整体 GPU 耗时减少 11.55%，端到端减少 11.46%。相对有效 Inductor 的倍率较高，主要也包含回退到更快 eager 的收益。',True)
groups=[]
for k in names:
 for size in ['小','大']:
  g=next(g for g in ps['groups'] if g['kind']==k and g['size']==size)
  chosen=sum(v for a,v in g['routes'].items() if a not in ['eager','inductor'])
  groups.append([codes[k],size,f'{chosen}/9',fmt(g['gpu_vs_eager']['geomean']),fmt(g['e2e_vs_eager']['geomean']),fmt(g['gpu_vs_inductor']['geomean']),fmt(g['e2e_vs_inductor']['geomean']),g['gpu_vs_inductor']['cases']])
table(['组合','规模','资源采用','GPU/E','E2E/E','GPU/I','E2E/I','I 有效'],groups,[.52,.42,.8,1.05,1.05,1.05,1.05,.66],8.7,pad=42)
p('E=eager，I=max-autotune-no-cudagraphs；各组 9 例。T01–T12 名称见覆盖页。所有倍率包含自动路径的实际回退。小矩阵资源采用 30/108，大矩阵 17/108；Inductor 有效配对分别为 72 与 82 例。',True)
p('本轮未发现自动路径相对 eager 超过 5% 的 GPU 或端到端退化。该结果不证明所有融合候选更快，也不覆盖后续新增极值压力用例。原测试因其他用户占用而换卡并重测，不能写成全批固定一张物理卡；最新专项则已固定卡 7 串行完成。',True)

heading('169 例回退的原因',True)
p('依据逐例校准记录重新执行选择逻辑，216 例决策均复现。下表为互斥的用例级主要原因；同一用例的不同候选还可能同时存在精度、编译或执行失败，详细多标签记录保存在数据附件。[S5]')
cats=[('gpu_only','GPU 较慢，但有 E2E 合格候选'),('gpu_and_e2e','GPU 较慢且无 E2E 合格候选'),('precision','全部资源候选精度失败'),('compiler','全部资源候选编译失败'),('single_kernel','未满足小矩阵单内核要求')]
table(['主要原因','例数','占回退例数'],[[label,fs['categories'][k],f"{fs['categories'][k]/169*100:.2f}%"] for k,label in cats],[4.7,1.0,1.5],10)
table(['规模','类型','回退/总数','仅 GPU','GPU+E2E','精度','编译','单核'],[
 ['小' if x['size']=='small' else '大',dtypes[x['dtype']],f"{x['fallback']}/{x['total']}",*[x['categories'].get(k,0) for k in ['gpu_only','gpu_and_e2e','precision','compiler','single_kernel']]]
 for x in fs['by_size_dtype']],[.45,.75,.95,.85,1.2,.9,.9,1.2],9)
p('没有发现 GPU 已更快、仅因端到端开销而回退的用例。33 例已满足端到端验收，但 GPU 时间仍慢；97 例同时受到设备执行与普通调用延迟约束。31 例精度失败和 5 例编译失败不能靠改变计时方式解决。')
sub('可以继续优化的部分')
p('大矩阵 FP16 MatMul+Bias+ReLU 512×1024×1024 的切分候选距离 eager 的 GPU 时间约差 4.12%；FP32 Mul+Sum 32×256 约差 9.02%，keepdim 约差 8.85%。这些是历史校准中的接近胜出项，可优先复测，但不能跨批次直接宣布已胜出。')
p('BF16 回退集中在类型提升及舍入路径。后续关闭提升的诊断使 21 例归约故障通过原检查，但它仍可能使用 FP32 算术；30 个 MatMul 关闭提升后仍失败。依据用户选择，默认保持 BF16→FP32 兼容路径。')

heading('RTX 4090 全组合公平结果',True)
p('本页为 11 月 2 日完成的大矩阵优化后完整批次 [S2]，两轮结果合并后统计。F06 213/216 通过，3 个失败均为未改动的 BF16 MatMul+Bias 大矩阵；Inductor 155/216 通过。下面仅统计双方共同通过的项。',True)
grows=[]
for k in names:
 for size in ['small','large']:
  subset=[x for x in rr if x['kind']==k and x['size']==size]
  cols=[]
  for metric in ['gpu_us','e2e_us']:
   for b in ['eager','inductor']:
    values=[num(x[b+'_speedup_'+metric]) for x in subset if num(x[b+'_speedup_'+metric]) is not None]
    cols.append((gm(values),len(values)))
  grows.append([codes[k],'小' if size=='small' else '大',f"{sum(x['f06_status']=='passed' for x in subset)}/9",fmt(cols[0][0]),fmt(cols[2][0]),fmt(cols[1][0]),fmt(cols[3][0]),cols[1][1]])
table(['组合','规模','F06 通过','GPU/E','E2E/E','GPU/I','E2E/I','I 有效'],grows,[.52,.42,.8,1.05,1.05,1.05,1.05,.66],8.7,pad=42)
p('全量相对 eager：GPU 2.269×（213 对）、E2E 1.067×（213 对）。相对 Inductor：GPU 0.956×（155 对）、E2E 1.962×（155 对）。失败项不参与倍率。',True)
p('36 例大矩阵 Activation 补齐原生自动调优后，相对旧 F06 的 GPU / E2E 为 1.164× / 1.100×；相对 Inductor GPU 从 0.831× 提升到 0.997×。24 例形成单计算内核，12 例保留两内核；严格舍入边界仍可能使多内核更合适。',True)

heading('特殊值修复优化的效果与边界',True)
p('NaN/Inf 修复针对运算语义，FP32 也需要检查。当前 P800 指令路径的特殊值处理可能与目标语义不同，因此不能只凭“FP32”就移除修复。双乘与合并、整数分类修复及后续并行改写是本项目加入或改写的兼容逻辑，不应描述为 FlagTree 官方固定实现。[S3、S4]')
p('GELU tanh / SiLU 实验识别完整表达式，保留有限值算术和原始输入，调整受限激活输入及最终乘法操作数顺序，以减少通用标量修复。SiLU 的 +Inf 路径已在目标用例中修复；它不代表所有 BF16 或组合已达到全范围语义等价。')
table(['同配置配对实验','规模','有效配对','GPU 耗时降幅中位数','E2E 降幅中位数'],[
 [names[g['kind']].replace('MatMul + Bias + ','MM+B+'),g['scale'].replace('small','小').replace('large','大'),g['gpu']['n'],f"{g['gpu']['median']:.2f}%",f"{g['e2e']['median']:.2f}%"]
 for g in sp['statistics']['grouped'] if g['rewrite']!='control' and g['dtype']=='all'],[2.65,.45,.7,1.7,1.7],9)
p('上表是相对原修复实现的同配置净耗时变化，不能当作相对 eager 的加速比或纯算术占比。Mul+Sum 两行属于旧双乘优化阶段，后续单乘并行实验单独列出，不能合并统计。',True)
sub('完整验收仍有缺口')
p('216 例实验中，固定候选有 164 例至少存在一个当轮有效方案；33/47 次普通入口验证通过。重新调优候选通过更严格条件且 E2E 合格的编号为 19 个。这些集合不同，不能合并为“216 例优化成功”。SiLU 的目标 +Inf 检查：固定配置 13 项通过，重新调优 14 项通过。')
p('扩展检查发现 Inf×次正规数、最大有限值×最小正规数等边界仍有问题。激活扩展记录还保留有符号零差异；原验收按原比较器执行，不能进一步宣称所有路径逐位 IEEE 等价。专项实验与默认策略保持分离。')
p('逐候选状态、原始耗时、失败模式、实际内核资源和二进制标识均保存在 P800_special_216_cases_all_routes.csv 及其源 summary.json；未测、编译失败、执行风险排除与精度失败分别保留。',True)

primary=sorted([x for x in mm if x['primary']],key=lambda x:x['case'])
for size in ['small','large']:
 heading('MulSum 单乘并行修复 '+('小矩阵' if size=='small' else '大矩阵'),True)
 p('仅物理卡 7，串行完成；精度阈值、输入和 BF16 FP32 计算路径不变。每格以 GPU / E2E 表示，时间单位 µs，倍率为对照时间除以并行单乘时间。† 表示普通有限值可计时但扩展验收失败；F 表示原有限值验收失败而未计时。',True)
 rows=[]
 for x in primary:
  if (x['M']<=64)!=(size=='small'):continue
  rows.append([f"{x['case']:03}",dtypes[x['dtype']],f"{x['M']}×{x['N']}",pair(x['GPU_us'],x['E2E_us'],2),pair(x.get('vs_serial_GPU'),x.get('vs_serial_E2E')),pair(x.get('vs_old_dual_GPU'),x.get('vs_old_dual_E2E')),pair(x.get('vs_eager_GPU'),x.get('vs_eager_E2E')),pair(x.get('vs_inductor_GPU'),x.get('vs_inductor_E2E')),'通过' if x['all_gates'] else ('†' if x['GPU_us'] is not None else 'F')])
 table(['编号','类型','M×N','耗时','对串行','对双乘','对 eager','对 I','验收'],rows,[.38,.55,.78,1.4,.91,.91,.91,.91,.45],8.1,pad=50)
 p('编号 072–080、198–206 为 Mul+Sum；081–089、207–215 为 keepdim 变体。I 为 max-autotune-no-cudagraphs。所有 SDNN 对照采用相同切分；N≤256 为完整融合，N≥1024 采用 256 列分块后 cluster 汇总。64×1024 在本诊断中也采用分块，不满足最初小矩阵单内核部署要求。',True)
 if size=='small':
  p('4 例 FP16 16×128、32×256（含 keepdim）通过完整验收。其余 FP32/BF16 普通输入通过，但极值检查有错；不能作为可部署加速结论。FP16 64×1024 的两个变体未通过原有限值阈值。')
 else:
  p('全部 18 个大矩阵用例均未通过完整验收：12 个 FP32/BF16 有诊断时间，6 个 FP16 未通过普通有限值检查。BF16 原生 Inductor 有限值验收失败，相对该基线倍率留空。')
 p('本轮总计 36 例，28 例候选计时、4 例完整验收通过。目标 Inf×0 / NaN×0 的 12 项乘积检查未发现错误；完整压力测试仍包含其他失败。并行与串行单乘逐例的状态、特殊值与压力测试结果一致。[S4]',True)

heading('大矩阵只有百分之一速度的原因',True)
p('FP32 Mul+Sum 2048×2048 的同次实验数据如下。所有路径通过普通有限值检查，但均存在本轮扩展压力测试失败，因此本页性能仅用于归因。[S4]')
r200=[x for x in mm if x['case']==200]
rmap={'eager':'eager','inductor':'原生优化 Inductor','cluster':'cluster 完整融合','sdnn256_cluster':'SDNN 串行单乘修复','sdnn256_cluster_old_dual':'SDNN 旧双乘修复','sdnn256_cluster_parallel':'SDNN 并行单乘修复'}
table(['路径','GPU µs','E2E µs','GPU 相对 eager','扩展类别错误'],[[rmap[x['route']],fmt(x['GPU_us']),fmt(x['E2E_us']),fmt(x['vs_eager_GPU']),x['class_errors']] for x in r200],[2.3,1.1,1.1,1.45,1.25],9)
p('并行单乘 GPU 为 3357.158 µs，eager 为 34.949 µs，速度比 0.010410×，即 eager 速度的约 1.04%，耗时约为 eager 的 96.06 倍。相对 Inductor 为 0.047391×，耗时约为 21.10 倍。并行修复相对串行单乘快 4.853×，但相对旧双乘只有 0.751×，GPU 耗时增加约 33.10%。')
sub('已定位的调度问题')
p('该 SDNN 对照由项目诊断脚本强制采用 256 列分块、XBLOCK=1 等保守配置，并关闭相应自动调优。2048 列因此产生 8 个 SDNN 部分归约与 1 个 cluster 汇总，共 9 次启动。它不是原生 Inductor 的最优调优结果。cluster 完整融合 157.999 µs 与 Inductor 159.098 µs 接近，直接显示差距主要发生在特定 SDNN 调度路径。')
p('原单乘整数分类进入逐元素标量循环。新方案把 128/256 列按完整缓存行分给 8 个控制核，每核处理 16/32 项，仍需输入就绪与输出汇合两次同步。2048×2048 对应 16384 个行分块，累计 32768 个同步实例；这些实例分布于设备执行，不能解释成全部串行的等待时间。')
p('IR 还显示布局搬运与 SRAM 工作区。该用例输入是 2048×2048 的 A 和长度 2048 的广播向量 b，分别占 16 MiB 和 8 KiB。按列分块不代表 8 次重复读取整个 A；b 的实际读取取决于缓存与实现。32 KiB 工作区分配也不等于实际 DRAM 访问放大 32 倍，真实流量尚未由硬件计数器测得。')
sub('GPU 时间占比的解释')
p('3357.158 / 3400.626 = 98.72% 是 Graph 设备时间与普通调用 E2E 的比值，提示此方案主要受设备侧约束。设备时间同时包含算术、访存、布局转换、同步和等待。PMU 查询未获得可用计数器，插桩也尚不能给出原 F06 的互斥周期分解；真正算术执行占比目前未知。')

heading('当前方案的问题和后续工作',True)
sub('端到端开销已经定位并做了受保护的优化')
p('逐层绕过实验显示，CompiledFxGraph 的 record_function 包装是此前退化的重要来源：单分区约 24–27 µs，双分区提交阶段约 42–43 µs。固定 launcher 的收益较小，剩余开销还包含 AOT 包装、输出分配、设备/流处理和实际执行。项目加入仅在合格推理条件下启用的轻量入口，profiler/debug 条件下恢复原生路径，并保留原生收尾与分配。[S6]')
sub('BF16 计算格式与特殊值修复仍需逐层验证')
p('BF16→FP32 可由 Inductor 的运算提升规则及厂商 FlagTree/XPU lowering 共同产生；项目另有舍入与特殊值兼容处理。不能把最终所有 FP32 运算只归因于一个开关。关闭后端提升不等于形成原生 BF16 算术，实际 IR 必须确认。显式 FP16 替代仅 18/72 例通过原检查，额外范围检查全部失败，当前继续保留 FP32 路径。[S7]')
sub('当前可实施的优化方向')
p('大矩阵 Mul+Sum 应优先比较完整 cluster 融合，与完整乘法后接整行归约的双内核方案。切分时保留乘积 dtype 舍入和最终归约语义，恢复 Inductor 原生块尺寸与坐标下降调优；特殊值判定尽量进入 cluster 向量路径，减少 SDNN 标量循环及跨核同步。')
p('已实现显式启用的切分候选和 vector_multiply 实验入口，并完成 36 项本地单元检查。截至本报告封版，没有经过本地校验的正式 P800 全量性能结果，故不列新方案加速比，也不改默认选择策略。后续实测仍限制同时只使用一张正常 P800 卡。')
p('验收顺序应为：普通有限值 → NaN/Inf 与极值/次正规数 → 实际二进制和资源绑定 → 同一 Graph 协议的 GPU 测量 → 普通调用端到端。任何一项失败均保留原因；即便局部修复较快，也不能写成全链路可用。')
sub('覆盖与证据缺口')
p('需要补齐历史 Add+GELU exact/tanh、Add+SiLU 在大小矩阵与新公平口径下的测试；还需修复 RTX 4090 大矩阵 BF16 bias-only 的 3 例失败。P800 完整模型吞吐、冷缓存、H2D/D2H 与并发负载不在本轮算子链路结果范围内。')
p('已完成专项的结果、二进制绑定与哈希已保存本地，专项共校验 648 个绑定文件，远端原测试目录已清理。新切分实验目录尚用于待完成实验，不应在报告中宣称它也已经清理。',True)

heading('结果附件与复查索引',True)
p('后续附录逐例列出 P800 自动路径与 RTX 4090 各 216 个配置的时间、加速比和状态。所有原始精度数值、采样及失败原因在随附数据包中保留；PDF 表格只做显示舍入，计算采用原始值。')
table(['附件','内容'],[
 ['P800_216_cases.csv','216 个自动选择结果，含实际路线、回退原因和两基线耗时'],
 ['RTX4090_216_cases.csv','216 个配置，两轮耗时、状态、误差及倍率'],
 ['P800_MulSum_36_cases_all_routes.csv','36 个配置 × 6 路径，共 216 行；包括不合格与未计时项'],
 ['P800_special_216_cases_all_routes.csv',f"216 个配置的 {len(srows)} 条候选/基线记录"],
 ['P800_fallback_216_classification.csv','216 例分类及候选证据；其中 169 例回退'],
 ['sources 与 source_manifest.json','固定批次原始 JSON、验证记录、清理回执及 SHA256']],[3.45,3.75],9)
table(['编号','来源与适用结论'],[
 ['S0','用户提供的 2026-10-20 周报：历史 13 模板、2 平台适配口径'],
 ['S1','2026-11-03/f06-p800-e2e-opt：自动路径全量 216 例'],
 ['S2','2026-11-02/f06-large-matmul-4090：两轮全量及 36 例优化'],
 ['S3','2026-11-04/f06-p800-special-opt：特殊值修复实验'],
 ['S4','2026-11-05/f06-p800-parallel-mul：单卡单乘并行 36 例'],
 ['S5','2026-11-03/f06-p800-fallback-analysis 与 fallback-root-cause'],
 ['S6','2026-11-03/f06-p800-e2e-diagnosis：端到端逐层绕过实验'],
 ['S7','2026-11-03/f06-p800-bf16-formats：替代格式诊断'],
 ['S8','2026-11-03/f06-p800-counter-research：计数器与插桩限制']],[.45,6.75],9)
p('来源目录均位于项目 work/ 下。数据包复制了报告使用的结构化来源，不依赖远端测试目录。S0 是历史文档来源；117/117 的旧通过记录只适用于其原始小尺寸与检查集合，没有迁移成当前全量通过率。',True)
p('附录读法：E=eager，I=max-autotune-no-cudagraphs，A=P800 自动路径，F=RTX F06；时间/倍率均按 GPU / E2E 排列。短划线表示没有有效测量或有效配对。P800“回 E/回 I”不计融合成功。组合编号见覆盖页；非 MatMul 的 K 显示为空。',True)

# Appendices preserve the complete measured corpus in the report itself.
sec=doc.add_section(WD_SECTION_START.NEW_PAGE)
sec.orientation=WD_ORIENT.LANDSCAPE;sec.page_width=Inches(11);sec.page_height=Inches(8.5)
sec.top_margin=sec.bottom_margin=Inches(.5);sec.left_margin=sec.right_margin=Inches(.5)
sec.footer_distance=Inches(.2)
for platform,records in [('P800',sorted(prows,key=lambda x:x['ordinal'])),('RTX 4090',rr)]:
 for start in range(0,len(records),24):
  if not (platform=='P800' and start==0):doc.add_page_break()
  doc.add_heading(f'附录 {platform} 逐例结果 {start+1} 至 {min(start+24,len(records))}',level=1)
  p('时间单位 µs；各数值对依次为 GPU / 普通调用 E2E。E=eager，I=优化 Inductor。倍率基于同批次共同有效记录。',True)
  rows=[]
  for idx,x in enumerate(records[start:start+24],start):
   if platform=='P800':
    route=routes[x['selected']];candidate='auto'; rg=x['gpu_vs_eager'];re=x['e2e_vs_eager'];ig=x['gpu_vs_inductor'];ie=x['e2e_vs_inductor'];oid=x['ordinal'];status='当轮通过'
   else:
    route='F06';candidate='f06';rg=x.get('eager_speedup_gpu_us');re=x.get('eager_speedup_e2e_us');ig=x.get('inductor_speedup_gpu_us');ie=x.get('inductor_speedup_e2e_us');oid=idx;status='通过' if x['f06_status']=='passed' else '精度失败'
   shape='×'.join(str(x[a]) for a in ['M','N','K'] if x.get(a) not in [None,'','None'])
   rows.append([f'{oid:03}',codes[x['kind']],dtypes[x['dtype']],shape,route,pair(x.get('eager_gpu_us'),x.get('eager_e2e_us'),2),pair(x.get('inductor_gpu_us'),x.get('inductor_e2e_us'),2),pair(x.get(candidate+'_gpu_us'),x.get(candidate+'_e2e_us'),2),pair(rg,re,3),pair(ig,ie,3),status])
  table(['编号','组合','类型','M×N×K','实际路线','E 时间','I 时间','A/F 时间','相对 E','相对 I','状态'],rows,[.4,.4,.5,1.1,1.0,1.14,1.14,1.14,1.16,1.16,.86],8.1,pad=42)
  p('P800 当轮通过未包含后续 225 类乘积及 4096 对有限值的扩展压力验收；后续 Mul+Sum 专项见正文。RTX 失败项及无效 Inductor 配对均未赋予加速比。',True)

# Use the user's terminology while retaining the actual execution route in results.
replacements=[
 ('169 例回退的原因','169 例复用原生路径的原因'),
 ('回退到基线','复用原生执行路径'),('169 例回退','169 例复用原生路径'),
 ('166 例 eager 回退、3 例 Inductor 回退','166 例 eager、3 例原生 Inductor'),
 ('含回退','含原生路径复用'),('实际回退','实际原生执行路径'),
 ('回退例数','原生路径例数'),('回退/总数','原生/总数'),
 ('回退原因','原生路径采用原因'),('BF16 回退','BF16 原生路径采用'),
 ('回退是一项选择结果','原生路径复用是一项选择结果'),
 ('回退到更快 eager','复用更快 eager'),
 ('P800“回 E/回 I”','P800“原生 E/原生 I”'),
 ('原生路径复用到更快 eager','复用更快 eager'),
 ('依据用户选择，默认保持','当前默认保持'),
 ('不能写成全批固定一张物理卡','全批并未固定同一张物理卡'),
 ('不应在报告中宣称它也已经清理','尚未清理'),
 ('用户提供的 2026-10-20 周报','2026-10-20 项目周报'),
 ('仅因端到端开销而回退的用例','仅因端到端开销而采用原生路径的用例'),
]
allparas=list(doc.paragraphs)+[p for t in doc.tables for row in t.rows for c in row.cells for p in c.paragraphs]
for para in allparas:
 for run in para.runs:
  value=run.text
  for a,b in replacements:value=value.replace(a,b)
  if value!=run.text:run.text=value
 for b in list(para._p.iter(qn('w:pBdr'))):b.getparent().remove(b)
for style in doc.styles:
 for b in list(style._element.iter(qn('w:pBdr'))):b.getparent().remove(b)
docpath=OUT/'F06测试结果与方法问题报告_20261105.docx'
doc.save(docpath)
(DATA/'README.txt').write_text('F06 阶段报告数据包，2026-11-05\n\n所有时间单位为微秒，speedup=基线耗时/候选耗时。\nP800 216 自动路径含169回退，不能全部计为融合成功。\nRTX 4090 213/216 F06通过，Inductor155/216通过。\n单乘并行专项36例中28例计时、4例完整验收通过。all_gates=false的时间只用于诊断。\nP800特殊值实验的status及special_passed必须同时查看，不能仅按有时间字段判定正确。\n不同批次不混合求加速比；后续扩展检查不由历史passed替代。\nCSV使用UTF-8 BOM，JSON保留原始采样与详细原因。\nsource_manifest.json列出输入证据的SHA256。\n')
checks={'p800_cases':len(prows),'rtx_cases':len(rr),'special_cases':len(sp['cases']),'special_route_rows':len(srows),'mul_cases':len(primary),'mul_route_rows':len(mm),'mul_all_gates':sum(x['all_gates'] for x in primary),'p800_fallback':sum(x['fallback'] for x in prows),'rtx_pass':sum(x['f06_status']=='passed' for x in rr),'source_files':len(manifest)}
assert checks['p800_cases']==checks['rtx_cases']==216 and checks['mul_cases']==36 and checks['mul_all_gates']==4 and checks['p800_fallback']==169 and checks['rtx_pass']==213
for x in prows:
 for metric in ['gpu','e2e']:
  for base in ['eager','inductor']:
   value=x[metric+'_vs_'+base]
   if value is not None:assert math.isclose(value,x[base+'_'+metric+'_us']/x['auto_'+metric+'_us'],rel_tol=1e-12)
(OUT/'validation.json').write_text(json.dumps(checks,indent=2))
print(json.dumps(checks,ensure_ascii=False));print(docpath)
