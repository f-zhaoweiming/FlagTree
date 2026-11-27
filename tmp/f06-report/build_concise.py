from pathlib import Path
import json, csv, math, statistics
from docx import Document
from docx.shared import Mm, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_CELL_VERTICAL_ALIGNMENT
from docx.oxml import OxmlElement
from docx.oxml.ns import qn

ROOT=Path('/home/f/Documents/FlagTree')
OUT=ROOT/'output/f06-20261105'
P800=json.loads((ROOT/'work/2026-11-03/f06-p800-e2e-opt/summary.json').read_text())
FALL=json.loads((ROOT/'work/2026-11-03/f06-p800-fallback-root-cause/summary.json').read_text())
RTX=list(csv.DictReader((OUT/'data/RTX4090_216_cases.csv').open()))
MUL=list(csv.DictReader((OUT/'data/P800_MulSum_36_cases_all_routes.csv').open()))
SPEC=json.loads((ROOT/'work/2026-11-04/f06-p800-special-opt/summary.json').read_text())

KINDS=[
 ('matmul_bias','T01','MatMul + Bias'),
 ('matmul_bias_relu','T02','MatMul + Bias + ReLU'),
 ('matmul_bias_gelu_exact','T03','MatMul + Bias + GELU exact'),
 ('matmul_bias_gelu_tanh','T04','MatMul + Bias + GELU tanh'),
 ('matmul_bias_silu','T05','MatMul + Bias + SiLU'),
 ('add_softmax','T06','Add + Softmax'),
 ('add_layernorm','T07','Add + LayerNorm 无仿射'),
 ('add_layernorm_weight','T08','Add + LayerNorm 仅 weight'),
 ('add_layernorm_bias','T09','Add + LayerNorm 仅 bias'),
 ('add_layernorm_affine','T10','Add + LayerNorm 仿射'),
 ('mul_sum','T11','Mul + Sum'),
 ('mul_sum_keepdim','T12','Mul + Sum keepdim'),
]
def gm(xs):
    xs=[float(x) for x in xs if x not in ('',None)]
    return math.exp(statistics.mean(math.log(x) for x in xs)) if xs else None
def f(v,n=2): return '—' if v is None else f'{v:.{n}f}'
def pair(a,b,n=2): return '—' if a is None else f'{a:.{n}f} / {b:.{n}f}'
def find_group(kind,size):
    return next(g for g in P800['groups'] if g['kind']==kind and g['size']==size)

doc=Document()
sec=doc.sections[0]
sec.page_width=Mm(210); sec.page_height=Mm(297)
sec.top_margin=sec.bottom_margin=Mm(17); sec.left_margin=sec.right_margin=Mm(18)
sec.header_distance=sec.footer_distance=Mm(8)
for n in ['Normal','Title','Subtitle','Heading 1','Heading 2']:
    s=doc.styles[n]; s.font.name='Noto Sans CJK SC'; s.font.color.rgb=RGBColor(0,0,0)
    s._element.get_or_add_rPr().get_or_add_rFonts().set(qn('w:eastAsia'),'Noto Sans CJK SC')
    s.paragraph_format.space_after=Pt(5); s.paragraph_format.line_spacing=Pt(14.5)
    for b in list(s._element.iter(qn('w:pBdr'))): b.getparent().remove(b)
doc.styles['Normal'].font.size=Pt(10)
doc.styles['Title'].font.size=Pt(22)
doc.styles['Title'].paragraph_format.line_spacing=Pt(28)
doc.styles['Subtitle'].font.size=Pt(10)
doc.styles['Heading 1'].font.size=Pt(15)
doc.styles['Heading 1'].paragraph_format.space_before=Pt(10)
doc.styles['Heading 1'].paragraph_format.space_after=Pt(5)
doc.styles['Heading 2'].font.size=Pt(11.5)
doc.styles['Heading 2'].paragraph_format.space_before=Pt(7)
doc.styles['Heading 2'].paragraph_format.space_after=Pt(3)
doc.core_properties.title='F06 融合测试结果（精简版）'
doc.core_properties.subject='P800 与 RTX 4090 阶段测试摘要'
doc.core_properties.author='F06 项目'

foot=sec.footer.paragraphs[0]; foot.alignment=WD_ALIGN_PARAGRAPH.CENTER
r=foot.add_run('F06 · 2026-11-05 · 精简版  ·  '); r.font.size=Pt(8)
field=OxmlElement('w:fldSimple'); field.set(qn('w:instr'),'PAGE'); foot._p.append(field)

def p(text,size=10,after=5,bold_lead=None):
    par=doc.add_paragraph(); par.paragraph_format.space_after=Pt(after)
    if bold_lead:
        run=par.add_run(bold_lead); run.bold=True; run.font.size=Pt(size)
    run=par.add_run(text); run.font.size=Pt(size)
    return par
def bullet(text,size=10):
    par=doc.add_paragraph(style='List Bullet'); par.paragraph_format.space_after=Pt(3)
    par.paragraph_format.line_spacing=Pt(14)
    run=par.add_run(text); run.font.size=Pt(size)
def h(text,level=1):
    par=doc.add_heading(text,level=level)
    par.paragraph_format.keep_with_next=True
    return par
def table(headers,rows,widths,font=8.5,pad=45):
    t=doc.add_table(rows=1,cols=len(headers)); t.alignment=WD_TABLE_ALIGNMENT.CENTER; t.autofit=False
    for col,w in zip(t.columns,widths): col.width=Mm(w)
    borders=OxmlElement('w:tblBorders')
    for edge in ['top','left','bottom','right','insideH','insideV']:
        e=OxmlElement('w:'+edge); e.set(qn('w:val'),'single'); e.set(qn('w:sz'),'4'); e.set(qn('w:color'),'D9D9D9'); borders.append(e)
    t._tbl.tblPr.append(borders)
    trPr=t.rows[0]._tr.get_or_add_trPr(); trPr.append(OxmlElement('w:tblHeader'))
    for i,vals in enumerate([headers]+list(rows)):
        cells=t.rows[0].cells if i==0 else t.add_row().cells
        tr=t.rows[i]._tr.get_or_add_trPr(); tr.append(OxmlElement('w:cantSplit'))
        for j,(cell,val) in enumerate(zip(cells,vals)):
            cell.width=Mm(widths[j]); cell.vertical_alignment=WD_CELL_VERTICAL_ALIGNMENT.CENTER
            tcPr=cell._tc.get_or_add_tcPr(); mar=OxmlElement('w:tcMar')
            for e in ['top','bottom','left','right']:
                x=OxmlElement('w:'+e); x.set(qn('w:w'),str(pad)); x.set(qn('w:type'),'dxa'); mar.append(x)
            tcPr.append(mar)
            shd=OxmlElement('w:shd'); shd.set(qn('w:fill'),'DFEAF3' if i==0 else ('F6F8FA' if i%2==0 else 'FFFFFF')); tcPr.append(shd)
            par=cell.paragraphs[0]; par.paragraph_format.space_after=Pt(0); par.paragraph_format.space_before=Pt(0); par.paragraph_format.line_spacing=Pt(font*1.25)
            run=par.add_run(str(val)); run.font.size=Pt(font); run.bold=(i==0); run.font.color.rgb=RGBColor(0,0,0)
            par.alignment=WD_ALIGN_PARAGRAPH.LEFT if j in (0,1) else WD_ALIGN_PARAGRAPH.CENTER
    sp=doc.add_paragraph(); sp.paragraph_format.space_after=Pt(2); sp.paragraph_format.line_spacing=Pt(1); sp.add_run().font.size=Pt(1)
    return t

# ---------------- page 1 ----------------
doc.add_paragraph('F06 融合测试结果与方法问题',style='Title')
doc.add_paragraph('精简版 · 截止 2026 年 11 月 5 日 · 详细逐例结果见数据包',style='Subtitle')
p('F06 已在昆仑芯 P800 与 NVIDIA RTX 4090 两个平台验证。历史口径为 13 个融合模板变体，覆盖 FP32、FP16、BF16；近期公平复测按 12 类组合、3 种精度、6 档尺寸展开。下文倍率均为“基线耗时 ÷ 候选耗时”，大于 1 表示候选更快。',10.2,7)
h('结论摘要',1)
bullet('P800 自动路径 216 例当轮验收全部通过，但仅 47 例采用融合/切分，169 例复用原生执行路径。整体相对 eager 的 GPU/端到端倍率为 1.131×/1.129×，不能全部算作融合收益。')
bullet('RTX 4090 的 F06 通过 213/216；相对 eager 的 GPU/端到端倍率为 2.269×/1.067×，相对有效 Inductor 配对为 0.956×/1.962×。GPU 已接近 Inductor，端到端优势主要来自调用路径。')
bullet('最新 P800 Mul+Sum 专项完成 36 例：28 例可计时，仅 4 例通过完整扩展验收。并行单乘比串行单乘快 2.6–5.6×，但通过完整验收的 4 例仍慢于旧双乘；大矩阵不可部署。')
bullet('主要瓶颈已不是单纯的 Python 调用包装，而是 P800 的资源选择、过细分块、归约代码生成、特殊值标量修复和同步。')

h('测试范围与口径',1)
table(['批次','规模','用途与限制'],[
 ['历史模板适配','2 平台 × 117 配置','13 模板 × 3 精度 × 3 小尺寸；旧计时口径'],
 ['P800 自动路径','216 例','12 类 × 3 精度 × 6 尺寸；含 169 例原生路径'],
 ['RTX 4090 公平复测','216 例 × 2 轮','F06 213 通过；Inductor 155 通过'],
 ['P800 特殊值实验','216 例','实验模式；逐候选状态另存'],
 ['P800 Mul+Sum 单乘专项','36 例','固定卡 7 串行；4 例完整验收通过'],
],[28,42,104],8.5)
p('近期 216 例由 12 类组合构成，其中 LayerNorm 拆成无仿射、仅 weight、仅 bias、仿射四个变体。历史 13 模板还包含 Add+GELU exact、Add+GELU tanh、Add+SiLU，这三类尚未纳入新口径全量复测，因此不能宣称 15 个组合都已完成公平复测。',8.5,4)
p('精度阈值逐例沿用原记录：FP32 3e-4/3e-4，FP16 3e-3/3e-3，BF16 3e-2/3e-3；大矩阵 BF16 MatMul 为 3e-3/3e-3。没有统一放宽阈值。普通有限值、NaN/Inf 语义、极值/次正规数压力测试分列判定。',8.5,2)

doc.add_page_break()
# ---------------- page 2: results ----------------
h('核心结果',1)
p('平台总体结果如下；“I 有效”表示与 Inductor 双方都有有效配对的样本数。',9.2)
table(['平台与对照','GPU 倍率','端到端倍率','有效样本'],[
 ['P800 自动路径 / eager','1.131×','1.129×','216'],
 ['P800 自动路径 / Inductor','3.489×','2.788×','154'],
 ['RTX 4090 F06 / eager','2.269×','1.067×','213'],
 ['RTX 4090 F06 / Inductor','0.956×','1.962×','155'],
],[52,34,34,30],9)
p('P800 相对 Inductor 的 3.489× 包含 169 例复用更快 eager 的收益，不是纯融合加速。RTX 的 GPU 0.956× 表示 F06 平均略慢于 Inductor，端到端 1.962× 主要来自更薄的调用路径。',8.5,5)

h('分组合倍率（相对 eager）',2)
rows=[]
for kind,code,name in KINDS:
    ps=find_group(kind,'小'); pl=find_group(kind,'大')
    rs=[x for x in RTX if x['kind']==kind and x['size']=='small']
    rl=[x for x in RTX if x['kind']==kind and x['size']=='large']
    rows.append([
        code+' '+name,
        pair(ps['gpu_vs_eager']['geomean'],ps['e2e_vs_eager']['geomean']),
        pair(pl['gpu_vs_eager']['geomean'],pl['e2e_vs_eager']['geomean']),
        pair(gm([x['eager_speedup_gpu_us'] for x in rs]),gm([x['eager_speedup_e2e_us'] for x in rs])),
        pair(gm([x['eager_speedup_gpu_us'] for x in rl]),gm([x['eager_speedup_e2e_us'] for x in rl])),
    ])
table(['组合','P800 小','P800 大','RTX 小','RTX 大'],rows,[55,31,31,31,31],7.1,pad=35)
p('每格为 GPU / 端到端倍率；P800 为自动路径实际结果，RTX 为 F06 通过样本。不同平台的绝对值不可直接比较，只比较各自相对 eager 的倍率。P800 归约类多数倍率接近 1，表示最终选择了原生路径。RTX 的 T01 大矩阵 F06 通过 6/9，其余组合均为 9/9；未通过项不参与倍率。',7.8,2)
p('完整逐例数据（216 例 × 两平台、216 条特殊值路线、36 例 Mul+Sum 六路径）见附件。',8,2)

doc.add_page_break()
# ---------------- page 3: method ----------------
h('当前方法',1)
h('1. 复用 Inductor，而不是另写一套算子库',2)
p('静态 FX 图匹配与语义守卫 → 枚举资源与合法切分边界 → 调用 Inductor 做分解、调度、代码生成和自动调优 → 验证真实生成内核和数值 → 固定计划执行。项目增加的是模板资格、资源计划、语义修复、推理入口和验收逻辑；Inductor 仍负责原生代码生成与搜索。')
h('2. 按融合组合和尺寸选资源',2)
p('MatMul 优先考察 SDNN；逐元素后处理和归约同时考察 cluster。小矩阵候选要求实际只生成一个内核；大矩阵允许 SDNN 完整路径，或在 MatMul / Bias 后切分给 cluster，允许多内核。是否切分必须按整条调用实测，计入中间张量读写和同步。')
h('3. 先过正确性，再谈性能',2)
p('候选依次通过普通有限值、输入更新、重复调用、NaN/Inf 分类与传播、极值/次正规数压力检查；随后核对实际执行的二进制和资源绑定。未通过项保留原因，不进入加速比。')
h('4. 端到端门槛与回退',2)
p('候选必须在两轮普通调用中均比稳定基线降低至少 3% 端到端耗时，最后再按 GPU 时间择优；没有合格候选时复用 eager 或原生 Inductor。回退是选择结果，不是融合成功，也不能静默算作收益。')

h('公平计时',1)
table(['项目','做法'],[
 ['GPU 时间','CUDA Graph 重放 + 设备事件；同一用例各路径使用相同调用次数'],
 ['端到端','预置设备输入上的普通同步调用，包含提交、输出分配、执行和完成等待'],
 ['排除项','编译、调优、预热、Graph 捕获、输入构造与 H2D/D2H'],
 ['基线','eager 与 max-autotune-no-cudagraphs；同批次、同输入、同卡比较'],
 ['汇总','逐例倍率几何平均；标注有效配对数，接近 1 视为持平'],
],[38,136],8.5)
p('GPU 时间包含内核内部搬运、同步和切分中间读写，不能当作“纯算术时间”。端到端也不是 Graph 重放的主机耗时；两者不能相减后直接解释为 CPU 开销。',8.5,2)
p('P800：驱动 5.0.21.47，PyTorch 2.9.0+cu128，FlagTree/Triton 3.6.0，XPUDriver arch=3。RTX 4090：PyTorch 2.11.0+cu130，Triton 3.6.0，驱动 615.71.10。',8,2)

doc.add_page_break()
# ---------------- page 4: issues ----------------
h('主要问题与归因',1)
h('P800 自动路径：169 例复用原生执行',2)
table(['主要原因','例数','说明'],[
 ['GPU 较慢，但端到端已合格','33','设备侧仍差，继续优化才有机会被选中'],
 ['GPU 较慢，端到端也未达标','97','不能只归因于主机调用开销'],
 ['全部资源候选精度失败','31','以 BF16 为主'],
 ['全部资源候选编译失败','5','BF16 向量元素类型不匹配'],
 ['小矩阵未满足单内核','3','实际生成了两个内核'],
],[54,18,102],8.3)
p('没有发现“GPU 已更快、仅因端到端门槛而回退”的用例。169 例中 166 例最终执行 eager，3 例执行原生 Inductor。',8.2,4)

h('归约与特殊值修复是当前最大设备侧瓶颈',2)
p('Softmax 和四种 LayerNorm 的 90 例 P800 自动路径全部复用原生执行。多数正确候选已是单内核，但设备耗时仍明显落后；66 例可与原生 Inductor 配对的样本中，60 例 GPU 时间差在 5% 内。这说明瓶颈更可能在共同的 Inductor/P800 归约代码生成与调优，而不是 F06 的 Python 分发层。')
p('NaN/Inf 修复需要保留，但原实现会进入逐元素标量循环和同步。GELU-tanh、SiLU 的整段 SDNN 路径因此付出很高代价；并行化后相对串行单乘快 2.6–5.6×，但大矩阵仍慢于基线。Mul+Sum 单乘专项 36 例中只有 4 例通过完整验收。')

h('大矩阵 Mul+Sum 的 1% 倍率是调度问题',2)
p('FP32 2048×2048 的 SDNN 对照被强制为 256 列分块、XBLOCK=1，并关闭自动调优：8 个 SDNN 部分归约加 1 个 cluster 汇总，共 9 次启动；整张矩阵对应 16384 个行分块。GPU 为 3357 µs，eager 为 34.9 µs，Inductor 为 159.1 µs。该结果说明这条强制切分路径不可用，不能代表 Inductor 融合的一般能力。')

h('BF16 与覆盖缺口',2)
bullet('P800 大矩阵 BF16 的 36 例全部复用原生路径：25 例精度失败、5 例编译失败、6 例 GPU 较慢。关闭类型提升不等于形成原生 BF16 算术，实际 IR 仍需逐层确认。')
bullet('RTX 4090 仍有 3 例未改动的 BF16 MatMul+Bias 大矩阵失败，未计入加速比。')
bullet('历史 Add+GELU exact/tanh、Add+SiLU 尚未按新口径测大小矩阵；报告不把这 3 类算作已完成。')

doc.add_page_break()
# ---------------- page 5: next ----------------
h('结论与下一步',1)
h('结论',2)
bullet('小矩阵已有可用收益，尤其是 MatMul+激活；P800 的融合收益与调用路径收益必须分开表述。')
bullet('大矩阵不应强制单一资源或过细分块。应比较“完整 cluster 融合”和“完整乘法 kernel + 整行归约 kernel”等方案，保留 Inductor 的块尺寸与坐标下降调优。')
bullet('归约和特殊值语义仍是设备侧主瓶颈；在修正前，大矩阵 Mul+Sum 与复杂激活不应进入默认路径。')
bullet('当前默认策略未改；BF16→FP32 计算路径、精度阈值和 NaN/Inf 验收均保持不变。')

h('优先工作',2)
table(['优先级','工作项','完成标准'],[
 ['P0','移除 256 列/XBLOCK=1 的强制 SDNN 分块，比较大矩阵整图融合与双内核方案','同一输入、同卡、完整调用实测；不再以 1% 级强制路径作结论'],
 ['P0','修整归约代码生成与特殊值向量化，减少标量循环和跨核同步','Softmax/LayerNorm 与 Mul+Sum 通过原精度和特殊值检查，并接近或超过基线'],
 ['P1','补齐历史 Add+GELU exact/tanh、Add+SiLU 的大小矩阵公平复测','2 平台 × 3 精度 × 6 尺寸，统一 Graph/E2E 口径'],
 ['P1','修复 P800 BF16 大矩阵精度/编译失败与 RTX 3 例失败','逐例保留失败阶段，不放宽阈值'],
],[15,85,74],8.3)
p('后续 P800 实验仍只占用一张正常卡；测试结果写回本地并校验后，再清理远端测试目录。',8.5,2)

h('证据与复现',1)
p('本精简版保留核心结论、方法和问题；完整逐例时间、候选状态、失败原因、实际内核与二进制绑定在数据包中。详细版报告共 30 页，含两平台 216 例逐页附录。',9)
table(['文件','内容'],[
 ['F06测试结果与方法问题报告_20261105.pdf / .docx','完整版：方法、问题、两平台逐例附录'],
 ['P800_216_cases.csv','P800 216 例自动路径与两基线'],
 ['RTX4090_216_cases.csv','RTX 216 例两轮结果'],
 ['P800_special_216_cases_all_routes.csv','216 例特殊值实验的 982 条路线'],
 ['P800_MulSum_36_cases_all_routes.csv','36 例 × 6 路径，含不合格与未计时项'],
 ['source_manifest.json + sources/','58 个来源文件及 SHA256'],
],[68,106],7.8)
p('所有倍率只使用同批次共同有效样本；未通过精度、编译、执行或单内核约束的项不参与加速比。报告中的“通过”只代表对应检查集合，不代表所有后续扩展压力测试均通过。',8,2)

docpath=OUT/'F06测试结果_精简版_20261105.docx'
doc.save(docpath)
print(docpath)
