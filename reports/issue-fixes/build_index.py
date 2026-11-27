"""Build the repository issue reference from archived evidence and dated updates."""
import collections
import hashlib
import html
import json
import os
import re
import stat
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
ARCHIVE = ROOT / 'work/2026-11-13/flagtree-101-review'
OLD = ROOT / 'work/2026-11-12/f06-issue-pr-review'
DIAG = ROOT / 'work/2026-11-19/flaggems-diagnosis'
MODEL = ROOT / 'review-local/f06-inductor-model-20261119'
REPAIR = ROOT / 'work/2026-11-19/flaggems-repair'
COPY_ORIGIN = ROOT / 'work/2026-11-19/flaggems-copy-origin'
OFFICIAL_GEMS = ROOT / 'work/2026-11-19/flaggems-official'
GEMS_FIXES = ROOT / 'work/2026-11-19/flaggems-official-fixes'
ATTENTION_OPT = ROOT / 'work/2026-11-19/flaggems-attention-opt'
MINILM_RESIDUAL = ROOT / 'work/2026-11-19/minilm-residual'
PUBLIC_REVIEW = OUT / '13项提交与厂商包归属-20261119.md'
MAIN_RETEST = OUT / '官方main-P800复测-20261119.md'
SUBMITTED_PAIRS = {'F21': (1458, 1465), 'F22': (1640, 1641), 'F24': (1642, 1643),
                   'U07': (1651, 1652), 'N01': (1645, 1646)}
WITHOUT_FIX_PR = {'F17', 'F20', 'U05', 'U09', 'U12', 'U13', 'U20', 'N04'}
GITHUB_REPO = 'https://github.com/flagos-ai/FlagTree'
source = json.loads((ARCHIVE/'index.json').read_text())
old = json.loads((OLD/'index.json').read_text())
group_names = {'flagtree':'FlagTree 源码', 'xpu-packages':'XPU 工具包与预编译链路',
               'local':'本地 F06 与集成', 'ledger':'历史线索与未解决项'}

def ref(label, path):
    path = Path(path)
    if not path.is_absolute(): path = ROOT / path
    assert path.exists(), path
    return {'label':label,'path':str(path.resolve())}

def link(r):
    return f"[{r['label']}](<{r['path']}>)"

items=[]
for row in source['items']:
    item={k:row.get(k,'') for k in ('id','title','component','solution','evidence')}
    item.update(group=group_names[row['group']], status=row['solution_type'],
                date='2026-11-13', historical_status=row.get('declared_historically',''),
                baseline='官方固定提交 e6a8799cc；本地条目为历史 v51e 对照。',
                related=[], notes=[], refs=[], fix_paths=[])
    labels={'issue':'Issue 说明','pr':'修复或规避方案','reproduce':'复现脚本',
            'evidence':'原始证据','before':'修复前实测','after':'修复后实测','patch':'审阅补丁'}
    for kind,p in row['links'].items():
        item['refs'].append(ref(labels.get(kind,kind),ARCHIVE/p))
    if row['group']=='ledger':
        item['notes'].append('保留用于排重和查找；不计作已根修。状态与证据来自 11 月 13 日复筛。')
        if (OLD/row['id']/'ISSUE.md').exists():
            item['refs'].append(ref('历史问题说明',OLD/row['id']/'ISSUE.md'))
    patch=old.get('patches',{}).get(row['id'],{})
    for p in patch.get('paths',[]):
        if p[0] not in item['fix_paths']:item['fix_paths'].append(p[0])
    if patch:
        item['notes'].append('历史本地差异：'+patch['before']+' → '+patch['after']+
            ('（tree 指当时工作树快照，不是已合入提交）。' if patch['after']=='tree' else '；不表示官方已合并。'))
    for r in row.get('source_references',[]):
        if r['path'] not in item['fix_paths']:item['fix_paths'].append(r['path'])
        item['refs'].append({'label':f"固定基线源码 {r['path']}:{r['line']}",'path':r['url']})
    p=ARCHIVE/row.get('links',{}).get('issue','__missing__')
    if p.is_file():
        patch_path=p.parent/'review.patch'
        if patch_path.exists() and not any(r['path']==str(patch_path.resolve()) for r in item['refs']):
            item['refs'].append(ref('历史审阅差异',patch_path))
    if row['id'] in ('F23','F30','F31'):
        item['status']='本地绕行，回归通过'
        item['notes'].append('通过准入保护或选择替代实现交付；不能据此宣称被排除的底层路径已修好。')
    if row['id'] in ('F41','N24'):
        item['status']='计时保护已修，底层异常未根修'
        item['notes'].append('修复的是无效计时的消费、重测和排名策略；设备时间戳异常的底层根因未因此闭环。')
    if row['id']=='U07':
        item.update(date='2026-11-14')
        item['notes'].append('11 月 14 日复核：CPU 准备输入和参考时，物理卡 3、7 的 65536 个 SDNN 输出通过，最大绝对误差 1.5434e-5。原生 linspace/SiLU/归约仍可同步超时，属于另列的设备执行状态诊断 D01。SiLU 修复只覆盖已测 [-3,3] 范围。')
        item['refs'].append(ref('11 月 14 日数值及超时复核','work/2026-11-14/U07-timeout-diagnosis/README.md'))
        item['related']=['D01','U25']
    if row['id']=='U09':
        item.update(date='2026-11-14')
        item['notes'].append('11 月 14 日补充：两卡的 3 份原始 TTIR 与 3 份分支交换对照均完成编译；generic1 的单 program 65536 值数值通过。完整 2048×2048 grid 未取得数值结果，简化样例仍可遇到 unroll/vectorize 的后续问题；不能扩大为所有分支或整图均已修复。')
        item['refs'].append(ref('11 月 14 日审查与边界','work/2026-11-14/U09-review/README.md'))
    known_public={
        'F17':[('历史同题工单 #1248','https://github.com/flagos-ai/FlagTree/issues/1248')],
        'F20':[('历史误报记录 #1249','https://github.com/flagos-ai/FlagTree/issues/1249'),
               ('历史误报记录 #1126','https://github.com/flagos-ai/FlagTree/issues/1126')],
        'N04':[('历史缺件说明 PR #1330','https://github.com/flagos-ai/FlagTree/pull/1330')],
    }
    if row['id'] in known_public:
        item['refs'] += [{'label':label,'path':url} for label,url in known_public[row['id']]]
        item['refs'].append(ref('2026-11-13 公开工单核对','work/2026-11-13/13项源码修复-官方Issue与PR核对.md'))
    if row['id'] in SUBMITTED_PAIRS:
        issue, pr = SUBMITTED_PAIRS[row['id']]
        item['refs'].extend([
            {'label': f'官方 Issue #{issue}', 'path': f'{GITHUB_REPO}/issues/{issue}'},
            {'label': f'官方修复 PR #{pr}', 'path': f'{GITHUB_REPO}/pull/{pr}'},
        ])
        item['notes'].append(f'2026-11-19 核对，f-zhaoweiming 已提交 Issue #{issue} 和 PR #{pr}，两者均开放。')
    if row['id'] in WITHOUT_FIX_PR:
        item['notes'].append('2026-11-19 核对，官方仓库未找到对应修复 PR。已有工单、相关记录及组件归属见当日核对表。')
    if row['id'] in SUBMITTED_PAIRS or row['id'] in WITHOUT_FIX_PR:
        item['refs'].append(ref('13 项提交与厂商包归属（2026-11-19）', PUBLIC_REVIEW))
        item['refs'].append(ref('最新 main 的 P800 实卡复测', MAIN_RETEST))
        item['notes'].append('2026-11-19 在 P800 上重建并复测官方 main 5cf8ca193，本项现象仍可复现。')
    if row['id']=='N04':
        item.update(title='XPU 打包遗漏 SDK 自带的 LLVM 19 链接器',
                    component='FlagTree 的 XPU 工具打包',
                    solution='将当前 XTDK 自带的 bin/llvm-link 装入后端的 llvm19/bin 并随 wheel 分发。',
                    evidence='main 5cf8ca193 实测，官方 raw 向量加法的 8 项测试均因缺少链接器而失败。仅在原目录提供同一 SDK 的 llvm-link，8 项全部通过。')
    items.append(item)

def add(id,title,component,status,solution,evidence,paths,refs,related=(),notes=(),group='本次 FlagGems 诊断',baseline='2026-11-19 P800 诊断镜像；物理 GPU6 为有效算子与模型对照。'):
    items.append(dict(id=id,title=title,component=component,status=status,solution=solution,evidence=evidence,
        fix_paths=paths,refs=[ref(label,path) for label,path in refs],related=list(related),notes=list(notes),
        group=group,date='2026-11-19',historical_status='',baseline=baseline))

add('D01','P800 原生算子也可能卡在设备同步','设备与驱动执行状态','已定位并隔离，未根修',
    '先用 FlagGems OFF 的最小原生算子验卡；本次将有效对照移至物理 GPU6。未重置设备或宣称修复驱动。',
    '物理 GPU0 的 32 元素原生 add 在同步处等待，同配置 GPU6 通过；驱动有 task timeout / XPUERR_EVENTWAIT。11 月 14 日卡 3、7 已有同类原生执行等待的独立记录。',
    ['work/2026-11-19/flaggems-diagnosis/run-card6.sh','work/2026-11-19/flaggems-diagnosis/op_probe.py'],
    [('既往超时排查','work/2026-11-14/U07-timeout-diagnosis/README.md'),('本次原生等待栈',DIAG/'cpp-add-cap.log'),
     ('GPU0 驱动错误',DIAG/'driver-errors.txt'),('GPU6 正常对照',DIAG/'card6-add.json')],
    related=['U07','U09','N18','M08'],notes=['不同日期和卡号的相似等待不能证明同一个硬件根因；不能从日志认定硬件损坏。'])
add('D02','FlagGems 非连续 copy_ 回退报 invalid device function','FlagGems kunlunxin copy_','历史规避；本次实验修补通过',
    '旧 x-benchmark 文档记载在 enable 时排除 copy_。本次为保持全量开启，让已测同设备、非别名、正步长 FP16 的 2D/3D/4D 复制走已有 Triton pointwise copy。',
    '只开启 copy_ 即可复现；OFF 对照通过。attention mask [10,54,54]、4D 转置/带填充布局及四种二维转置权重在修补后逐位正确；MiniLM 三条路径全部通过。根因路径为 _can_use_triton 拒绝非连续输入后进入 CompositeExplicitAutograd redispatch。',
    ['flag_gems/runtime/backend/_kunlunxin/ops/copy.py:_can_use_triton/copy_',
     'work/2026-11-19/flaggems-diagnosis/experimental_fixes.py:apply'],
    [('既往 copy_ 复现与规避','review-local/model-usefulness-20261117/x-benchmark/docs/p800_flag_gems_copy_unavailable_reproduction.md'),
     ('本次安装源码快照',DIAG/'kunlunxin-copy.py'),('进程内修补',DIAG/'experimental_fixes.py'),
     ('3D/4D 最小复现',DIAG/'copy_probe.py'),('2D 权重复现',DIAG/'weight_copy_probe.py'),
     ('最终模型结果',DIAG/'model-on-fixed-v4.json')],related=['D03','D04','M05'],
    notes=['此前是过滤规避；本次直接 flag_gems.enable() 的 870 项全量注册不使用该过滤，因此会再次触发。旧文档中的过滤配置不代表本次实验已启用。',
           '修补仅作用于实验进程，未修改安装包；任意 dtype、重叠写入和其他布局尚未证明通用修复。M05 是 cat 编译问题，不能因都含非连续布局就合并。'])
add('D03','新建 grid lambda 地址进入缓存键导致重复编译','当前 FlagTree XPU JIT 与 FlagGems 调用方式','实验规避通过，运行时通用修复待落地',
    '在进入 JIT 前把 callable grid 求值为整数 tuple。当前进程修补覆盖 lift_fresh；通用修复应稳定运行时缓存键，并保留 XPU 所需的 grid 编译信息。',
    'JITFunction.run 将 grid 放入 options，缓存键中出现 lambda 地址。empty/lift_fresh 预热后各 5 次调用仍编译 5 次；仅将 grid 求值后，两者均为 0 次。主机同步中位数分别 214.894→0.125 ms、204.992→0.128 ms。修 copy/empty 后 encode 仍约 743 ms，进一步修 grid 后约 18–19 ms。',
    ['third_party/xpu/python/triton/runtime/jit.py:JITFunction.run/compute_cache_key',
     'flag_gems/ops/lift_fresh.py','work/2026-11-19/flaggems-diagnosis/experimental_fixes.py'],
    [('被测运行时快照',DIAG/'runtime-jit.py'),('缓存键与编译次数',DIAG/'grid-cache-results.json'),
     ('最小对照脚本',DIAG/'grid_cache_probe.py'),('encode 调用剖析',DIAG/'encode-cprofile.json')],
    related=['F15','F38','N09','N10','N26','D04'],
    notes=['F15 是输入地址变化导致重复准备，N09/N10 是命中后的调用与元数据成本；它们不能作为本次 callable-grid 缓存键已修好的证明。',
           '对照保持算法源码和输入，改变了 grid 编译选项，不声明两种二进制逐字节相同；不能简单从缓存键删掉 grid。'])
add('D04','empty 分配后额外发射零填充内核','被测 FlagGems empty.memory_format','进程内修补通过',
    '保留基于 meta stride 的 empty_strided 分配，省去不必要的零填充发射；返回未初始化内存符合 empty 的契约。',
    '3 次 LayerNorm 触发 6 次 empty 编译，累计约 1.497 秒。替换为仅分配的实现后，全量 ON LayerNorm 主机同步中位数由约 485.009 降至 0.352 ms，已测 FP16 误差检查通过。',
    ['flag_gems/ops/empty.py','work/2026-11-19/flaggems-diagnosis/experimental_fixes.py'],
    [('安装源码快照',DIAG/'flaggems-empty.py'),('编译剖析',DIAG/'ln-cprofile.json'),
     ('修补前',DIAG/'ln-on.json'),('修补后',DIAG/'ln-fixed-empty.json')],related=['D03'],
    notes=['485 ms 是含主机编译和同步的调用耗时，不是 LayerNorm 设备计算时间。未把本次局部修补标记为 FlagGems 上游已合入。'])

# Follow-up repair: preserve the earlier experiments while recording deployment and tests.
repair = json.loads((REPAIR/'summary.json').read_text())
by_id = {i['id']:i for i in items}
by_id['D01']['evidence'] += ' 后续 GPU5/7/2 的最小原生 add 仍同步等待；GPU3 上传等待且另有并发进程。GPU4 原生检查通过，用于最终模型对照。GPU6 首轮模型设备计时等待期间观察到额外 probe 任务，污染轮次单独归档，不据此断言因果。'
by_id['D01']['refs'] += [ref('GPU4 原生健康检查',REPAIR/'health-gpu4.log'),ref('GPU6 并发任务证据',REPAIR/'gpu6-interference.txt'),ref('GPU5 原生等待',REPAIR/'health-gpu5.log')]
for id in ('D02','D03','D04'):
    by_id[id]['baseline'] += ' 后续运行时及算子回归在 GPU6，最终模型 ON/OFF 对照在 GPU4，详见复验报告。'
    by_id[id]['refs'] += [ref('复用旧修复后的实测报告',REPAIR/'report.html'),
                         ref('修复复验汇总',REPAIR/'summary.json')]
by_id['D02'].update(status='旧过滤方案复用并验证；底层回退未根修',
    solution='后续复验按旧 x-benchmark 记录使用 flag_gems.enable(unused=["copy_"])，实际注册 869 项。copy_ 使用原生路径；此前保留全量 870 项的进程内 strided-copy 修补仍单独归档。')
by_id['D02']['evidence'] += ' 后续复验的 FP16/FP32、2D/3D/4D 非连续 D2D/D2H/H2D 检查通过，MiniLM ON/OFF 三条路径共 6 项全部通过。'
by_id['D02']['notes'] = ['早期诊断直接全量 enable，未使用旧过滤；后续修复已实际启用该过滤，不能将两轮启用策略混写。',
                         '未宣称 copy_ 的 CompositeExplicitAutograd 错误已根修，也未覆盖任意重叠写入。M05 的 cat 编译问题另列。']
by_id['D02']['fix_paths'].append('work/2026-11-19/flaggems-repair/bench.py')
by_id['D02']['refs'].append(ref('27 项算子检查',REPAIR/'operators.json'))
by_id['D02']['status'] = '官方旧版回退不兼容；被测镜像过滤规避'
by_id['D02']['evidence'] += ' 来源复核：镜像 copy.py 与官方旧提交 4fa5805b8 逐字节一致；本地干净 HEAD 的 copy.py 与官方 master 6260a69a 一致，已移除旧路径。新增独立进程对照不导入 FlagGems：普通 copy_ 通过，显式 CompositeExplicitAutograd redispatch 报相同错误。CUDA 注册表指向 torch_xmlir，失败 C++ 栈则进入 direct_copy_kernel_cuda，证实回退绕过了 P800 适配。'
by_id['D02']['notes'].append('“未根修”指被测旧镜像未替换 copy_，不表示当前官方源码没有修改。官方 #5971 首次移除该非连续拒绝和回退，#6340 后续再次移除 redispatch；当前实现还显式拒绝跨设备复制，不能未经 H2D/D2H 等验证就直接替换旧包。')
by_id['D02']['refs'] += [ref('官方／本地／安装包来源定位',COPY_ORIGIN/'README.md'),
                         ref('源码指纹与根因汇总',COPY_ORIGIN/'summary.json'),
                         ref('不加载 FlagGems 的回退失败及 C++ 栈',COPY_ORIGIN/'copy-origin-redispatch.json'),
                         ref('同环境普通 copy_ 通过',COPY_ORIGIN/'copy-origin-native.json')]
by_id['D02']['refs'] += [{'label':'官方旧 copy.py','path':'https://github.com/flagos-ai/FlagGems/blob/4fa5805b8f21c45126a31562ff0706cb730954bd/src/flag_gems/runtime/backend/_kunlunxin/ops/copy.py'},
                         {'label':'已核验的官方当前 copy.py','path':'https://github.com/flagos-ai/FlagGems/blob/6260a69ae6822fc5c045386e0421006f8ad1ae31/src/flag_gems/runtime/backend/_kunlunxin/ops/copy.py'}]

by_id['D03'].update(status='XPU 运行时源码已修，实卡回归通过',
    solution='JITFunction.run 在 binder 绑定 constexpr/default 参数后将 callable grid 求值为 tuple，同时写入缓存键和编译选项，并复用该值发射。保留 XPU 必需的 grid 信息；callable grid 不进入 _bound_launchers，避免逐次保留闭包。')
by_id['D03']['evidence'] += ' 后续正式回归：旧运行时 6/6 失败、修正版 6/6 通过，覆盖 tuple/list 回调、值变化、静态/回调缓存共享、warmup、autotune、闭包释放。27 项算子检查及 6 项模型对照通过；LayerNorm/GELU/MM 预热后均 0 次重编译。'
by_id['D03']['notes'].append('源码已写入本地 FlagTree；P800 通过独立容器只读挂载运行时复验，未更新公共基础镜像或宣称上游已合入。')
by_id['D03']['fix_paths'].append('python/test/unit/runtime/test_xpu_callable_grid_cache.py')
by_id['D03']['refs'] += [ref('运行时补丁',REPAIR/'runtime.patch'),ref('旧版回归失败',REPAIR/'grid-before.log'),ref('修正版回归通过',REPAIR/'grid-after.log')]
by_id['D04'].update(title='通用 empty/lift_fresh 遗漏已有昆仑芯专用实现',status='复用已有 vendor 实现，实卡验证通过',
    solution='从本地 FlagGems HEAD 原样复用 kunlunxin empty（仅分配）和 lift_fresh（返回输入）实现，只读挂载到旧镜像的对应通用模块。empty 不零填充；lift_fresh 保持 aten 的 identity/alias 语义，不与 lift_fresh_copy 混同。')
by_id['D04']['evidence'] += ' 已确认两份文件与本地 FlagGems HEAD 一致，来源与哈希有记录。12 项 empty dtype/shape/memory_format、4 项 lift_fresh 别名和数值检查通过；本轮 LayerNorm 主机中位数 0.342 ms，预热后重编译为 0。'
by_id['D04']['notes'] = ['现有源码已经有专用实现，不代表此前固定测试镜像使用了它；本次验证实际挂载文件的哈希。',
                         '485 ms 是含主机编译和同步的旧调用耗时，不是设备计算时间。复用范围仅本轮实验容器。']
by_id['D04']['fix_paths'] += ['work/2026-11-19/flaggems-repair/vendor-empty.py','work/2026-11-19/flaggems-repair/vendor-lift_fresh.py']
by_id['D04']['refs'] += [ref('复用源码来源',REPAIR/'reused-sources.json'),ref('算子验证',REPAIR/'operators.json')]
add('D05','不完整覆盖的小 grid 比预期多写一个块','XPU 编译／发射路径','原版与修正版均复现，未根修',
    '单独保留最小复现；本轮正常完整覆盖 grid 的缓存回归和模型验证通过。没有把该异常计作缓存修复已解决。',
    'n=128、BLOCK=32、grid=1/2、仅使用 i<n mask 的复制加一内核，分别写入 64/96 元素而非 32/64；旧版和 grid 缓存修正版均复现。具体 lowering 根因尚未判定。',
    ['work/2026-11-19/flaggems-repair/debug_grid.py'],
    [('最小复现',REPAIR/'debug_grid.py'),('旧运行时结果',REPAIR/'debug-grid-before.log'),('修正版结果',REPAIR/'debug-grid.log')],related=['D03'],
    notes=['缩小 grid 是有意义的独立语义检查，不能因常规工作负载完整覆盖就宣称该行为正确。'])

by_id['D02']['status'] = '当前官方 D2D 通过；跨设备复制仍不支持'
by_id['D02']['solution'] += ' 最新实验已完整安装官方 36a2e186a，不排除 copy_；仅开启 copy_ 和全量开启各 10 项 D2D 复制通过。当前官方显式拒绝跨设备复制，不能继续沿用旧版“只需排除非连续回退”的结论。'
by_id['D02']['evidence'] += ' 5,867 个安装源码文件与官方提交一致；实际 1,793 项注册包含 copy_。FP16/FP32 转置、切片、注意力布局及连续输入均逐位正确；4 项 H2D/D2H copy_ 和 tensor.to 拒绝跨设备。最初 D2D 用例失败在结果回读，退出 Gems 上下文后原生回读确认内核正确。'
by_id['D02']['notes'].append('当前官方全量 ON 的 MiniLM 尚未跑通，新的注意力限制与 FakeTensor 兼容问题分别见 D06、D07；不能与旧镜像过滤 copy_ 的通过结果混写。')
by_id['D02']['related'] += ['D06','D07']
by_id['D02']['refs'] += [ref('当前官方安装及实测报告',OFFICIAL_GEMS/'README.md'),ref('当前官方结果汇总',OFFICIAL_GEMS/'summary.json'),ref('全量开启 D2D 通过',OFFICIAL_GEMS/'copy-scoped-on.json'),ref('跨设备入口失败',OFFICIAL_GEMS/'copy-only.json')]
by_id['D04']['notes'].append('当前官方 36a2e186a 完整安装后自动选择 vendor empty/lift_fresh；不再额外挂载旧文件。加载路径及源码校验见当前官方报告。')
by_id['D04']['refs'].append(ref('新版自动选择 vendor 实现',OFFICIAL_GEMS/'import-probe.json'))
official_gems_baseline='官方 FlagGems 36a2e186a；现有定制 FlagTree 与 PyTorch 2.9.0；P800 GPU4。'
add('D06','当前官方高效注意力拒绝 MiniLM 的 attn_bias','FlagGems kunlunxin efficient attention','官方明确不支持；模型未跑通',
    '保留模型 padding mask 语义并记录失败；未删除 mask 或修改官方实现绕过检查。',
    '同权重、同输入 MiniLM OFF 完整通过；当前官方全量 ON eager 首次 forward 在 _scaled_dot_product_efficient_attention.py:370 的 attn_bias is None 断言失败。',
    ['flag_gems/runtime/backend/_kunlunxin/ops/_scaled_dot_product_efficient_attention.py:370'],
    [('当前官方报告',OFFICIAL_GEMS/'README.md'),('eager 首次失败',OFFICIAL_GEMS/'model-on.json'),('同输入原生模型通过',OFFICIAL_GEMS/'model-eager-off.json')],related=['D02','D07'],baseline=official_gems_baseline)
add('D07','官方 randperm 导入替换使 FakeTensor 进入真实内核','FlagGems Python 层替换与 Inductor 图编译','触发机制已定位；未修复',
    '用警告升级为异常，在访问 FakeTensor.data_ptr 时提前中止；未修改官方代码。普通 vendor 排除表不足以阻止导入时的 torch.randperm 替换。',
    'baseline 和 QKV 各自独立进程编译报非法访存。诊断栈定位 Inductor randperm_index_add_pattern → _kunlunxin_randperm_dispatch → randperm_affine_kernel → libentry.data_ptr。8 元素 FakeTensor 最小复现中，原生通过，只 import FlagGems 即失败；全量开启时 randperm 不在注册列表中但 Python 函数仍已被替换。',
    ['flag_gems/runtime/backend/_kunlunxin/ops/randperm.py:503-514','flag_gems/utils/libentry.py:1609'],
    [('诊断栈',OFFICIAL_GEMS/'model-fake-pointer.json'),('最小复现脚本',OFFICIAL_GEMS/'fake_randperm_probe.py'),('原生 FakeTensor 通过',OFFICIAL_GEMS/'fake-randperm-off.json'),('仅导入即复现',OFFICIAL_GEMS/'fake-randperm-import.json'),('全量启用与排除表核对',OFFICIAL_GEMS/'fake-randperm-on.json'),('独立 baseline',OFFICIAL_GEMS/'model-baseline-only.json'),('独立 QKV',OFFICIAL_GEMS/'model-qkv-only.json')],
    related=['D02','D06','M04'],notes=['首次多路径实验的后续错误受上下文污染，正式编译证据使用独立进程。当前编译栈未升级至新版官方 extra 要求，不能泛化为所有硬件环境结论。'],baseline=official_gems_baseline)

# Later on the same day: keep the pristine-source failures above as history,
# and distinguish the locally patched wheel from upstream integration.
current_by_id = {item['id']: item for item in items}
for issue_id in ('D02', 'D06', 'D07'):
    item = current_by_id[issue_id]
    item['status'] = '官方基线本地修复；安装包回归与模型通过'
    item['baseline'] = '官方 FlagGems 36a2e186a + 本地 p800fix1；固定 FlagTree grid 修复；P800 GPU6。'
    item['refs'] += [ref('保持 copy_ 开启的修复报告', GEMS_FIXES/'README.md'),
                     ref('四处源码修复与回归用例', GEMS_FIXES/'patch.diff'),
                     ref('安装包 42 项回归', GEMS_FIXES/'compat-wheel.xml'),
                     ref('同权重同输入三条模型路径', GEMS_FIXES/'model-on-validation.json'),
                     ref('实际安装文件校验', GEMS_FIXES/'installation-verification.json'),
                     ref('最终 ON/OFF 同输入性能与验收汇总', GEMS_FIXES/'summary.json'),
                     ref('安装包完整 ON 模型与设备图重放', GEMS_FIXES/'perf-on-final.json')]
    item['notes'].append('此状态来自 11 月 19 日后续本地修复，不表示官方已合入；全量 enable(unused=[]) 注册 1,793 项且包含 copy_。安装包 ON/OFF 共 6 条模型路径及换输入图重放通过。QKV ON 的 encode 16.026 ms、设备 11.763 ms，仍慢于 OFF 6.671/1.619 ms；不能把兼容修复称为全量 Gems 的性能收益。')
current_by_id['D02']['solution'] = '当前官方 D2D 路径保留；覆盖注册前捕获厂商 CUDA copy_ kernel，让跨设备分支调用该实现。没有排除 copy_，也不再进入旧 CompositeExplicitAutograd 回退。'
current_by_id['D02']['evidence'] += ' 后续本地安装包的 13 项 H2D/D2H 回归通过，覆盖 FP16/FP32/int64、连续/转置、阻塞/异步、类型转换、广播和存储偏移；三条 MiniLM 路径的 forward/encode/换输入通过。'
current_by_id['D02']['notes'] = [n.replace('当前官方全量 ON 的 MiniLM 尚未跑通', '未修补官方基线的全量 ON MiniLM 当时未跑通') for n in current_by_id['D02']['notes']]
current_by_id['D06']['solution'] = '复用 dense mask 转换并合并 bias/causal 语义，保留负无穷屏蔽，修正在线 softmax 初值和全屏蔽行 LSE；Philox 元数据显式置于 CPU，避免默认 GPU 设备影响导入。'
current_by_id['D06']['evidence'] += ' 后续本地修复的 28 项 FP16/FP32 注意力检查对 CPU double 参考通过；MiniLM eager、baseline、direct_qkv 同输入输出一致性通过，最大绝对误差不超过 0.000367。'
current_by_id['D07']['solution'] = '移除导入时对 torch.randperm 的全局替换，保留 FlagGems 的标准注册机制，使厂商已有排除表与 FakeTensor 分派生效。未额外增加用户排除项。'
current_by_id['D07']['evidence'] += ' 后续本地修复的全量开启 FakeTensor factory 回归通过，baseline 与 direct_qkv 两条 fullgraph 路径均编译并完成换输入验证。'

add('D08','P800 注意力 bias 兼容路径的二维 mask 展开过慢','FlagGems Kunlunxin attention mask 布局','本地优化并验证；整模型有收益',
    '将上一轮兼容修复复用的 64×64 二维 mask 展开改为 512 元素一维 tile，连续写入包括 padding 的完整缓冲区；因此去掉独立 infinity 初始化。保留 QK、softmax、FP32 P×V 精度及 FlagGems/copy_ 全量开启。',
    '同进程分段计时定位 mask 约 0.95 ms、QK 约 0.036 ms、P×V 约 0.042 ms。只改 constexpr 或二维尾部条件仍慢；一维实现把 mask 降到 0.056 ms，注意力设备时间 0.993→0.135 ms。同数据/权重 MiniLM QKV encode 16.419→13.130 ms，设备 12.053→6.246 ms；OFF 对照 6.789/1.633 ms，ON 仍较慢。',
    ['work/2026-11-19/flaggems-attention-opt/FlagGems/src/flag_gems/runtime/backend/_kunlunxin/ops/_scaled_dot_product_efficient_attention.py:_kl_sdpa_mask'],
    [('定位、消融与模型性能报告',ATTENTION_OPT/'README.md'),('仅本次优化补丁',ATTENTION_OPT/'optimization.patch'),
     ('分段前后测量',ATTENTION_OPT/'profile-flat-mask.json'),('布局消融（含未完成调参）',ATTENTION_OPT/'profile-mask-ablation.json'),
     ('最终三方验证与性能',ATTENTION_OPT/'summary.json'),('安装包回归',ATTENTION_OPT/'compat-wheel.xml'),
     ('安装包指纹',ATTENTION_OPT/'deployment.json')],related=['D02','D06','D09'],
    notes=['基线为官方 36a2e186a 上的本地 p800fix1，优化包 p800fix2；不是未经修改的官方 efficient attention 支持 bias 后自然得到的结果。',
           '整段融合触及 64 个流水事件上限，直接融合 bias 到 QK 数值失败，均撤回；未声称这些编译器问题已修好。没有指令级证据进一步归因具体机器指令。',
           '84 项安装包检查通过、4 项既有问题严格 xfail。共享主机计时窗口的采样未发现同卡其他任务；不等于严格独占。'],
    baseline='官方 FlagGems 36a2e186a + p800fix1→p800fix2；同一固定 FlagTree JIT；P800 GPU6。')
add('D09','非方形 attention 的 33/65、65/33 序列出现 NaN','FlagGems Kunlunxin two-pass attention','优化前后复现；未修复',
    '保留优化前后独立进程复现，将 4 个 dtype 用例标为严格 xfail。新增 mask 单独逐单元检查在相同形状通过，不能把后续 attention 计算失败计作 mask 优化已解决。',
    'Q/K=33/65 与 65/33 两组形状，FP16/FP32 各自失败；旧 p800fix1 对同脚本也有 4 项失败，另 4 项非方形/布局用例通过。当前 MiniLM 10×54 的三方共 9 条模型路径通过，不覆盖这些失败形状。',
    ['work/2026-11-19/flaggems-attention-opt/FlagGems/tests/test_kunlunxin_model_compat.py:test_attention_mask_strides_and_rectangular_lengths'],
    [('优化前独立复现',ATTENTION_OPT/'rectangular-before.log'),('优化后首次扩展检查',ATTENTION_OPT/'compat-78.log'),
     ('严格 xfail 与独立 mask 检查',ATTENTION_OPT/'compat-wheel.xml'),('边界说明',ATTENTION_OPT/'README.md')],
    related=['D06','D08'],notes=['具体 lowering 根因尚未判定，不能泛化为任意非方形 attention 均失败。'],
    baseline='p800fix1 和 p800fix2 均复现；固定 P800/FlagTree 环境。')
current_by_id['D06']['related'] += ['D08','D09']
current_by_id['D06']['notes'].append('后续 p800fix2 已优化 dense mask 布局，模型端到端确有收益，详见 D08；额外非方形形状的既有 NaN 问题见 D09。')
current_by_id['D06']['refs'].append(ref('注意力性能优化及边界',ATTENTION_OPT/'README.md'))

add('D10','MiniLM mask 修复后的 LayerNorm 与矩阵准备开销','FlagGems Kunlunxin LayerNorm/mm','本地包修复；数值与三方性能复测通过，仍慢于 OFF',
    '按 U06/U20/U08a 将宽度 384 的 LayerNorm 改用 64×512 cluster tile、行首中心化和二维归约状态；按 F44/N06 验证尾部，短 FP16 转置权重投影复用官方 addmm_kernel 的 beta=0 路径，省去整理权重、补齐和复制输出。不关闭 FlagGems/copy_，不改变 FlagTree JIT。',
    '运行时追踪确认每个 forward 走 6 次 efficient-attention；QKV 的 12 次投影已消除 _padded_or_direct 调用。优化前 layernorm.py/mm.py 与官方 36a2e186a 哈希相同，原 LayerNorm 对宽度 384 使用 RBLOCK=8192。p800fix4 分进程 187 pass/4 既有严格 xfail。预留 GPU6 后三方共 9 条模型路径和换输入图重放通过，最大绝对误差 0.000366。QKV encode 13.157→10.561 ms（下降 19.73%），设备 6.360→3.002 ms（2.118×）；同方案 OFF 6.568/1.649 ms，ON 端到端仍慢约 61%。',
    ['work/2026-11-19/minilm-residual/FlagGems/src/flag_gems/runtime/backend/_kunlunxin/ops/layernorm.py:_layer_norm_384_kernel',
     'work/2026-11-19/minilm-residual/FlagGems/src/flag_gems/runtime/backend/_kunlunxin/ops/mm.py:_try_fp16_strided_projection'],
    [('本轮定位与复测状态',MINILM_RESIDUAL/'README.md'),('官方与本地文件来源',MINILM_RESIDUAL/'source-origin-before.json'),
     ('实际模型调用和数值验证',MINILM_RESIDUAL/'final-validation.json'),('仅本轮修复补丁',MINILM_RESIDUAL/'residual-only.patch'),
     ('矩阵回归',MINILM_RESIDUAL/'mm-regression.xml'),('兼容及 LayerNorm 回归',MINILM_RESIDUAL/'compat-final-recheck.xml'),('LayerNorm 图重放复核',MINILM_RESIDUAL/'ln-graph-recheck.xml'),
     ('安装包指纹',MINILM_RESIDUAL/'deployment-final.json'),('共享卡计时排除证据',MINILM_RESIDUAL/'occupancy-audit-contended.json'),
     ('正式三方性能及对齐检查',MINILM_RESIDUAL/'reserved-comparison.json'),('有效轮次占用审计',MINILM_RESIDUAL/'reserved-occupancy-audit.json'),
     ('两套安装包逐文件核验',MINILM_RESIDUAL/'reserved-package-verification.json')],
    related=['D06','D08','D09','U06','U20','U08a','F44','N06','F41','N24'],
    notes=['仅覆盖已测 P800、精度、形状与布局；不是官方合入，也不是编译器底层问题根修。',
           'p800fix4 首次合并回归超时；随后 128 pass/4 既有 xfail（单独复核的图重放用例在该进程取消选择），另有 58 项矩阵及 1 项图重放通过，合计 187 pass/4 xfail。保留超时日志，不宣称执行状态问题已根修。',
           '当前 MiniLM 已使用 D08 一维 mask；旧 U26b/U26c、D06 功能支持与 D08 性能修复是不同问题。',
           '早期额外注意力分段重放在 stream 切换同步出现 status 702，根因尚未归属。后续预留窗口的 9 条整模型图重放全部通过，但没有据此宣称该分段问题已根修。',
           '正式测试为 22:14–22:17，三份有效运行分别有 33/34/21 个占用采样，未见同卡其他进程；首次有干扰的 before 尝试已排除。event 与独立墙钟中位数最大相差约 0.154%。共享主机、采样监控，不等于驱动强制独占。'],
    baseline='官方 36a2e186a + 本地 p800fix2→p800fix4；同权重、原始 10 句、FP16；固定 FlagTree JIT。')
next(item for item in items if item['id']=='D08')['notes'].append(
    '后续来源核对：未修改官方二维 helper 与本地兼容 helper 在同一模型布局下都慢，一维版快；但把该 helper 接入 efficient attention 是本地决定。当前 MiniLM 实际入口已走一维版本，剩余 LayerNorm/mm 开销见 D10。共享卡辅助微测只作定位线索。')
next(item for item in items if item['id']=='D08')['refs'].append(ref('mask 辅助内核官方/本地对照',MINILM_RESIDUAL/'mask-origin-probe.json'))

model_common=dict(group='后续整模型集成',baseline='f06_p800 0.3.0+v51e.inductor1；2026-11-19 P800 独立整模型验证。')
mr=[('本轮整模型报告',MODEL/'report.html'),('结果与实际路由',MODEL/'results.json'),('验收汇总',MODEL/'verification.json'),
    ('包源码差异',MODEL/'package-changes.patch'),('安装包与实装指纹',MODEL/'source-verification.json')]
add('M01','厂商 Linear 阻碍 FakeTensor 编译及整图调度','F06 与 Inductor 接入','本地包实现并验证',
    '在 AOTAutograd 前将厂商 opaque Linear 暴露为 aten.linear；Dynamo 图交给真正的 Inductor 统一调度、分配和代码生成，按整图同步调用耗时选路。',
    '三个模型 × FP32/FP16 共 6 项通过，117 项模板和 103 项专项检查通过。入口 f06.compile_model 的审计记录 backend=inductor_whole_graph。',
    ['python/triton/experimental/graph_fusion/_model_inductor.py:expose_linear/inductor_backend',
     'python/triton/experimental/graph_fusion/_model.py','packaging/f06/src/f06/__init__.py'],mr,
    related=['F09','F12','F37','N09','N27'],notes=['显式入口生效，未把所有 torch.compile 默认后端改为 F06。限定静态形状推理。'],**model_common)
add('M02','CLIP 残差仍有消费者时不能只返回归一化结果','Inductor 多输出 LayerNorm','本地包实现并验证',
    '用同一次设备发射写出归一化值和仍被使用的残差值；保留广播、dtype 和可见输出契约。只改写符合条件的相邻纯操作。',
    '18 项双输出单次发射检查通过；FP16 CLIP 实发 46 次双输出残差归一化和 2 次单输出归一化。',
    ['python/triton/experimental/graph_fusion/_p800_model_layernorm.py:_lower/rewrite',
     'python/triton/experimental/graph_fusion/_p800_layernorm.xpu'],mr,
    related=['F05','F10','F28'],notes=['一个捕获图中出现模式不等于实发融合核；该项保留二进制与实际发射证据。'],**model_common)
add('M03','QuickGELU 的完整表达式缺少融合支持','模型图改写','本地包实现并验证',
    '匹配 x * sigmoid(1.702 * x)，检查消费者及相邻关系；支持的低精度模式内部使用 FP32，外部输出保留原 dtype。',
    'FP16 CLIP 实发 24 次融合 QuickGELU，整模型精度检查通过。',
    ['python/triton/experimental/graph_fusion/_model_inductor.py:expose_fp32_patterns'],mr,
    related=['F25','F26','N19'],notes=['QuickGELU 与精确 erf GELU、tanh 近似 GELU 是不同函数，不能混用替换。'],**model_common)
add('M04','embedding 与高级索引的设备断言符号无法链接','XPU 模型编译与厂商算子','已验证局部绕行',
    '在显式启用的整模型进程中为 embedding 和 index.Tensor 保留厂商实现，让原有越界检查继续生效。',
    '该路径进入三模型六项最终验证并通过；修补前整图编译缺少 __assert_fail 设备符号。',
    ['python/triton/experimental/graph_fusion/_model_inductor.py:inductor_backend'],mr+[
        ('入口与语义约定','packaging/f06/README.md')],related=['U12','N25'],
    notes=['这是 fallback 交付，不是补齐 XPU 断言降层；没有关闭越界校验。'],**model_common)
add('M05','非连续 transpose 加 cat 触发 XPU 代码生成错误','XPU concat 与模型接入','已验证局部绕行',
    '为 aten.cat 保留厂商路径并覆盖相关 decomposition，避免有问题的非连续拼接降层。',
    '代码与包说明记录该复现及 fallback；包含 CLIP 的最终六项模型验证通过。',
    ['python/triton/experimental/graph_fusion/_model_inductor.py:inductor_backend'],mr+[
        ('运行约定','packaging/f06/README.md')],related=['D02'],
    notes=['与 FlagGems copy_ 的 redispatch 错误是不同入口和机制；当前没有证据表明底层 cat 代码生成已根修。'],**model_common)
add('M06','局部归约正确不代表整模型精度通过','模型级数值验收与敏感算子路由','模型级保护与绕行已验证',
    '以完整模型 FP32 eager 参考验收候选，必要时保留厂商 LayerNorm/GELU/tanh；最终低精度输出使用独立厂商转换，保留仍需使用的 FP32 中间值。',
    'FP32 BGE、CLIP 的部分自定义归约候选被整模型精度检查拒绝，最终使用厂商 LayerNorm；6 项模型输出与 103 项专项检查通过。',
    ['python/triton/experimental/graph_fusion/_model_inductor.py:_compare/promote_whole_graph',
     'python/triton/experimental/graph_fusion/_p800_model_layernorm.py:preserve_unmatched_norms/_final_cast'],mr,
    related=['F28','F39','N19','U20'],notes=['仅证明已测模型与数值契约；没有将任务质量或训练支持扩展为已验收。'],**model_common)
add('M07','冻结参数转换缓存必须随参数修改失效','整模型 FP32 参数缓存','本地包实现并专项验证',
    '仅缓存冻结参数转换，以对象身份、存储地址、版本计数、dtype、device、shape、stride 作为键，并处理流间等待；不缓存输入激活与输出。',
    '代码中的参数缓存键及当前包专项检查保留在验收记录中，实装源码与 wheel 指纹一致。',
    ['python/triton/experimental/graph_fusion/_model_inductor.py:_ParameterPack'],mr+[
        ('缓存边界说明','packaging/f06/README.md')],related=['F08','F11','F15','N26'],
    notes=['这是新接入中的一致性保护记录，不据此认定旧版本一定产生过陈旧输出。',
           '普通原地更新会失效；绕过 PyTorch 版本计数的外部写入及 parameter.data.copy_ 不在该缓存约定内。'],**model_common)
add('M08','矩阵补齐的 GPU 图校准可在冷编译时等待','模型选路与冷启动校准','局部规避，冷启动根因未闭环',
    '关闭该独立 shape-padding GPU 图校准，完整图选路仍使用带同步的主机计时。',
    '同卡旧→新→旧的追加对照未复现超过 5% 的稳态回退；旧包仍有原生 GEMM 冷启动校准超时，缺失样本未补值。',
    ['python/triton/experimental/graph_fusion/_model_inductor.py:inductor_backend'],mr+[
        ('同卡回归对照',MODEL/'comparison.json')],related=['D01','N18','F41','N24'],
    notes=['当前稳态结果和 117 项模板通过，均不证明冷启动等待已彻底解决。'],**model_common)

benchmark_updates_path = OUT / 'model-benchmark-updates-20261119.json'
benchmark_updates = json.loads(benchmark_updates_path.read_text())
by_id = {i['id']: i for i in items}
# The cumulative local wheel integrates the padded QK fix with the D08 and D10 operators.
local_fix5 = ROOT / 'work/2026-11-24/xpu-gems-eager/all-local-fixes'
local_minilm = local_fix5 / 'evidence/43.179.154.78/e9e2c8815f70dad48afc'
d06 = by_id['D06']
d06['date'] = '2026-11-25'
d06['solution'] += ' 累计包 p800fix5 将 QK 分数缓冲区的填充尾列初始化为负无穷，并保留 D08 的一维掩码布局和 D10。'
d06['evidence'] += ' 在本地累计修复版上，28 项注意力回归全部通过。官方适配器的 MiniLM 端到端输出相对 eager 最大绝对误差为 0.000244，3 次预热、10 次同步前向的 eager / ON 均值为 6.80 / 16.69 毫秒。'
d06['refs'].extend([ref('累计包与编译器指纹', local_fix5 / 'pins.json'),
                    ref('FlagGems D06 Issue 与已有方案', OUT / 'FlagGems-D06-Issue.md'),
                    ref('全量本地包注意力回归', local_minilm / 'operator-regression/local-on/pytest.xml'),
                    ref('累计包 MiniLM 前向', local_minilm / 'main/local-on/state.json')])
d09 = by_id['D09']
d09.update(date='2026-11-25', status='累计包 p800fix5 已通过历史四项数值回归',
           solution='p800fix5 延用 D08 的一维 mask，将 QK 分数缓冲区的填充尾列初始化为负无穷。11 月 19 日保留的矩形用例在相同 FP64 参考和容差下通过。',
           baseline='历史 p800fix1 和 p800fix2 可复现。累计本地 FlagTree + FlagGems p800fix5 在 P800 以无图 eager 路径复验。')
d09['evidence'] += ' 11 月 25 日，两类长度 33/65、65/33 各自的 FP16/FP32 检查均在累计包上得到 XPASS(strict)。第二张通过验卡的 P800 使用 --runxfail 对同函数全部 8 个参数组合独立复验，8 项通过，输出与 LSE 对照 CPU FP64，绝对容差 0.002、相对容差 0.005。'
d09['refs'].extend([ref('严格预期失败标记下的原始结果', local_minilm / 'operator-regression/local-on/pytest.xml'),
                    ref('关闭预期失败标记的独立复验', local_minilm / 'fix-verification/local-on/pytest.xml'),
                    ref('累计包指纹', local_fix5 / 'pins.json')])
for update in benchmark_updates['updates']:
    item = by_id[update['id']]
    # A verified extension may supersede an incomplete historical fix. Keep
    # its old evidence and notes, while displaying the current repair scope.
    for field in ('status', 'solution', 'evidence', 'baseline', 'date'):
        if field in update:
            item[field] = update[field]
    for path in update.get('fix_paths', []):
        if path not in item['fix_paths']:
            item['fix_paths'].append(path)
    item['notes'].extend(update.get('notes', []))
    item['refs'].extend(ref(r['label'], r['path']) for r in update.get('refs', []))
sdk_records_path = OUT / 'p800-sdk' / 'records.json'
sdk_records = json.loads(sdk_records_path.read_text())
for extra in [*benchmark_updates['items'], *sdk_records]:
    assert extra['id'] not in by_id, extra['id']
    extra = dict(extra)
    extra['refs'] = [ref(r['label'], r['path']) for r in extra['refs']]
    items.append(extra)
    by_id[extra['id']] = extra
for issue_id, note in {
    'M75': '正式清单四个模型的 eager 在官方及本地 FlagTree 上生成相同的昆仑芯原生 BMM 模板诊断。逐头二维 mm 的模型级复测和后续阻断点见修复对照。',
    'M06': 'M2-BERT 三个上下文版本的原生 eager 已定位到复数除以实标量的类型转换。具体算子输入、第二个 FFT 阻断点和调用方规避见 SDK05。',
}.items():
    by_id[issue_id]['notes'].append(note)
    by_id[issue_id]['refs'].append(ref('P800 eager 逐模型修复对照', OUT/'P800-eager-修复对照-20261126.md'))
assert len(items)==119+len(benchmark_updates['items'])+len(sdk_records) and len({i['id'] for i in items})==len(items)
assert {i['id'] for i in items[:101]}=={i['id'] for i in source['items']}
ids={i['id'] for i in items}
for i in items:
    assert set(i['related'])<=ids
    # Fix symbols are navigational hints; verify actual current paths where present.
    for p in i['fix_paths']:
        base=p.split(':',1)[0]
        if (ROOT/base).is_file():
            r=ref('本地源码 '+base,ROOT/base)
            if r not in i['refs']:i['refs'].append(r)

groups=['本次 FlagGems 诊断','后续整模型集成','全清单测试与环境兼容','FlagTree 源码','XPU 工具包与预编译链路','本地 F06 与集成','历史线索与未解决项']
scope=f"保留原始 101 项编号，补充 10 项 FlagGems 诊断、8 项整模型接入与 {len(benchmark_updates['items'])} 项全清单测试记录和 {len(sdk_records)} 项 P800 SDK 接口记录，共 {len(items)} 条查询记录；这不是 {len(items)} 个独立且全部已修复的缺陷。"
latest = max(i['date'] for i in items)
intro='''# FlagTree 问题修复索引

更新日期：{latest}。覆盖 FlagTree、F06、Inductor 集成、P800 工具链和 FlagGems 相关修复与规避。

这份索引把问题症状、处置方式、验证范围、代码及原始证据放在一起。可按原编号（如 F24、U09）、报错（如 invalid device function）、算子（如 LayerNorm）或机制（如缓存、grid、非连续）搜索。旧编号保持不变；D、M 是本地查询编号，不是 GitHub Issue 编号。

{scope}

## 当前目标的持续要求

每次遇到错误和着手修复前，都先查询本手册及 `reports/issue-fixes/index.json`，核对已有方案、适用版本、实际部署与验证边界。不能仅因报错相似就认定同一根因，也不能重复把已知局部绕行写成新的根修。

相同根因的新复现补到旧条目；新的解决方案验证后加入手册，附复现脚本、修复前后结果、源码或安装包指纹和仍未解决的范围。未验证方案保留为待验证。此要求持续适用于 749 个模型的测试目标，测试队列继续执行。

## 状态如何理解

11 月 13 日归档的 101 项包括：13 项在固定官方基线上验证过源码修复，8 项验证过绕行，56 项本地实现／回归或处置记录，24 项历史、重复或尚未解决的线索。13 项源码修复为 F17、F20、F21、F22、F24、U05、U07、U09、U12、U13、U20、N01、N04；8 项绕行为 U06、U08a、U14、U24、U26b、U26c、U27a、U29c。

“源码修复”表示所列固定基线与用例上的修复验收；“本地回归通过”仅表示对应本地断言通过；“绕行”保留底层问题；“实验修补”只在指定进程生效。它们都不自动表示官方 main 已合并、所有服务器已安装或所有形状已覆盖。各条目的最新补充日期与原验证版本分开记录。

官方复验基线为 `e6a8799cc14819def1bdb8fe68b018fd774fe4c9`；更早审核包使用 `d65b6d70511e4d9aaeb3bf52ae762a96212def0a`，本地历史基线为 `feaf3f96973da3ce605fc23ce45d2df178a91526` 及 v51e。11 月 19 日复查 13 项，已有 5 对官方 Issue 与 PR，余下 8 项未找到对应修复 PR。[提交记录与厂商包归属](issue-fixes/13项提交与厂商包归属-20261119.md)

## 本次问题与以前工作的关系

| 这次遇到的现象 | 以前已有的记录 | 本次增加的结论与处置状态 |
|---|---|---|
| 原生 add／同步也等待 | U07 的 11 月 14 日设备排查；U09、N18 的等待记录 | D01：不同卡对照及驱动证据；仅隔离到健康卡，未宣称设备根修。 |
| 非连续 copy_ 报 invalid device function | x-benchmark 已记录同类错误，给出排除 copy_ 的规避 | D02：早期全量 enable 未用旧过滤而复现；后续已复用排除 copy_ 的旧方案，869 项注册及模型检查通过。 |
| 相同形状仍重复编译 | F15、F38、N09、N10、N26 涉及缓存及热调用 | D03：lambda 地址进入 JIT key 是独立机制；通用 XPU 运行时已修，6 项旧版失败／修正版通过。 |
| LayerNorm 主机调用约 485 ms | 归一化正确性与性能已有 F28、U06、U20 等项 | D04：复用此前已存在的 kunlunxin empty/lift_fresh 实现，27 项算子检查通过；不要与归一化数学缺陷混为一项。 |

早期诊断全量 FlagGems 保持 870 个注册项，修补仅在测试进程内。MiniLM FP16、10×54、物理 GPU6 上，eager、Inductor baseline 和 QKV 候选均通过数值及新输入图回放检查；它们的 encode 中位数分别为 18.85、15.31、15.78 ms，设备时间分别为 5.12、10.23、6.90 ms，仍慢于同轮 OFF。正确性修补通过不等于性能问题全部解决。

后续按旧记录修复：FlagTree grid 缓存源码落地，复用已有 vendor empty/lift_fresh，启用旧 copy_ 排除策略（869 项）。6 项运行时回归、27 项算子检查、三条 MiniLM 路径 ON/OFF 共 6 项通过。最新性能与原始结果见 [复验报告]({repair_report})，不与上述早期 870 项实验混算。新增 D05 保留小 grid 多写的未解决异常。

官方 FlagGems `36a2e186a` 的本地 `p800fix1` 修复兼容问题，`p800fix2` 优化 attention mask，最新 `p800fix4` 优化 LayerNorm-384 与短 FP16 矩阵准备路径。不排除 copy_，实际 1,793 项注册；187 项检查通过、4 项既有问题严格 xfail。预留 GPU6 后同权重同输入的三方共 9 条模型路径及换输入图重放通过。QKV 的 p800fix2→p800fix4 encode 13.157→10.561 ms（下降 19.73%）、设备 6.360→3.002 ms（2.118×），仍慢于 OFF 6.568/1.649 ms。兼容修复见 D02、D06、D07，mask 优化见 D08，剩余开销与正式复测见 D10，未修复形状边界见 D09；不表示官方已合入。

M 系列来自另一轮 `f06.compile_model` 安装包与三模型验证，不与 D 系列的 FlagGems 全量 ON 时延混算。当前交付包为 `0.3.0+v51e.inductor1`，其 6 项模型、117 项模板和 103 项专项检查通过；仍保留冷启动校准未闭环的 M08。

## 按症状快速查找

| 症状或关键词 | 优先查看 |
|---|---|
| invalid device function、non-contiguous、copy_ | D02；拼接代码生成问题另看 M05 |
| attn_bias is not supported、注意力 mask | D06 |
| MiniLM 实际注意力入口、mask 修好仍慢、LayerNorm-384、mm 准备开销 | D08、D10 |
| randperm、FakeTensor data_ptr、导入后编译非法访存 | D07 |
| 卡在 synchronize、xpu_wait、冷启动超时 | D01、M08、N18；先做原生 OFF 对照 |
| cache miss、重复编译、grid、lambda、热调用成本 | D03、D04、F15、F38、N09、N10、N26 |
| exp2 底数、tanh 饱和、SiLU 查表 | F17、U05、U07、U24、U25 |
| BF16 舍入、宽块写回、片上 store | F23、F24、U14、N19 |
| scf.if、yield、resizeSplat、类型布局 | U09、F22、U08a、U08b |
| uni_sram 耗尽、被包装的编译异常 | F20 |
| xpu86、llvm-link、math_rsqrt、__assert_fail | U12、N04、U29c、M04 |
| LayerNorm 高偏置、模型级超差 | F28、U06、U20、M06 |
| 输出被后续调用覆盖、参数更新未生效 | F08、F11、N11、M02、M07 |
| 负时间、错误选路、污染缓存、计时口径 | F36、F37、F39、F40、F41、F42、N23、N24、N26 |
| 整图、Linear、FakeTensor、CLIP、QuickGELU | M01、M02、M03、M04、M05 |
| File name too long、长描述内核名 | M09 |
| Tokenizer、AutoConfig、离线 revision 缓存 | M10、M12 |
| ModernBERT、Flash Attention、reference_compile | M11；设备断言另查 M04 |
| 退出后仍 running、下载超时、占用误判、证据缺失 | M13、M14、M15、M16 |
| kunlun_ops 导入失败、XFA undefined symbol | SDK01 |
| XFFT、复数 FP16 计划返回 14 | SDK02 |
| kunlun_ops FP16 GELU 的 exact 误差 | SDK03 |
| kunlun_ops Add+LayerNorm 权重 dtype 拒绝、双输出精度 | SDK04 |
| P800 原生 eager 的 BMM 模板缺 OneLoop、复数除实数、FP8 权重 cast、带掩码 SDPA head_dim 512、局部窗口变长 FlashAttention | M75、SDK05、SDK06、SDK07、SDK08 |
| Core ML、mlpackage、误下载导出权重 | M17 |

## 条目索引
'''.format(scope=scope,latest=latest,repair_report=REPAIR/'report.html')

md=[intro]
for group in groups:
    subset=[i for i in items if i['group']==group]
    md+=['\n### '+group+f'（{len(subset)} 条）\n','| 编号 | 问题 | 处置状态 |\n|---|---|---|']
    md += [f"| [{i['id']}](#{i['id'].lower()}) | {i['title']} | {i['status']} |" for i in subset]
md+=['\n## 问题与修复详情\n']
for group in groups:
    for i in [x for x in items if x['group']==group]:
        md+=['\n### '+i['id']+'\n','**'+i['title']+'**\n',
              f"状态：{i['status']}。归属：{i['component']}。最新记录：{i['date']}。\n",
              '**修复或处置：** '+i['solution']+'\n','**验证与边界：** '+i['evidence']+'\n',
              '**适用版本：** '+i['baseline']+'\n']
        if i['fix_paths']:md+=['**修补位置：** '+'；'.join('`'+x+'`' for x in i['fix_paths'])+'。\n']
        if i['historical_status']:md+=['**历史状态原文：** '+i['historical_status']+'。\n']
        md += [note+'\n' for note in i['notes']]
        if i['related']:md+=['关联条目：'+'、'.join(f'[{x}](#{x.lower()})' for x in i['related'])+'。\n']
        md+=['证据与复现：'+' · '.join(link(r) for r in i['refs'])+'。\n']

md+=['''\n## 后续维护规则

遇到相似报错时先按编号、报错文本、算子和根因查询。相同现象而根因不同的条目互相链接；同一机制的重复复现补到已有条目，保留新版本、卡号和时间。

新增或更新一条记录应包含：

- 编号、症状与精确触发条件，包含 shape、dtype、stride、启用项。
- 原因定位与所属层：应用、FlagGems、FlagTree、Inductor、工具包或设备状态。
- 修复方法及实际落点，注明根修、过滤规避、候选隔离还是进程内实验。
- 原版与修正版的源码／wheel／镜像指纹、设备和日期。
- 可执行复现、输出误差及首次编译／稳态主机／设备计时口径。
- 具体通过范围、仍失败的入口、部署范围及关联编号。
- 官方工单与合入状态若有更新，另附当次核验日期；不要仅凭本地通过改写为官方已关闭。

后续已补做 D02–D04 的源码修复／旧实现复用与实卡回归；没有重跑历史 101 项全套。索引的数据来源和 SHA256 保存在同目录索引数据中；修改生成器里的补充记录后重新生成，可同步更新 Markdown 与可搜索 HTML。
''']
sources=[sdk_records_path,MINILM_RESIDUAL/'reserved-comparison.json',MINILM_RESIDUAL/'reserved-occupancy-audit.json',MINILM_RESIDUAL/'reserved-package-verification.json',MINILM_RESIDUAL/'README.md',MINILM_RESIDUAL/'source-origin-before.json',MINILM_RESIDUAL/'final-validation.json',MINILM_RESIDUAL/'deployment-final.json',MINILM_RESIDUAL/'residual-only.patch',benchmark_updates_path,ROOT/benchmark_updates['lookup_evidence'],ROOT/benchmark_updates['goal_requirements'],ATTENTION_OPT/'summary.json',ATTENTION_OPT/'optimization.patch',ATTENTION_OPT/'compat-wheel.xml',GEMS_FIXES/'summary.json',GEMS_FIXES/'deployment.json',GEMS_FIXES/'compat-wheel.xml',GEMS_FIXES/'model-on-validation.json',GEMS_FIXES/'patch.diff',OFFICIAL_GEMS/'summary.json',PUBLIC_REVIEW,MAIN_RETEST,REPAIR/'summary.json',REPAIR/'reused-sources.json',REPAIR/'runtime.patch',REPAIR/'grid-after.xml',REPAIR/'operators.json',ARCHIVE/'index.json',OLD/'index.json',DIAG/'summary.json',DIAG/'model-on-fixed-v4.json',
         MODEL/'results.json',MODEL/'verification.json',MODEL/'source-verification.json',
         ROOT/'packaging/f06/README.md',ROOT/'work/2026-11-13/13项源码修复-官方Issue与PR核对.md',
         ROOT/'work/2026-11-14/U09-review/README.md',ROOT/'work/2026-11-14/U07-timeout-diagnosis/README.md',
         ROOT/'review-local/model-usefulness-20261117/x-benchmark/docs/p800_flag_gems_copy_unavailable_reproduction.md']
inventory={'updated':latest,'scope':scope,'count':len(items),'group_counts':dict(collections.Counter(i['group'] for i in items)),
           'historical_baseline_counts':source['counts'],'items':items,
           'sources':[{'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in sources]}
def publish_text(path, content):
    # Benchmark workers may read the index while a new edition is being built.
    mode = stat.S_IMODE(path.stat().st_mode) if path.exists() else 0o644
    with tempfile.NamedTemporaryFile(dir=path.parent, mode='w', encoding='utf-8', delete=False) as stream:
        staged = Path(stream.name)
        try:
            stream.write(content)
            stream.flush()
            os.fchmod(stream.fileno(), mode)
            staged.replace(path)
        finally:
            staged.unlink(missing_ok=True)


OUT.mkdir(exist_ok=True,parents=True)
publish_text(OUT/'index.json',json.dumps(inventory,ensure_ascii=False,indent=2))
md_path=ROOT/'reports/FlagTree问题修复索引.md'
publish_text(md_path,'\n'.join(md))

def h(x):return html.escape(str(x),quote=True)
def a(r):return f'<a href="{h(r["path"])}">{h(r["label"])}</a>'
cards=[]
for group in groups:
    for i in [x for x in items if x['group']==group]:
        body=''.join(f'<p><b>{h(k)}</b> {h(v)}</p>' for k,v in [
            ('修复或处置',i['solution']),('验证与边界',i['evidence']),('版本',i['baseline']),
            ('修补位置','；'.join(i['fix_paths']))] if v)
        body+=''.join('<p class="note">'+h(n)+'</p>' for n in i['notes'])
        body+='<p class="refs">'+' · '.join(a(r) for r in i['refs'])+'</p>'
        if i['related']:body+='<p>关联：'+' · '.join(f'<a href="#{x}">{x}</a>' for x in i['related'])+'</p>'
        cards.append(f'<details class="issue" id="{i["id"]}" data-group="{h(group)}" data-status="{h(i["status"])}"><summary><b>{i["id"]}</b> {h(i["title"])}<span>{h(i["status"])}</span></summary><div class="body"><p class="meta">{h(group)} · {h(i["component"])} · {i["date"]}</p>{body}</div></details>')
css='''body{font:16px/1.7 system-ui,sans-serif;max-width:1160px;margin:35px auto;padding:0 22px;color:#203448;background:#f7f9fc}h1{font-size:30px;margin-bottom:10px}a{color:#075eae}p{margin:10px 0}.intro{background:#eaf2f8;padding:18px 22px;border-radius:8px}.toolbar{position:sticky;top:0;background:#f7f9fc;z-index:2;padding:12px 0;border-bottom:1px solid #d8e2eb;display:flex;flex-wrap:wrap;gap:9px;align-items:center}input,select,button{font:inherit;padding:7px;border:1px solid #b8c9d8;border-radius:5px;background:white}input{flex:1;min-width:240px}button{cursor:pointer}.issue{background:white;border:1px solid #dbe5ec;border-radius:7px;margin:12px 0;scroll-margin-top:100px}.issue[hidden]{display:none}summary{padding:13px 16px;cursor:pointer}summary b{display:inline-block;min-width:50px;color:#0b527e}summary span{display:block;font-size:13px;color:#5b6e7d;margin-left:55px}.body{padding:0 20px 16px}.meta,.note{font-size:14px;color:#506375}.refs{font-size:14px;overflow-wrap:anywhere}code{background:#edf3f7;padding:1px 4px}.small{font-size:14px}#count{font-size:14px;color:#536777}'''
options=lambda xs:''.join(f'<option value="{h(x)}">{h(x)}</option>' for x in xs)
javascript=r'''const rows=[...document.querySelectorAll('.issue')];const q=document.getElementById('q'),g=document.getElementById('group'),s=document.getElementById('status');
function filter(){let n=0;const terms=q.value.trim().toLowerCase().split(/\s+/).filter(Boolean);for(const r of rows){const match=(!g.value||r.dataset.group===g.value)&&(!s.value||r.dataset.status===s.value)&&terms.every(t=>r.textContent.toLowerCase().includes(t));r.hidden=!match;if(match){n++;if(terms.length)r.open=true;}}document.getElementById('count').textContent=n+' / '+rows.length+' 条';}
for(const e of [q,g,s])e.addEventListener(e===q?'input':'change',filter);
document.getElementById('expand').onclick=()=>rows.filter(r=>!r.hidden).forEach(r=>r.open=true);
document.getElementById('collapse').onclick=()=>rows.forEach(r=>r.open=false);
function reveal(){const id=decodeURIComponent(location.hash.slice(1));const r=document.getElementById(id);if(r&&r.classList.contains('issue')){q.value=g.value=s.value='';filter();r.open=true;r.scrollIntoView();}}
window.addEventListener('hashchange',reveal);filter();reveal();'''
page='<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>FlagTree 问题修复索引</title><style>'+css+'</style></head><body>'
page+=f'<h1>FlagTree 问题修复索引</h1><p>更新于 {latest} · 原始编号保留 · 可按症状、报错、算子、源码路径查询</p>'
page+='<p><b>持续执行要求：</b>每次遇错及修复前先查本手册，核对已有方案的适用版本与部署范围；新方案验证后登记复现、前后结果、指纹与边界。同一机制补旧条目，未验证方案不标已解决。</p>'
page+='<div class="intro"><p>'+h(scope)+'</p><p>旧 101 项中：13 项源码修复验证、8 项绕行验证、56 项本地实现或处置、24 项历史／待验。通过测试、安装到某镜像、官方合并是不同状态。</p><p>后续已复用 copy_ 旧过滤（869 项），修复 grid 缓存并复用已有 empty/lift_fresh。6 项缓存回归、27 项算子检查和 6 项模型对照通过；小 grid 异常见 D05。请优先查 <a href="#D01">D01</a>、<a href="#D02">D02</a>、<a href="#D03">D03</a>、<a href="#D04">D04</a>。</p><p class="small">M 系列为另一路 f06.compile_model 包验证；不与 D 系列 FlagGems 全量 ON 的测量混算。历史官方状态以原记录日期为准。</p></div>'
page+='<p>'+a(ref('Markdown 完整文档',md_path))+' · '+a(ref('结构化索引',OUT/'index.json'))+' · '+a(ref('本次 FlagGems 完整诊断',DIAG/'report.html'))+' · '+a(ref('后续整模型报告',MODEL/'report.html'))+'</p>'
page+='<p>官方 FlagGems 36a2e186a 的本地 p800fix4 已完成 LayerNorm-384 与矩阵准备路径优化，FlagGems/copy_ 保持开启。187 项检查通过、4 项既有问题 xfail；预留 GPU6 后三方共 9 条模型路径及换输入图重放通过。QKV encode 13.157→10.561 ms，设备 6.360→3.002 ms，仍慢于 OFF 6.568/1.649 ms。mask 优化见 <a href="#D08">D08</a>，正式复测见 <a href="#D10">D10</a>，非方形序列边界见 <a href="#D09">D09</a>。不表示官方已合入；'+a(ref('完整优化报告',MINILM_RESIDUAL/'README.md'))+'。</p>'
page+='<div class="toolbar"><input id="q" aria-label="搜索问题" placeholder="搜索：copy_、grid、BF16、U09…"><select id="group" aria-label="按归属筛选"><option value="">全部归属</option>'+options(groups)+'</select><select id="status" aria-label="按状态筛选"><option value="">全部状态</option>'+options(sorted({i['status'] for i in items}))+'</select><button id="expand">展开结果</button><button id="collapse">收起</button><span id="count" aria-live="polite"></span></div>'
page+=''.join(cards)+'<p class="small">更新记录时保留触发条件、版本指纹、修补位置、复现与前后验证、部署范围；不要把过滤规避、诊断或性能测量改写成底层已根修。</p><script>'+javascript+'</script></body></html>'
publish_text(ROOT/'reports/FlagTree问题修复索引.html',page)
print(json.dumps({'records':len(items),'groups':inventory['group_counts'],'markdown':str(md_path)},ensure_ascii=False,indent=2))
