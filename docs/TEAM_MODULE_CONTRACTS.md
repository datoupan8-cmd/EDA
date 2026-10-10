# 团队模块接口与维护边界

公共顺序：Image → Text → Component → Pin Localization → Pin Semantics → Wire → Topology → Submission。

## 谁改哪里

| 负责人 | 算法目录 | 自己的注册文件 | 自己的默认选择 | 新测试 |
|---|---|---|---|---|
| 元件 | `pcb/component/`、`pcb/text/` | 各目录 `registry.py` | `configs/component/current.json` | `tests/component/` |
| 引脚 | `pcb/pin/` | `pcb/pin/registry.py` | `configs/pin/current.json` | `tests/pin/` |
| 导线与网络 | `pcb/wire_stage/`、`pcb/topology_stage/` | 各目录 `registry.py` | `configs/wire_topology/current.json` | `tests/wire/`、`tests/topology/` |

`pcb/core/`、`pcb/schema.py`、Submission、坐标转换和原评测器属于公共契约；普通算法迭代不修改它们。历史根目录文件有兼容导入，看到旧名字时不要在兼容文件里另写一套算法。

## 输入输出

| Stage | 输入 | 标准输出 |
|---|---|---|
| Text | 原图、Context | TextStageOutput：texts / diagnostics / debug |
| Component | 原图、TextStageOutput、Context | ComponentStageOutput：components / texts / roles / diagnostics / debug |
| Pin Localization | 原图、ComponentStageOutput、Context | PinLocalizationOutput：有序 `(Component, terminals)` 列表 |
| Pin Semantics | ComponentStageOutput、Localization、Context；需要局部OCR时显式接收原图 | PinSemanticsOutput：components（含pins）/ diagnostics / debug |
| Wire | 原图、Scene、Context | WireStageOutput：mask / suppressed / corridor / diagnostics / debug |
| Topology | Scene、WireStageOutput、Context | TopologyStageOutput，并按原行为填充 Scene.nets |
| Submission | Scene、Context | 原官方JSON结构 |

所有标准容器继续使用 `pcb/core/interfaces.py`，领域数据继续使用 `pcb/schema.py`。不复制新的 Component / Pin / Scene 类。

## 坐标与字段

- 图像、bbox、body_bbox、terminal.tip/base、Pin.tip/base 均为原图左上角原点的像素坐标，顺序为 x/y；不在Stage里翻转Y轴。
- bbox格式为 `(x1,y1,x2,y2)`；body_bbox沿用现有器件本体边界含义，不扩大成文字框。
- base是现有候选的器件侧基点；tip是供后续处理及Submission转换使用的引脚定位点。不能把线网最远端自动当tip。
- side为 left/right/top/bottom；number、name、exportable 等字段保持原语义。
- 官方坐标转换只由原Submission/coordinates完成；评分仍是原 `evaluate_v2.py` 的本地非官方口径。

## 需要原图的Pin语义版本

旧版本继续实现 `run(component, localization, context)`。需要局部文字读取的新版本实现 `run_image(image, component, localization, context)`。

公共Pipeline只调用 `run_pin_semantics` 接口桥接函数。桥接按方法能力传入原图，不识别任何版本名称、阈值或模型。L39已采用这个接口，不再把图片或Scene塞入万能Context。

Context.resources仅用于设备、模型路径和缓存路径。components、terminals、pins、wires和nets必须走明确的Stage参数及返回值。

## 默认保留的实现

- `peer_latest`：同学 `d648403` 的最新Component/Text实现，函数逻辑原样迁移。
- `l39`定位：V3非box分支＋原L1 box定位模型。
- `l39`语义：原BoxBBox/E6/L3完整词读取/L36联合保序/L37公开元件库约束。
- `v3_frame_guard`：原Wire V3＋已验证F框线隔离；不是本轮新增算法。
- Topology V3、原Submission保持不变。
- 同学Pin V4独立命名为 `peer_v4`，原Pin V3/V4～V9保留，不抢占同名实现。

L39内部已有临时评分函数替换并在finally恢复。本轮保持其串行行为；不可擅自并行执行同一进程中的多个L39语义调用。

## 新增一个版本的最小操作

1. 在自己的算法目录新增版本文件，实现上述接口。
2. 在自己的 `registry.py` 增加一个唯一名称；公共core注册表无需改。
3. 在自己配置片段中选择新名称；`configs/team.json`无需改。
4. 加自己的测试，执行完整工程测试和开发集A/B。

配置加载器限制每个片段可改变的字段。例如元件配置不能偷偷选择Pin版本，Pin配置不能改变Wire策略。

## 隔离的含义

隔离保证源码目录、版本名和日常配置选择互不覆盖，不代表上游输出变化对后续指标完全没有影响。Component的数量、文字和bbox变化会影响Pin输入；这通过同一完整工程的A/B检测，不能靠隔离目录消除。
