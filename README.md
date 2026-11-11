# QAproject：海洋语料处理工作台

更新日期：2026-11-11。当前源码版本为 v36，尚未部署服务器。

系统提供资料接入、可追溯的 Corpus 整理、知识条目、主题检索和已有 QA/Instruction 加工能力。参考 OceanPile 的实际文件结构与公开方法；格式兼容不等于完整复现浙大内部算法。

本说明只描述当前方案和实际边界，不保留已经被替代的设计。详细业务规范见 [业务开发规划](docs/OceanPile全流程业务开发规划.md)。本轮接通共享资料准备：勾选生成QA即自动归档资料，再复用冻结证据生成QA。QA的提示词、审核、编辑及受控提交策略保持原有实现。

## 运行

需要 Python 3.12。每台机器独立创建环境，不跨 macOS、Windows、Linux 复制 `.venv`。

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
cp config/env.local.example .env.local
# 在本机编辑 .env.local，填写 API_KEY；不要复制到源码或前端。
.venv/bin/python run.py
```

Windows 使用 `py -3.12 -m venv .venv`、`.venv\Scripts\python.exe`；复制配置可用 `Copy-Item config/env.local.example .env.local`。macOS 可使用 `启动QAproject.command`。

默认访问 `http://127.0.0.1:8765/`。命令参数：

```sh
.venv/bin/python run.py --no-browser
.venv/bin/python run.py --data-dir /persistent/qa-data --library-root /persistent/library
```

- `--data-dir`/`QA_RUNTIME_DIR`/`runtime_dir` 可指定持久目录，默认 `runtime/`。修改目录不自动搬迁历史数据。
- 不配置资料库时不扫描服务器目录，本机选择文件继续可用。
- 远程访问还需正确配置监听地址、`access.allowed_hosts`、`access.allowed_origins` 和网络权限。本版没有登录功能，不表示已完成生产部署。
- 一份数据目录只允许一个应用进程。更新后停止旧服务并重新启动，不通过另开端口叠加工作进程。

## 四个模块

| 模块 | 功能 |
| --- | --- |
| 文件处理 | 选择/拖入文件、从资料库选择、网页接入、独立任务记录及重试 |
| 主题模糊搜索 | 关键词和正式主题检索、本地语义主题推荐；空查询不显示结果 |
| QA汇总 | 保留原有双语QA与现有指令加工、详情、编辑重审、受控入库与导出 |
| 资料汇总 | 资料成果、知识条目、知识层级 |

首页三项统计只计算实际投入资料。内部演示资料与正式工作区隔离，不自动注入正式页面。

文件处理提供三个可组合勾选项：**生成QA对、整理知识条目、整理资料**。默认仅选QA，至少选择一项。生成QA会自动保存资料成果，无需再勾选整理资料。知识条目仍需单独勾选。资料准备成功后QA及所选知识任务才可领取，资料失败时等待重试，不各自重新解析。同文件任务互斥；QA仍使用原双语QA流程，不等于启动另一套指令生成。生成指令的现有入口保留。

## Corpus 整理

### 来源与版本

- 文件上传保存原件；资料库文件只登记原位置，不复制进 `originals`。
- 来源位置、内容hash、冻结证据和加工版本保持关联，原件改变或失效时阻止生成、提交和导出。
- 资料版本不可变。“重新整理”创建新版本，保留历史成果，不改写已有QA原文、图片或审核记录，也不自动发起QA生成。
- “资料已归档”表示结构与文件hash校验通过，**不代表全文科学内容已经审核通过**。
- 列表只加载摘要；正文、图片、结构和标注在详情读取。

### 网页

网页接入接受指定的 HTTP/HTTPS 公网网址，保存原始HTML/文字快照。程序负责抓取，GLM负责依据实际正文整理知识和审核；目前没有自动搜索互联网或从论文术语自动寻找百科解释的流程。

Corpus 的HTML处理优先 `article/main` 正文，保留标题、科学条件、表格单元分隔、图注，排除导航、脚本、部分广告结构、明确占位文本与完全相同的重复长段落。删除记录保存在版本加工记录中；不按固定最短长度丢弃简短定义，不截断长科学段落。

网页配图：

1. 从正文提取图片网址与图注，排除明显图标和装饰来源。
2. 每次请求及跳转重新核验公网地址，固定校验后的IP连接；不允许内网、带凭据地址或任意端口。
3. 只接受真实PNG/JPEG/WebP，原字节保存；单图5MB、4000万像素限制，不自动裁剪。
4. 通过公司GLM适配器传入真实图片及网页上下文，判断相关性与可用性；未经审核的图片不进入收录单元。
5. 单页最多采集12张、累计20MB；超过范围、失败和被排除的图片分别记录，不声称全部采集成功。

网页正文请求上限8MB、单次抓取总时限35秒；网页与图片采集并发受控。动态渲染、登录页面、SVG和动画图片不在当前支持范围。重新整理复用已成功抓取及审核的图片缓存，不保证刷新同一URL已发生变化的远程图片。

### 论文与文档

新Corpus沿用原有`pdf_layout`与MinerU解释器/模型配置。当前配置为`auto`，有可用MinerU或匹配结果时优先MinerU；`off`使用原生，`manifest`只用匹配缓存。自动模式不可用时的原生兜底会明确记录。完整页图保留，不裁剪：

| 解析方式 | 当前能力与限制 |
| --- | --- |
| 结构化原生解析（按原配置兜底或显式选择） | PyMuPDF读取文字行、字体、坐标和原生表格，推断标题层级；识别重复页边文字和页码，输出带页定位的Markdown阅读稿 |
| 本地MinerU（当前auto配置优先） | 复用已有解释器、模型和匹配的历史MinerU JSON；没有结果时执行本地解析，保存文字、标题、公式与表格并检查页面完整性；不启动旧裁图流水线 |

原生方式不能保证完整恢复复杂版面、公式LaTeX或扫描文字；扫描原图继续保留，不虚构OCR。显式选择MinerU但环境不可用、或MinerU实际执行失败/超时，会明确失败，不冒充成功。安装与配置使用 `scripts/setup_mineru.py`、配置的 `mineru.python` 或本机 `MINERU_PYTHON`；不自动安装或下载模型。本机原有MinerU环境已经存在并可用；本轮也已用同一环境实际解析隔离PDF验证新共享链路，未重装或修改模型。复杂长文档和全库识别质量仍需分别验收。

Markdown、结构、清洗及模型判断记录随版本保存。它们是本项目加工成果，不宣称与浙大使用的MinerU版本及内部规则完全相同。

### 近重复处理

当前在同一份新Corpus内，以局部文本相似度筛选160–6000字符的近重复候选，再用公司GLM确认是否完全等义、后者是否包含新增事实；数值或否定表达不同直接保留。只合并确认等义的段落，保存两段原文、ID和审核依据。模型结果缓存绑定输入，失败可续跑。

这不是全库语义聚类或跨资料实体合并。没有候选不增加该模型请求；短段落和超长段落不进行语义合并，科学事实保留优先。原始资料与旧冻结证据不变。

## 知识条目

处理顺序：准备冻结正文 → 公司GLM同时提取实体和原文支持的解释 → 程序校验片段与字段 → 独立GLM审核 → 仅通过项进入正式知识表。

- 解释来自当前资料，不借用模型常识或自动联网补全；论文事实保留研究范围、条件及不确定性。
- 八类实体：`Location / Instrument / Organism / Process / Phenomenon / Substance / Property / Theory`。
- 实体类型不同于正式主题，也不同于知识层级；不把论文标题、目录或“被研究”等空泛文字当知识。
- 同一文件同名同类型实体可以汇集已审核事实，不跨文件、别名或相似名称猜测合并；条件和不同结论分别保留。
- 详情区分网页知识、论文/文档事实、图片与数据资料。论文实体不会被伪装为网页采集成果。
- 审核绑定原始行hash、证据hash和逐字引用；失败候选只留内部审计。

正式CSV严格七列：

```text
entity_name,entity_type,entity_description,page_urls,page_titles,full_text,full_text_length
```

**统一全文口径**：知识条目导出与网页Corpus导出共用来源正文组装规则。先找到与事实证据hash匹配的冻结资料版本；没有匹配Corpus时使用该知识任务冻结的全部正文；导入知识CSV时只汇集该实体对应来源记录的全文，不混入其他网页记录。按原顺序连接非图像可读段落，完全相同段落只保留一次，以空行分隔；`full_text_length`为实际字符数。

`entity_description`仍汇集已审核事实；全文不是模型生成。原始行、引用片段及审核hash不改写，详情通过独立导出视图显示统一正文。缺失正文绑定时拒绝导出，不混用另一版本或悄悄退回少量摘录。来源网址只使用真实登记值，普通论文无网址时保持空值。此口径是明确的本项目规则，不声称复现官方未公开的全文清洗算法。

## 图片与参考标注

图片可登记类别、框和图注，必须填写真实来源；默认未人工确认。定位框明确为像素或归一化坐标，并检查真实图片边界。

支持两种JSON导入，每批1–100条：

- 内部格式：`unit_id,kind,value,verified,source_note`。
- 参考格式：生物的 `image_path,category,prompt`，声呐的 `image_path,object,bnd_box,input,output`。`image_path`仅映射已登记图片，不用于任意路径读取。类别可通过用户明确提供的 `class_map` 统一，原类别和参考文本保留。

导出图片/数据类资料包包含实际图片、完整内部标注、`Open-Dataset/OceanVision/OceanVision_train_data.json`、`Open-Dataset/OceanSona/sona_train_data.json` 和说明。图片路径从包根目录解析；同一图片不重复打包。仅人工确认类别/框进入对应参考文件，未确认标注仍保留内部审计。目标框导出为归一化 `[x1,y1,x2,y2]`。

没有原始参考问题时，`prompt/input/output`保持空值，不自动生成QA或伪装成可训练成品；已有参考文本原样保留，也不等于通过本项目QA审核。

## 对应导出

| 成果 | 正式结构 |
| --- | --- |
| Web | 上述七字段，`Web/wiki_updated_full.csv`，UTF-8无BOM、CRLF；原始快照及收录配图随包 |
| 真实PDF论文 | `paper_id,title,pdf_path,md_path,images_dir,image_count`，UTF-8带BOM、CRLF |
| 论文资料包 | `Paper/Pdf/`、`Paper/CleanedData/papers.csv`、`Paper/CleanedData/ProcessedData/<id>/`，索引路径与文件核对 |
| 图片/数据资料包 | 原件、证据、加工清单、图片及参考标注，不伪造论文索引或通用官方CSV |

参考：OceanCorpus revision `fecbddc5ce3900931c90a2bc87ac9f4964de7843`。[实际目录](https://huggingface.co/datasets/zjunlp/OceanCorpus/tree/main)的Web文件是`wiki_updated_full.csv`，官方README仍有旧`data.csv`表述；以实际文件核对格式，以[论文](https://arxiv.org/html/2605.00877v1)理解方法。官方未完整公开实体初始清单、解释生成提示词及全文合并规则，不自行断言。

## QA和Instruction：共享资料准备，保留生成审核策略

原QA保留中英配对生成、实际文字/图片证据、模型审核与修订、候选核对、受控入库、编辑冲突保护与重审、原文预览和六字段导出。内部主题、配对ID、审核状态不进入正式列：

```text
input,thinking,output,source,image,language
```

新QA入口先准备Corpus，QA及可选知识条目共用同一冻结证据；原生PDF在首次证据读取时一并取得结构信息，不再独立重开PDF提取结构。资料准备完成立即保存检查点，后续质量接口或归档失败重试可复用。资料失败时依赖任务不领取，重试自动接续。已经冻结的历史QA仍使用原证据；历史资料缺少结构时允许一次补充整理，显式重新整理或更换解析引擎不属于同一次解析复用。一次准备不等于只读取一次文件字节，hash复核及必要渲染继续保留。更新前已有队列和历史QA不自动批量补归档，新入口按新流程执行。原件和图片是真实持久路径，`source`保留真实页段或独立图片定位，不虚构非分页文件页码。

现有指令加工保留以下子集，不因资料整理修订而重新生成或改写历史候选：

| 子集 | 现有正式导出列 |
| --- | --- |
| Science | `input,thinking,output,source` |
| MultiModel-Science | `input,output,image_path,language` |
| Sonar-Open | `input,output,image_path,language,source` |
| Bio | `input,output,image_path,source` |

现有子集CSV为UTF-8无BOM、LF；语言列使用`Chinese/English`。图片随ZIP提供相对路径；source为实际项目来源标识，不冒用官方来源。知识层级目前手工登记并关联来源，不冒充完整OCG；补充知识从已登记资料中选择。Benchmark、专家评分平台、训练平台不在本轮开发范围。

## 调度、检索和数据保护

- 所有后台文件共用统一队列，默认最多2份文件、全应用最多2个模型请求；同文件互斥。
- PyMuPDF准备及论文结构读取使用单个原生解析线程；不多开应用进程叠加并发。
- 文档、知识和质量审核统一使用公司GLM适配器及允许配置，不切换其他远程AI。
- 本地Embedding仅用于已有主题短文本推荐及已有QA查重用途；本轮没有用它向量化Corpus全文。文件任务优先，嵌入让位。模型缺失按已有质量规则提示，不伪装实测。
- 正式主题按整篇概括，中文名称；关键词有独立文件关系。细词检索不能扩展到整个宽主题的无关资料。
- 任务保存租约、阶段、脱敏错误、成功检查点和审核缓存。错误文件不阻断其他文件；中断后恢复未完成工作。
- 密钥仅在每台机器的`.env.local`由应用读取。不得进入源码、数据库、日志、前端、导出包或测试固定数据。
- `.env.local`、`runtime/`、原始文件、图片、数据库、模型和`.venv`不提交GitHub。`config/env.local.example`保持`API_KEY=`空值。
- Windows默认SQLite DELETE模式，macOS/Linux默认WAL；可通过`QA_SQLITE_JOURNAL_MODE`显式指定。不要手动删除运行中的WAL/SHM。

## 备份与恢复

先停止服务。备份包含业务库及引用的内部文件，不含源码配置和密钥；外部资料库原件另行备份。

```sh
.venv/bin/python scripts/manage_data.py backup /backup/qa-snapshot --data-dir /persistent/qa-data
.venv/bin/python scripts/manage_data.py restore /backup/qa-snapshot --data-dir /persistent/qa-data
```

恢复要求快照绑定的原数据路径且目标工作区为空；不静默重写历史source/image，不能以换目录代替搬迁流程。当前未提供跨路径自动恢复。

## 主要接口

| 操作 | 接口 |
| --- | --- |
| 文件/资料库接入 | `POST /api/imports`、`POST /api/library/imports`；重复`processing=qa/knowledge/corpus`参数表示多选 |
| 网页接入 | `POST /api/corpus/web` |
| 整理/重新整理 | `POST /api/documents/{id}/corpus`，`reprocess:true`创建新版本，`pdf_engine:configured/native/mineru`选择论文解析 |
| 资料列表/任务/详情 | `GET /api/corpus`、`/api/corpus/jobs`、`/api/corpus/{version_id}` |
| CSV/资料包 | `GET /api/corpus/csv/{version_id}`、`/api/corpus/export/{version_id}` |
| 内部标注 | `GET/POST /api/corpus/{version_id}/annotations`、`POST .../annotations/import` |
| 参考标注导入 | `POST /api/corpus/{version_id}/annotations/reference` |
| 知识条目 | `GET /api/knowledge`、`/api/knowledge/{id}`、`/api/knowledge/export` |

写入保留会话和来源检查；只接受已登记ID或来源版本，不开放任意路径读取。旧单值`both`、`corpus_qa`保持接口兼容；后者仍是已有Corpus后接Instruction的历史接口，当前多选入口不使用它。

## 验证与边界

当前完整462项自动测试通过，包括原QA回归、网页正文、图片字节、两处导出一致、论文结构、标注路径/类别映射、版本恢复及历史证据保护。自动测试中的MinerU和语义等价判断使用替身；另外已完成一份真实本地MinerU解析验证，不把单份验证扩大为全面质量验收。 共享准备专项使用真实PyMuPDF解析隔离PDF，核对原生解析仅一次、QA阶段不再次解析；模型部分为替身。另验证归档/质量失败后的检查点恢复、父任务失败阻止下游领取及后补知识任务依赖。记录见`runtime/shared-preparation-20261111/report.json`。

另在隔离目录实测NOAA网页：26个证据单元，收录1张真实配图并通过公司GLM审核，排除1个小图标；知识整理最终完成，5个通过实体，两处CSV内容及全文长度核对一致。过程中1次网络错误，续跑复用了完成批次；没有生成QA，隔离库QA和候选均为0。报告：`runtime/corpus-check-20261111/report.json`。正式资料未操作，服务未重启，未做本轮浏览器目视或Windows/Linux实机验收。真实本地MinerU另测一份隔离PDF：默认继承auto配置、成功产出4个文字单元；报告`runtime/mineru-shared-check-20261111/report.json`，无QA候选。

```sh
.venv/bin/python -m pytest tests/test_corpus_processing.py tests/test_corpus.py tests/test_corpus_api.py tests/test_knowledge.py tests/test_web_sources.py -q
.venv/bin/python -m pytest -q
```

Windows/Linux仍需分别做安装、解析、公司接口、入库导出、任务和备份恢复实机验收。当前不是服务器部署完成证明，也未完成超大资料库吞吐验收。

## 共享解析的官方依据与工程边界

论文Methods先描述数据预处理，再在Instruction Data Synthesis以文档、图片及其描述/标签为输入，结合OCG和权威补充资料生成指令。因此采用“基础资料先准备，下游任务复用”的业务关系。公开资料没有承诺内部只打开PDF一次，也没有要求所有文档先转为Web知识条目；共享一次证据准备及任务依赖是本项目实现选择，不能称为官方公开的调度或缓存算法。已有QA模式仍不是完整OCG知识增强流程。
