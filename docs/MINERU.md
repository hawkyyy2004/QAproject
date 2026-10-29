# MinerU 版面清单：安装与使用

项目原先遇到含图页就整页 144 DPI 截图，因为 PyMuPDF 只能告诉你"这页有矢量线或位图"，说不出哪一块才是图。MinerU 补的正是这一段：它按版面把每页切成正文、图、表、公式，我们只裁这几块，正文继续走文字证据。

服务会自己调用 MinerU：上传后的文件在解析阶段先做一遍版面识别，写出 `<PDF hash>/manifest.json` 和裁图，再按块出证据。也可以提前用脚本批量生成清单（离线预跑），服务到时直接复用，见第二节。

没有 MinerU 的机器照旧整页处理，功能不会因此报错。

MinerU 装在独立虚拟环境里，服务进程不导入它，`requirements.txt` 也不加它。

## 一、首次安装

```powershell
python scripts/setup_mineru.py            # basic：版面模型，约 0.8 GB
python scripts/setup_mineru.py --tier both  # 再加 standard 主模型，约 1.2 GB
```

脚本会在 `runtime/mineru-venv` 建虚拟环境、安装 `mineru==4.0.12`，并从 ModelScope 下载模型到 `C:\Users\<你>\.mineru\models`。模型不进 git；第二台机器可以直接把这个目录整个拷过去。

模型来源必须是 ModelScope，否则会去连 HuggingFace，在国内经常卡住：

```powershell
$env:MINERU_MODEL_SOURCE = 'modelscope'
```

`scripts/mineru_layout.py` 默认就用这个来源，手工调 `mineru` 命令时才需要自己设。

## 二、生成清单

**默认：不用你动手。** 上传 PDF 后，服务在解析阶段调用 MinerU（`pdf_layout=auto`），清单落在 `--layout-dir` 指向的目录，同一个文件按内容 hash 只做一次：再次解析直接复用清单，不重复耗时。

首次解析会多花 1–3 分钟，界面进度显示"正在按版面拆图"。

要提前批量预跑（比如晚上挂机先把 E 盘几十篇跑完，白天上传时秒过）：

```powershell
# 一个目录（递归找 PDF），4 个文件并行
python scripts/mineru_layout.py "E:\pdf\海洋学报" --jobs 4 `
    --mineru-python runtime\mineru-venv\Scripts\python.exe `
    --layout-dir runtime\live\layout

# 已有路径清单（中文带引号的文件名建议走这种）
python scripts/mineru_layout.py --list pdfs.txt --jobs 4 `
    --mineru-python runtime\mineru-venv\Scripts\python.exe `
    --layout-dir runtime\live\layout
```

`--layout-dir` 要和服务的数据目录一致：本机默认是 `runtime/live/layout`，即 `runtime/live/state.sqlite3` 所在的那一层。

| 参数 | 说明 |
| --- | --- |
| `--tier basic` | 版面 + OCR + 公式，默认；`standard` 更准但更慢更占内存 |
| `--dpi 200` | 裁图分辨率；表格多、字号小可以提到 300 |
| `--jobs 4` | 同时解析的文件数；受 CPU 与接口内存限制，4–6 是安全区间 |
| `--force` | 已有清单也重做 |
| `--limit N` | 只做前 N 个，适合先试跑 |

预跑与服务自动生成写的是同一份清单，谁先做谁生效，不会冲突。

每完成一个文件打印一行 JSON：`ok` 正常、`skipped` 已有清单、`failed` MinerU 报错、`rejected` 识别结果明显不全（见下）。整批可随时中断重跑，已完成的文件会跳过。

## 三、清单长什么样

```
runtime/live/layout/<pdf sha256>/
  manifest.json
  mineru.json          MinerU 原始输出，排查用
  crops/
    page-02-fig-01.png
    page-04-tbl-01.png
    page-03-eq-01.png
```

`manifest.json` 每页一组块：

| kind | 证据形态 | `text` 内容 |
| --- | --- | --- |
| `text` | 纯文字单元 | 段落原文 |
| `figure` | 图片单元 + 裁图 | 中文图注 + 英文图注 |
| `table` | 图片单元 + 裁图 | 表注 + HTML 表体 |
| `formula` | 图片单元 + 裁图 | LaTeX |
| `page` | 整页扫描 | 图注（通常为空） |

服务解析时会把裁图复制到 `assets/images/<文件ID>/`，所以 `layout/` 只是中间产物，导入完成后删掉也不影响已入库数据。

## 四、规则与兜底

- **面板合并**：同一张图的多个面板（上下叠、左右并排、2×2 网格）合成一张裁图；两个各自已有图注的块不会合并，避免把两张独立图糊在一起。
- **图注继承**：面板组自己没图注时，向上或向下 3% 页高内找"图 N / Fig. N / 表 N / Table N"开头的文字块，中英两行一起挂上，并从正文里移除，避免同一句图注再出一次题。
- **图内标注**：落在图框里的方程块（如 θ0=40°）并入图，不单独成为公式证据。
- **公式留白**：公式框上下各加 6pt，避免下标被切掉。
- **保底判断**：MinerU 识别出的字符数不到 PDF 文字层的一半，就直接拒写清单，该文件继续整页处理。整页处理是安全的老路径，不会因为 MinerU 漏页而丢证据。
- **页级兜底**：清单里没有的页（扫描页、MinerU 解析失败的页）仍按整页逻辑出证据。

## 五、开关与回退

`config/settings.json`：

```json
"pdf_layout": "auto",
"mineru": {"python": "", "tier": "basic", "dpi": 200, "timeout_seconds": 900}
```

- `pdf_layout = auto`：有清单就用；没有就当场调 MinerU 生成；生成失败才整页兜底。
- `pdf_layout = manifest`：只用已有清单，服务不调 MinerU。适合 MinerU 装在内网另一台机器、清单靠预跑同步过来的部署。
- `pdf_layout = off`：忽略清单，完全回到整页行为，不需要删任何文件。

`mineru.python` 留空时按 `MINERU_PYTHON`（可在 `.env.local` 写）→ `runtime/mineru-venv` → 当前解释器 的顺序找，并且会 `import mineru` 探活，探不到就当没装、走整页。`tier`、`dpi`、`timeout_seconds` 只影响服务自己生成的那一份清单。

改完配置后重启服务才生效。探活失败另带 5 分钟冷却：装好 MinerU 或改对 `mineru.python` 后，重新处理该文件即可生效，不必为一次探测失败重启服务。

判断这次到底走没走 MinerU，看两处：文件列表与详情页的提示条（`documents.layout_note`，清单生成成功后自动清空），以及该文件的事件流 `layout_prepared`。有 `pages>0` 就是按图裁的；`note` 写明失败原因（大多数情况用"未找到可用的 MinerU 解释器"就能定位到 `mineru.python` 配错）。

在中文 Windows 上，探测与被调脚本的输出一律按 UTF-8 解码（子进程环境另外注入 `PYTHONIOENCODING=utf-8`、`PYTHONUTF8=1`）。否则未开启"Beta: UTF-8"的机器会用 GBK 解码子进程输出，抛 `UnicodeDecodeError`，表现为"点重试 0.0x 秒就失败、怎么重试都一样"。

## 六、性能与容量（本机实测）

机器：i7-12700H、16 GB、RTX 3060 6 GB（只用 CPU）、MinerU 4.0.12 basic 档、200 DPI、海洋学报 5 篇 37 页。

| 项目 | 数值 |
| --- | --- |
| 单篇耗时 | 86–171 秒（4–15 秒/页，公式密集更慢） |
| 裁图 | 89 张，约 6 MB |
| 同批整页渲染（144 DPI） | 约 25 MB |
| 识别结果 | 27 图、14 表、51 公式 |

全库 6.8 万篇、约 76 万页，按 4 并发估算 10–12 天；只处理含图表页可压到 3–4 天。存储按裁图估算约 60 GB，整页方案约 200 GB。

## 七、常见问题

**`failed`：`No module named mineru`**
`--mineru-python` 指错了，要指向 `runtime/mineru-venv/Scripts/python.exe`，不是项目原来的解释器。

**`failed`：下载模型超时**
确认 `MINERU_MODEL_SOURCE=modelscope`；或从有模型的机器拷 `~/.mineru/models`。

**`rejected`**
MinerU 这轮识别明显不全（多为扫描质量差或公式版式异常）。该文件继续整页处理；真要抢救可以 `--tier standard --force` 单独重跑。

**图注为空**
MinerU 自己没识别出图注时不会凭空补。裁图仍然是有效证据，模型会直接读图。

**导入后再改清单**
已入库的证据是冻结的（含 hash 校验），重做清单不会改写历史。文件还没入库（草稿/待审）时可以用脚本重建证据，不用重新上传：

```powershell
# 先看会动哪些文件，不写库
python scripts/rebuild_evidence.py --data-dir runtime/live --all-unapproved
# 确认后执行：清空该文件的证据与草稿，重新排队解析
python scripts/rebuild_evidence.py --data-dir runtime/live --all-unapproved --apply
```

有已批准问答或已生效修订的文件会被跳过，脚本只打印原因。
