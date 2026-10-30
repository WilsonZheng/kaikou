# Kaikou（开口）

开口就能打字。一个只给自己用的 macOS 语音输入工具：按一下左 Control + Shift 开始说话，说完再按一下，整理好的文字直接出现在光标处。全部在本机运行，不花钱；声音和文字都不会离开你的电脑（唯一的联网是用 agent-slack 读你自己的 Slack，见下文）。

对标 Typeless，只做一件事：中英混说的语音转文字，并且像 Typeless 一样把口语整理成通顺的文字（去口头禅、合并改口和说重了的话、长段落分段，并列的几点或几个步骤整理成列表）。

## 用法

| 操作 | 效果 |
|---|---|
| 快速连按两下 fn | 开始录音（默认快捷键，不用一直按着） |
| 再按一下 fn | 结束，识别、整理，然后粘贴到光标处 |
| 录音时按 Esc，或点波形旁边的“取消” | 取消 |
| 按住 ⌃⌥Space | 备用快捷键，不需要任何权限，结果放进剪贴板 |

fn 按住不放、或者和别的键一起按（比如 fn+F1 调亮度），都不会触发。如果 macOS 自带的“听写”也设成了连按 fn，请在系统设置 → 键盘 → 听写里把它关掉或换成别的键。

**改快捷键**：面板 →“设置”→“录制新快捷键”，直接按你想用的组合，立即生效，不用重启；“恢复默认”回到连按两下 fn。可以用两个以上修饰键的组合（左⌃ 左⇧、⌥⌘…）、右边的单个修饰键（右⌥）、fn，或修饰键加一个键（⌥Space、⌃`，会被 Kaikou 独占）；还可以选“按一下开始、再按一下结束”或“按住说话、松开结束”。太容易误触的键（比如单独一个左 ⇧）会被拒绝。

录音时屏幕下方有一条跟着你声音跳动的短波形和一个小“取消”；结束后变成三个跳动的小点，表示正在整理。

打开面板：点菜单栏的麦克风图标 →“打开面板”；菜单栏图标被刘海挡住时，用 Spotlight 搜“Kaikou”回车也会直接打开面板。面板里有：

- **概览**：今天和全部的次数、字数、省下的打字时间、松手后等待时间、最近 14 天的字数和一段分析总结。
- **词库**：你的词汇（可以搜索、添加、删除）、自动学会的纠错、所有纠错规则。
- **历史**：最近的听写，点一下复制。

## 越用越准

- **自我学习**：识别错了，直接在输入框里改掉就行。Kaikou 会在粘贴后观察那段文字，发现你把某个词改掉了，就记下来：改过一次，下次作为提示交给整理模型，并把正确的词放到识别热词最前面；同样的修改出现两次，就变成自动替换。学到的内容在面板的“词库”里，学错了可以删掉。只能观察能读取文字的输入框（大部分原生 App、Slack 等），终端里读不到。在 Orca 这类终端里说给 Claude Code 的话，Kaikou 每小时会把你实际发出去的消息（在 Claude Code 的对话记录里）和它当时输入的文字对比一次：你在发送前改掉的词（比如把 Hoppen 改成 Hoffman），也会照样学会。普通英文单词和两个字的中文词只当提示，不会变成自动替换，免得误改正常的话。不想要这个功能，在 `config.json` 里写 `"agent_learning": false`。
- **结合上下文**：开始录音时会读当前窗口的标题和光标前的一小段文字，帮助整理模型判断人名、仓库名怎么写（只参考，不写进结果）。
- **知道你在聊谁**：在 Orca 这样的终端里读不到窗口文字，所以 Kaikou 还会看最近 30 分钟里有更新的 Claude Code 对话记录、你最近两天在 Slack 上跟谁聊过什么，以及你最近三天说过的话，挑出里面的人名、产品名和仓库名（比如正在和 Hoffman 讨论 ezyVet site），在识别时优先往这些词上靠。整理时，转写里出现读音很像其中某个名字的词（比如 Hopon、Hoppen），会提示整理模型多半是 Hoffman。人名的大小写也会统一（HOFFMAN、ezyvet 写成 Hoffman、ezyVet）。Slack 通过 [agent-slack](https://github.com/stablyai/agent-slack) 读取（它用 Slack 桌面版的登录状态），平时每 4 小时在后台刷新一次，你在 Slack 里用过语音输入后 30 分钟内会提前刷新一次；只在你没说话时刷新，并且以后台低优先级运行，不会拖慢识别。消息只留在内存里；没装 agent-slack 就自动跳过。不想让它读 Claude Code 记录或 Slack，在 `config.json` 里写 `"agent_context": false` 或 `"slack_context": false`。

## 安装

需要 Apple Silicon Mac、Xcode 命令行工具和 [uv](https://github.com/astral-sh/uv)。不需要管理员权限。

```bash
./scripts/install.sh
```

脚本会编译 App 并装到 `~/Applications/Kaikou.app`，在 `~/Library/Application Support/Kaikou/venv` 建好推理环境，并下载两个模型（约 8 GB）。

第一次启动时，要在「系统设置 → 隐私与安全性」里给 Kaikou 打开两项权限：

- **麦克风**：用来听你说话。
- **辅助功能**：用来监听 ⌃⇧，以及把文字粘贴到其他 App。

**公司管理的 Mac 上开辅助功能需要管理员**。这种情况下不用申请：只要从一个已经有辅助功能和麦克风权限的终端（比如 Orca）里启动 Kaikou，它就沿用那个终端的权限。`scripts/kaikou-ensure` 做的就是这件事：如果 Kaikou 没在运行，或者在运行但没有权限，就从当前终端重新启动它。安装脚本会把它链接到 `~/.local/bin/kaikou-ensure`，在 `~/.zshrc` 里加一行，让每次打开 Orca 终端时自动在后台运行一下：

```bash
[[ "$TERM_PROGRAM" == "Orca" && -x "$HOME/.local/bin/kaikou-ensure" ]] && ( "$HOME/.local/bin/kaikou-ensure" >/dev/null 2>&1 & )
```

完全没有辅助功能权限时也能用：按住 ⌃⌥Space 说话，松开后文字会放进剪贴板，自己按 ⌘V 粘贴。权限一打开，App 会自动启用 ⌃⇧ 和自动粘贴，不用重启。

每次重新编译安装后，macOS 会把它当成新的 App，需要在设置里把这两项权限重新打开一次。

## 怎么做到又准又快

- **识别**用 Qwen3-ASR 1.7B（8 位量化），把你最常用的约 220 个专有名词作为提示词交给它。
- **整理**用本地的 Qwen3-8B，提示词里带完整词汇表和你自己的写作风格示例。
- **边说边识别、边说边整理**：你一停顿，前面那段就在后台识别并整理好，松手后只剩最后一小段要处理，所以说 10 秒还是说 2 分钟，等待时间都差不多（中位数约 1.2 秒）。
- **提示词缓存和抄写加速**：固定不变的提示词只计算一次；整理结果大多是照抄原话，所以一次猜一串字，让模型一次性验证。
- **保险**：整理结果如果加了原话没有的内容（比如开始回答你的问题）、丢了整句话、或者加了“请”“您”这类客套话，就改用只删口头禅、只修错字的保守模式再整理一次；还不行就输出原始识别结果。

模型选型和测试数据见 [docs/DESIGN.md](docs/DESIGN.md)。

## 个人数据

词汇表、示例、历史录音都放在 `~/Library/Application Support/Kaikou/`，不进这个仓库。

- `lexicon.json`：词汇表和纠错规则，改完下一次说话时自动生效。
- `config.json`（可选）：覆盖默认设置，比如换模型。
- `history/`：每次说话的录音、原始识别和最终文字，方便以后调优。
- `maintenance.json`：每天的自我整理报告。引擎每天第一次空闲时会删掉 30 天前的录音（文字记录保留）、清理只出现过一次且 60 天没再出现的纠错、以及互相冲突的纠错（比如同一个词被改成两种写法时，保留改得更多的那个），日志超过 5 MB 时只留最后 1 MB。结果显示在面板“概览”的分析总结里。

## 开发

```bash
swift test --package-path app                                   # App 单元测试
~/Library/Application\ Support/Kaikou/venv/bin/python -m unittest discover engine/tests   # 引擎测试
./scripts/build_app.sh                                          # 只编译 App
```

引擎日志在 `~/Library/Logs/Kaikou/engine.log`，当前权限和引擎状态在 `~/Library/Application Support/Kaikou/status.json`。

不需要任何权限的自测：

```bash
~/Applications/Kaikou.app/Contents/MacOS/Kaikou --mic-test 3                        # 麦克风录 3 秒
~/Applications/Kaikou.app/Contents/MacOS/Kaikou --transcribe --realtime a.wav b.wav  # 按真实语速走完整链路
```

`./scripts/install.sh` 默认只更新引擎（放在 App 包外面），不会改动 App 的签名，所以权限不会丢；加 `--app` 才会重装 App。
