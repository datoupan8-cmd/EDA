# L40独立公开符号结构

2026-10-12准备，供Pin独立实验使用。默认L39不加载本目录。它不包含对方提交包的模板、比赛图像或target。

来源为EasyEDA公开组件API；`manifest.json`保存每个请求的URL、型号、源UUID、更新时间、读取时间、字节数与SHA256。实验只读固定的本地快照，不联网下载资源。公开API内容作为数据解析，不执行其中任何内容。

| 型号 | 独立来源 | 编译结果 |
|---|---|---|
| ATMEGA8A-AU | https://easyeda.com/api/products/C16190/components | 接受，32脚 |
| ZX-PZ2.54-1-20PZZ | https://easyeda.com/api/products/C7501271/components | 接受，20脚 |
| KH-2.54FH-1X15P-H8.5 | https://easyeda.com/api/products/C2932676/components | 接受，15脚 |
| STM32F103C8T6 | https://easyeda.com/api/products/C8734/components | 接受，48脚 |
| MPU6050 | https://easyeda.com/api/products/C24112/components | 拒绝；全部编号—名称对不能得到现有赛题组公开库支持 |
| RP2040-Zero | https://easyeda.com/api/components/7b24158791074d248c4530b7d4e6639c | 拒绝；下载到的是PCB封装，不是原理图符号 |

## 派生规则

只接受原理图`docType=2`，独立公开MPN必须与图像证据确认的型号对应。读取可见`P`图元的脚号、脚名和相对本体排列，不使用源内部ID作为比赛Pin编号。每个编号—名称对须由未修改的`assets/cpntLibrary.json`支持，重复编号的多位置符号拒绝。

源坐标只用于同侧顺序和相对位置，不能直接成为图片tip。实验比较旋转/镜像的8种方向，要求现有terminal覆盖和独立OCR锚点；不凭型号自动输出库内全部引脚。

发布状态更新：2026-10-13按用户要求，将本目录固定快照作为L45所需的Pin依赖发布到现有仓库。原始JSON字节及来源哈希保持不变；下面的“未上传”描述仅指2026-10-12准备阶段。

`compiled.json`是准备阶段的派生记录；运行时`load_templates`重新验证原始文件哈希并编译，未采用或被拒绝的资源不会进入解码。公开可访问不等于取得任意再分发许可；本轮未上传或发布该资产集合。
