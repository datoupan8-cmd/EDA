# 网络模块拆分：保持现有结果

起点：integration/component-pin，05415bd3a0cf7a4c2f8ee844d1a2c356291745d6。远端整合分支相同；同学组件分支最新为d648403，已在此起点中整合。

本轮仅结构拆分。元件/Text、Pin、Wire算法、公共Pipeline/Schema/Submission、评测器、原默认配置与模型都不修改。拆分在独立feature/network-metrics-split工作树完成；原Pin研究目录保持。

网络处理顺序：冻结基础线图 → 独立NetHypergraph归属 → 独立NetLine线段表达 → 兼容组装到原Scene.nets。基础图的参考线段用于逻辑吸附，和最终导出的线段分开；后者改简化参数不能反向影响逻辑吸附。

- 共享契约/基础图/组装：pcb/network_shared/，仅集成维护。
- 网络归属：pcb/net_hypergraph/；configs/net_hypergraph/；tests/net_hypergraph/。
- 线段表达：pcb/net_lines/；configs/net_lines/；tests/net_lines/。
- 旧入口pcb/topology_stage/v2.py保持函数签名和冻结旧算法；父V2/V3 Stage调用可独立选择的组合模块。v1保留旧实现，总Pipeline和导出器保留。
- 两个选择片段默认legacy_v2，各自注册版本，不改core注册表和team.json。

验证：改前保存合成场景和现有开发35例的旧网络输出；改后逐项比较完整JSON、skeleton与原诊断，证明Component/Pin冻结。所有输入来自此前无GT预测和原图，拒绝target/保留集；不读GT、不跑评分或新150。

有意义的隔离检查包括：更换线段策略不能改变hyperGraph；更换逻辑策略可重分网络但不编造线段；输入视图不可变；未知版本和非法输出拒绝；跨维护目录的改动检查失败。

完成后跑现有工程测试、记录结果和使用说明，仅提交本轮网络拆分文件，上传上述GitHub分支，并在远端整合分支没有额外变更时快进同步integration/component-pin。镜像、包、依赖和模型不处理。本轮不承诺分数增加，评分项的网络匹配依赖仍存在。
