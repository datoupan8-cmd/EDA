# NetLine 维护边界

先读 docs/network/AGENT_HANDOFF.md 和 docs/network/README.md。

本目录只负责给定网络计划的线段表达。输入WireGraph/NetworkPlan，输出GeometryResult；保持group_id集合，不改变Pin成员、原始引脚坐标、网络归属或共享路径。

新增实现保留legacy_v2不变，注册只改本目录registry.py，选择只改configs/net_lines/current.json。测试、模型与文档分别放tests/net_lines/、models/net_lines/、docs/network/lines/。

不改NetHypergraph、Wire、network_shared、topology_stage、公共core/schema/Submission、评测器或他人配置。需要契约变化先拆成独立集成请求。

运行范围检查：python tools/check_network_change_scope.py --owner net_lines --base origin/integration/component-pin。再跑完整团队测试与冻结输入A/B。数据仅限0001～0150。
