# NetHypergraph 维护边界

先读 docs/network/AGENT_HANDOFF.md 和 docs/network/README.md。

本目录负责已有引脚与导线路径的网络归属。输入仅为只读SceneView/WireGraph，输出LogicalResult，所有成员引用已有Pin ref和路径ID；不要修改上游Pin/Component。

新增实现保留legacy_v2不变，注册只改本目录registry.py，选择只改configs/net_hypergraph/current.json。测试、模型与文档分别放tests/net_hypergraph/、models/net_hypergraph/、docs/network/hypergraph/。

不改NetLine、Wire、network_shared、topology_stage、公共core/schema/Submission、评测器或他人配置。需要契约变化先拆成独立集成请求。

运行范围检查：python tools/check_network_change_scope.py --owner net_hypergraph --base origin/integration/component-pin。再跑完整团队测试与冻结输入A/B。数据仅限0001～0150。
