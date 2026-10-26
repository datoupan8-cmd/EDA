# NetLine 准确性评估数据请求

项目文档将最新版官方公开开发集限定为 `200_train_cases/0001–0150`。不得使用 `0151–0200`、Golden、10GT 或 QuickTest。

此前检查的候选路径 `D:\EDA_Data\200_train_cases`、`D:\pcb_data\200_train_cases` 及文档中的旧机器路径在当前机器上均不存在。

为比较 `legacy_v2`、`path_v1` 和 `path_collinear_v1` 的准确性，请负责人提供官方开发集根路径，并确认图片与 `*_target*.json` 的编号映射及标注版本。数据确认后，再核验目录元数据并进行三种后端的同输入公平对照评估。

无需提供受限测试集、Golden、10GT 或 QuickTest。
