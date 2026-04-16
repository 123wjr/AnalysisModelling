import numpy as np
import pandas as pd
import random

# 设置随机种子，保证结果可复现
np.random.seed(42)
random.seed(42)


def _sample_likert_from_latent(latent, noise_sd=0.75):
    """
    将连续潜变量映射到1-5李克特分值。
    使用分位点切分而非硬编码阈值，避免分布过于僵硬。
    """
    z = latent + np.random.normal(0, noise_sd, size=len(latent))
    q = np.quantile(z, [0.2, 0.4, 0.6, 0.8])
    return np.digitize(z, q) + 1


def _sample_multiselect_with_state(base_probs, pressure, resilience, capability):
    """
    多选题按个体状态动态生成：压力高更倾向求助，韧性高更偏向自我调节。
    """
    n = len(pressure)
    cols = []
    pressure_z = (pressure - np.mean(pressure)) / (np.std(pressure) + 1e-8)
    resilience_z = (resilience - np.mean(resilience)) / (np.std(resilience) + 1e-8)
    capability_z = (capability - np.mean(capability)) / (np.std(capability) + 1e-8)

    for idx, p0 in enumerate(base_probs):
        # 每个选项共享压力主趋势，同时加上小幅差异项。
        delta = 0.15 * pressure_z
        if idx in (0, 2, 3, 5):
            delta += 0.06 * (1 - resilience_z)
        if idx in (1, 4):
            delta += 0.08 * resilience_z + 0.05 * capability_z

        p = np.clip(p0 + delta, 0.03, 0.95)
        cols.append(np.random.binomial(1, p, size=n))
    return cols


def generate_simulated_data(n=300):
    """
    生成符合数学性描述的问卷模拟数据（默认300条）

    description:
    在保留4个核心结论方向与序关系的基础上，加入更贴近现实的隐含维度：
    1. 年级梯度：压力随年级线性上升。
    2. 专业差异：新专业 > 传统专业 > 中外合作专业。
    3. 实习效应：无实习 > 本地实习 > 外地实习，且为强主效应。
    4. 负向因素：职业规划清晰度、毕业去向确定性越高，压力越低。

    新增现实机制：
    - 潜变量：就业能力(capability)、家庭经济压力(financial_stress)、心理韧性(resilience)。
    - 条件依赖：年级和能力影响实习概率；能力影响规划清晰度；规划与能力影响去向确定性。
    - 异方差：高压力个体波动更大。
    - 行为联动：多选题概率由压力和韧性等状态共同驱动，而非固定常数。

    输出列结构与原问卷保持一致，便于复用现有分析脚本。

    严格遵循给定的4个统计结论：
    1. 年级梯度：压力随年级线性上升
    2. 专业差异：新专业 > 传统专业 > 中外合作专业
    3. 实习效应：无实习 > 本地实习 > 外地实习，效应量最大
    4. 负向因素：职业规划清晰度、毕业去向确定性负向影响压力，联合解释度最高
    """
    # ---------------------- 1. 生成基本人口统计变量 ----------------------
    # 性别：1=男，2=女，比例符合示例数据的51:49
    gender = np.random.choice([1, 2], size=n, p=[0.51, 0.49])

    # 年级：1=大一，2=大二，3=大三，4=大四，分布符合示例数据
    grade = np.random.choice([1, 2, 3, 4], size=n, p=[0.1, 0.12, 0.22, 0.56])

    # 专业：2=传统工科，3=经管，4=数理统计，5=中外合作，6=新专业
    # 覆盖示例数据的专业范围，同时满足专业差异的序关系
    major = np.random.choice([2, 3, 4, 5, 6], size=n, p=[0.10, 0.20, 0.15, 0.25, 0.30])

    # ---------- 新增潜变量：更符合现实的数据结构 ----------
    # 就业能力：受年级（经验积累）与专业训练影响
    major_capability_bonus = {2: 0.05, 3: -0.05, 4: 0.10, 5: 0.00, 6: 0.15}
    capability = (
        0.22 * (grade - 2.5)
        + np.array([major_capability_bonus[m] for m in major])
        + np.random.normal(0, 0.75, size=n)
    )

    # 家庭经济压力：男女性别差异非常弱，仅作细微扰动
    financial_stress = (
        np.random.normal(0, 1.0, size=n)
        + 0.05 * (gender == 1).astype(float)
        - 0.03 * (gender == 2).astype(float)
    )

    # 心理韧性：与能力正相关，与经济压力轻微负相关
    resilience = (
        0.35 * capability - 0.18 * financial_stress + np.random.normal(0, 0.7, size=n)
    )

    # 实习经历：1=无，2=本地，3=外地，使用条件概率模拟现实路径
    # 年级越高、能力越强，越可能有实习；能力高者更可能外地实习。
    ability_z = (capability - np.mean(capability)) / (np.std(capability) + 1e-8)
    grade_z = (grade - np.mean(grade)) / (np.std(grade) + 1e-8)
    score_none = (
        1.2 - 1.1 * grade_z - 0.9 * ability_z + np.random.normal(0, 0.25, size=n)
    )
    score_local = (
        0.8 + 0.4 * grade_z + 0.3 * ability_z + np.random.normal(0, 0.25, size=n)
    )
    score_remote = (
        -0.2 + 0.6 * grade_z + 0.8 * ability_z + np.random.normal(0, 0.25, size=n)
    )
    logits = np.column_stack([score_none, score_local, score_remote])
    logits = logits - logits.max(axis=1, keepdims=True)
    probs = np.exp(logits)
    probs = probs / probs.sum(axis=1, keepdims=True)
    rand = np.random.rand(n)
    intern = np.where(
        rand < probs[:, 0],
        1,
        np.where(rand < probs[:, 0] + probs[:, 1], 2, 3),
    )

    # 职业规划清晰度：能力与韧性提升清晰度，高经济压力略降低清晰度
    clarity_latent = 0.55 * ability_z + 0.22 * resilience - 0.18 * financial_stress
    career_clarity = _sample_likert_from_latent(clarity_latent, noise_sd=0.8)

    # 毕业去向：1=直接就业，2=考研，3=保研，4=出国，5=考公，6=其他
    destination = np.random.choice(
        [1, 2, 3, 4, 5, 6], size=n, p=[0.20, 0.25, 0.10, 0.10, 0.25, 0.10]
    )

    # 计算毕业去向确定性：用于压力模型，确定性排序：保研>考公>出国>考研>直接就业>其他
    destination_certainty = np.zeros(n)
    for i in range(n):
        d = destination[i]
        if d == 3:  # 保研，确定性最高
            destination_certainty[i] = 5
        elif d == 5:  # 考公
            destination_certainty[i] = 4
        elif d == 4:  # 出国
            destination_certainty[i] = 3
        elif d == 2:  # 考研
            destination_certainty[i] = 2
        elif d == 1:  # 直接就业
            destination_certainty[i] = 1
        else:  # 其他，确定性最低
            destination_certainty[i] = 0

    # 让去向确定性在“类别基线”上受能力与规划清晰度微调，更接近现实
    destination_certainty = destination_certainty + np.clip(
        0.35 * (career_clarity - 3)
        + 0.45 * ability_z
        + np.random.normal(0, 0.35, size=n),
        -1.0,
        1.0,
    )
    destination_certainty = np.clip(destination_certainty, 0, 5)

    # ---------------------- 2. 生成整体就业压力（核心数学模型） ----------------------
    # 说明：问卷原始量表是1-5分，因此在该量表内拟合 goal.md 的方向与效应强弱。
    base_pressure = 3.0

    # 结论1：年级线性上升（大一到大四逐级增加）
    # 这里适度加大斜率，确保年级主效应能够覆盖“高年级更容易实习、能力更强”的反向相关项。
    grade_effect = 0.58 * (grade - 1)

    # 结论3：实习经历的决定性主效应（无实习 > 本地实习 > 外地实习）
    intern_effect_map = {1: 0.7, 2: 0.0, 3: -0.3}
    intern_effect = np.array([intern_effect_map[i] for i in intern], dtype=float)

    # 结论2：专业序关系（新专业 > 传统工科 > 中外合作）
    major_effect = np.zeros(n, dtype=float)
    for i in range(n):
        m = major[i]
        if m == 6:  # 新专业，压力最高
            major_effect[i] = 0.35
        elif m == 2:  # 传统工科
            major_effect[i] = 0.05
        elif m == 5:  # 中外合作，压力最低
            major_effect[i] = -0.25
        else:  # 其他专业
            major_effect[i] = 0.0

    # 结论4：毕业去向确定性与职业规划清晰度为负向核心因素
    # 采用中心化写法，避免整体均值被过度压低到1分附近。
    clarity_effect = -0.18 * (career_clarity - 3)
    destination_effect = -0.2 * (destination_certainty - 2.5)

    # 新增隐含维度效应：经济压力抬升压力，韧性与能力降低压力
    financial_effect = 0.18 * financial_stress
    resilience_effect = -0.16 * resilience
    capability_effect = -0.12 * ability_z

    # 随机噪声：异方差，高压力基线个体波动更大
    rough_baseline = (
        base_pressure
        + grade_effect
        + intern_effect
        + major_effect
        + clarity_effect
        + destination_effect
        + financial_effect
        + resilience_effect
        + capability_effect
    )
    noise_sd = np.clip(0.35 + 0.08 * (rough_baseline - 3), 0.28, 0.75)
    noise = np.random.normal(0, noise_sd, size=n)

    # 合并计算压力值
    pressure = (
        base_pressure
        + grade_effect
        + intern_effect
        + major_effect
        + clarity_effect
        + destination_effect
        + financial_effect
        + resilience_effect
        + capability_effect
        + noise
    )

    # 压力水平校准层：在不改变个体相对差异方向的前提下，
    # 将整体均值与离散度拉回更合理区间，避免“整体普遍过高”。
    target_pressure_mean = 3.65
    dispersion_scale = 0.90
    pressure = target_pressure_mean + dispersion_scale * (pressure - np.mean(pressure))

    # 截断到1-5的问卷取值范围，四舍五入为整数选项
    pressure = np.clip(pressure, 1, 5)
    pressure = np.round(pressure).astype(int)

    # ---------------------- 3. 生成压力来源打分题 ----------------------
    # 12个压力来源题共享较强公共因子，以提升量表内部一致性（信度）。
    # 同时保留四个维度的微小差异，避免题目完全同质化。
    pressure_z = (pressure - np.mean(pressure)) / (np.std(pressure) + 1e-8)
    dim_latent = np.random.normal(0, 0.22, size=(n, 4))
    item_dim_map = [0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3]

    source_columns = []
    for item_idx in range(12):
        dim_idx = item_dim_map[item_idx]
        item_bias = np.random.normal(0, 0.06)
        item_noise = np.random.normal(0, 0.17, size=n)
        latent_item = (
            0.92 * pressure_z
            + 0.16 * dim_latent[:, dim_idx]
            + 0.08 * ability_z
            + item_bias
            + item_noise
        )
        s = 3.3 + 0.72 * latent_item
        s = np.clip(s, 1, 5)
        s = np.round(s).astype(int)
        source_columns.append(s)

    # ---------------------- 4. 生成多选题：压力缓解方式 ----------------------
    # 6个选项，选中概率来自示例数据的统计均值
    relief_probs = [0.49, 0.41, 0.41, 0.11, 0.41, 0.11]
    relief_columns = _sample_multiselect_with_state(
        relief_probs, pressure, resilience, capability
    )

    # ---------------------- 5. 生成多选题：就业信息渠道 ----------------------
    # 7个选项，选中概率来自示例数据
    channel_probs = [0.3425, 0.41, 0.595, 0.315, 0.27, 0.4375, 0.0925]
    channel_columns = []
    pressure_z = (pressure - pressure.mean()) / (pressure.std() + 1e-8)
    for idx, p in enumerate(channel_probs):
        # 高压力更依赖线上渠道和同伴信息，能力高者更偏向官方/专业渠道。
        p_dynamic = p + 0.08 * pressure_z
        if idx in (0, 2, 5):
            p_dynamic += 0.06 * ability_z
        if idx in (3, 6):
            p_dynamic += 0.05 * (1 - resilience)
        p_dynamic = np.clip(p_dynamic, 0.03, 0.95)
        col = np.random.binomial(1, p_dynamic, size=n)
        channel_columns.append(col)

    # ---------------------- 6. 学校就业服务满意度 ----------------------
    # 1-5分，均值和标准差符合示例数据
    service_satisfy = (
        2.7
        + 0.12 * (career_clarity - 3)
        - 0.18 * (pressure - pressure.mean())
        + 0.08 * ability_z
        + np.random.normal(0, 0.9, size=n)
    )
    service_satisfy = np.clip(service_satisfy, 1, 5)
    service_satisfy = np.round(service_satisfy).astype(int)

    # ---------------------- 7. 多选题：就业服务需求 ----------------------
    # 8个选项，选中概率来自示例数据
    service_need_probs = [0.4325, 0.3825, 0.455, 0.33, 0.3, 0.2625, 0.205, 0.125]
    service_need_columns = []
    for idx, p in enumerate(service_need_probs):
        # 压力高、满意度低者更倾向选择更多服务需求。
        p_dynamic = p + 0.10 * pressure_z - 0.08 * (service_satisfy - 3)
        if idx in (0, 1, 2, 4):
            p_dynamic += 0.05 * (1 - ability_z)
        p_dynamic = np.clip(p_dynamic, 0.03, 0.95)
        col = np.random.binomial(1, p_dynamic, size=n)
        service_need_columns.append(col)

    # ---------------------- 8. 开放性问题 ----------------------
    # 大部分为"无"，少量为空
    open_question = np.array(["无"] * n)
    # 高压力且低韧性个体更可能留下建议文本。
    feedback_prob = np.clip(
        0.03 + 0.12 * pressure_z + 0.06 * (1 - resilience), 0.01, 0.35
    )
    has_feedback = np.random.binomial(1, feedback_prob, size=n).astype(bool)
    random_idx = np.where(~has_feedback)[0]
    keep_empty = np.random.choice(random_idx, min(10, len(random_idx)), replace=False)
    open_question[keep_empty] = ""
    open_question[has_feedback] = np.random.choice(
        [
            "希望增加校企对接岗位信息",
            "建议开展更多一对一职业咨询",
            "希望简历和面试实战训练更频繁",
            "建议提供跨地区实习资源",
        ],
        size=np.sum(has_feedback),
    )

    # ---------------------- 整理数据，与示例Excel格式完全对齐 ----------------------
    columns = [
        "1.您的性别",
        "2.您的年级",
        "3.专业",
        "4.您是否有过实习经历:",
        "5.您的职业规划清晰度:",
        "6.您的毕业去向意向:",
        "7.您目前感受到的整体就业压力程度是:（1=完全没有压力，5=压力极大）",
        "8.(1)同校同专业毕业生就业竞争激烈",
        "9.(2)本地适合本专业的就业岗位较少",
        "10.(3)企业招聘门槛越来越高(学历/技能)",
        "11.(1)缺乏简历制作与面试技巧",
        "12.(2)没有明确的职业发展方向",
        "13.(3)缺乏相关的专业实习经历",
        "14.(1)担心找不到自己较满意的工作",
        "15.(2)对未来职业发展前景感到迷茫",
        "16.(3)担心工作薪资待遇达不到预期",
        "17.(1)父母对自己的就业期望较高",
        "18.(2)身边同学的就业情况带来心理压力",
        "19.(3)毕业后的经济压力（房租/生活成本）",
        "20.当您感受到压力时，主要通过哪些方式缓解?（可多选）【多选题】_1",
        "20.当您感受到压力时，主要通过哪些方式缓解?（可多选）【多选题】_2",
        "20.当您感受到压力时，主要通过哪些方式缓解?（可多选）【多选题】_3",
        "20.当您感受到压力时，主要通过哪些方式缓解?（可多选）【多选题】_4",
        "20.当您感受到压力时，主要通过哪些方式缓解?（可多选）【多选题】_5",
        "20.当您感受到压力时，主要通过哪些方式缓解?（可多选）【多选题】_6",
        "21.您获取就业信息的主要渠道有哪些?（可多选）【多选题】_1",
        "21.您获取就业信息的主要渠道有哪些?（可多选）【多选题】_2",
        "21.您获取就业信息的主要渠道有哪些?（可多选）【多选题】_3",
        "21.您获取就业信息的主要渠道有哪些?（可多选）【多选题】_4",
        "21.您获取就业信息的主要渠道有哪些?（可多选）【多选题】_5",
        "21.您获取就业信息的主要渠道有哪些?（可多选）【多选题】_6",
        "21.您获取就业信息的主要渠道有哪些?（可多选）【多选题】_7",
        "22.您认为学校目前的就业指导服务是否满足您的需求?",
        "23.您最希望学校提供哪些就业服务?（可多选）【多选题】_1",
        "23.您最希望学校提供哪些就业服务?（可多选）【多选题】_2",
        "23.您最希望学校提供哪些就业服务?（可多选）【多选题】_3",
        "23.您最希望学校提供哪些就业服务?（可多选）【多选题】_4",
        "23.您最希望学校提供哪些就业服务?（可多选）【多选题】_5",
        "23.您最希望学校提供哪些就业服务?（可多选）【多选题】_6",
        "23.您最希望学校提供哪些就业服务?（可多选）【多选题】_7",
        "23.您最希望学校提供哪些就业服务?（可多选）【多选题】_8",
        "24.对于缓解大学生就业压力，您还有其他什么建议或想法?",
    ]

    # 合并所有列
    data = np.column_stack(
        [
            gender,
            grade,
            major,
            intern,
            career_clarity,
            destination,
            pressure,
            *source_columns,
            *relief_columns,
            *channel_columns,
            service_satisfy,
            *service_need_columns,
            open_question,
        ]
    )

    # 创建DataFrame
    df = pd.DataFrame(data, columns=columns)

    # 保存为Excel文件，与示例格式一致
    df.to_excel("simulated_questionnaire_data.xlsx", index=False, engine="openpyxl")
    print(
        f"✅ 模拟数据生成完成！共生成{len(df)}条符合数学性描述的问卷数据，已保存为 simulated_questionnaire_data.xlsx"
    )
    print(
        "ℹ️ 说明：本版本已加入潜变量、条件概率与异方差噪声，数据结构更贴近真实调查样本。"
    )
    return df


if __name__ == "__main__":
    generate_simulated_data(300)
