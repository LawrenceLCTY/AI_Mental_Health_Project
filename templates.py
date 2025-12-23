# templates.py

# 认知偏转模型 v2
TEMPLATE_CCOGNITIVE_FLIP = {
    "id": "cognitive_reframing_v2",
    "name": "深度认知重构 (Scenario-Based Reframing)",
    "description": "结构：误区 - 真相 - 机制 - 行动，结合具体场景。作用：通过专业概念解决具体情景中的痛点。适合：科普、观点类内容。",
    "structure": [
        {
            "section_id": "myth",
            "role": "直接与读者对话，将真实、具体、能引发共鸣的场景娓娓道来，使读者有被人说中痛点的感受",
            "source_mapping": "Targeting + Current State",
            "content_instruction": "1. 开篇：作为全文开篇段落，自然地将观众引入情景（例如：你是否也曾经历过……）\n 2. 场景描述：可以选取一个具体场景深入细致描绘，也可以简要罗列多个场景，但也须具体真实有细节；场景的选择需切中 [Targeting] 中描述的人群的痛点。\n 3. 基于 [Current State], 以一句话总结这种状态的困境。",
            "word_count_limit": 150
        },
        {
            "section_id": "shift",
            "role": "以犀利、明确、极简、有力的语气，指出认知误区，并引出专业理论作为解决之道",
            "source_mapping": "Insight",
            "content_instruction": "1. 承接上一段（例如：但实际上……）\n 2. 基于 [Insight] 直击前述痛点。结合 [Hook Type] 决定写法（例如：直接给出解决方案：“其实你应该……”，或一语道破原因：“这其实是因为……”，等等）。",
            "word_count_limit": 100
        },
        {
            "section_id": "mechanism",
            "role": "详细讲解专业理论的原理机制，并将其与前述具体场景紧密结合",
            "source_mapping": "Insight + Myth",
            "content_instruction": "1. 专业背书：首先以科学、学术的语言准确阐述 [Insight] 的原理机制，使用恰当的专业术语体现学术背景、增强说服力。\n2. 场景回扣：用通俗的语言，使用刚才讲的专业理论解释第一节 myth 中所描述的情景。必须使用第一节 myth 中描述的第一个具体情景。可以使用通俗的比喻增进理解。",
            "word_count_limit": 1000
        },
        {
            "section_id": "action",
            "role": "在上文讲解的基础上，提出几个适用于前述具体场景的简便行动建议",
            "source_mapping": "Desired State + Myth",
            "content_instruction": "1. 方案总结：基于 [Insight] 用一句话概括解决痛点的方案，可以较为 high-level（例如：因此，其实你需要的是……）. \n2. 对症下药：针对第一节 myth 中描述的具体场景，给出几个行动建议，以自然、流畅的语言描述。（例如：也许下次再……的时候，你可以……）\n3. 愿景收尾：简要描绘执行这些行动后的 [Desired State]。语言简单自然，避免给人压迫、说教感，避免使人厌烦。",
            "word_count_limit": 200
        }
    ]
}

# 认知翻转模型 v1
TEMPLATE_COGNITIVE_FLIP = {
    "id": "cognitive_flip",
    "name": "认知翻转模型 (The Cognitive Flip)",
    "description": "适合通过推翻一个大众常识，引入一个新的专业概念。适合科普、观点类内容。",
    "structure": [
        {
            "section_id": "hook",
            "role": "SCENE_SETTING",
            "title_pattern": "直击【Targeting】的某个具体错误行为",
            "content_instruction": "描述【Current State】中的具体场景。指出大众普遍认为正确的做法（误区），并制造‘但是’的悬念。",
            "word_count_limit": 100
        },
        {
            "section_id": "conflict",
            "role": "MYTH_BUSTING",
            "title_pattern": "为什么【Old Method】反而害了你？",
            "content_instruction": "分析为什么传统做法无效。从原理上否定旧认知。放大痛点。",
            "word_count_limit": 150
        },
        {
            "section_id": "solution",
            "role": "CONCEPT_INTRODUCTION",
            "title_pattern": "引入【Insight】的核心概念",
            "content_instruction": "正式提出【Insight】，先用专业语言进行最简洁的叙述，然后用通俗的比喻解释它。强调它是解决问题的唯一‘新钥匙’。",
            "word_count_limit": 200
        },
        {
            "section_id": "action",
            "role": "GUIDE_TO_CHANGE",
            "title_pattern": "从现在开始，做这件小事",
            "content_instruction": "基于新概念，给出 1-2 个极低门槛的行动建议。描绘【Desired State】的美好图景。",
            "word_count_limit": 150
        }
    ]
}