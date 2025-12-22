# templates.py

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