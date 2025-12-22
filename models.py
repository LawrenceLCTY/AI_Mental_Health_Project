# models.py
from pydantic import BaseModel, Field
from typing import List, Optional

class OutlineSection(BaseModel):
    """大纲的单个章节"""
    id: int = Field(description="章节序号，从1开始")
    title: str = Field(description="章节标题")
    intent: str = Field(description="本节的设计意图或写作指导")
    key_points: List[str] = Field(description="本节的关键点列表")
    # 注意：draft_content 不让 LLM 直接生成，而是后续步骤生成，这里仅作占位
    draft_content: Optional[str] = None 

class Transformation(BaseModel):
    """读者的状态转变"""
    current_state: str = Field(description="读者现在的糟糕状态")
    desired_state: str = Field(description="读者读完后获得的理想状态")

class Strategy(BaseModel):
    """沟通策略"""
    hook_type: str = Field(description="吸引点击的钩子类型")
    tone: str = Field(description="沟通的语气和人设")

class CreativeBrief(BaseModel):
    """深度创意简报"""
    targeting: str = Field(description="定位靶心：具体的目标受众及其痛点")
    insight: str = Field(description="核心洞察：唯一的解决方案或观点")
    transformation: Transformation = Field(description="价值跨越：读者状态的转变")
    strategy: Strategy = Field(description="沟通策略：钩子和语气")

class ContentBlueprint(BaseModel):
    """完整的生成蓝图"""
    brief: CreativeBrief
    outline: List[OutlineSection]