# app.py
import streamlit as st
import os
import time
from services import LLMService
from models import ContentBlueprint

# ==========================================
# 1. 状态管理与初始化
# ==========================================

def init_session_state():
    # 核心状态：'input' (输入阶段) 或 'workspace' (工作台阶段)
    if "stage" not in st.session_state:
        st.session_state.stage = "input"
    
    # 存储生成的蓝图对象（包含创意简报和大纲）
    if "blueprint" not in st.session_state:
        st.session_state.blueprint = None
    
    # 存储各章节的正文草稿 {section_id: text_content}
    if "drafts" not in st.session_state:
        st.session_state.drafts = {}
    
    # 存储各章节生成的插图 {section_id: image_path}
    if "illustrations" not in st.session_state:
        st.session_state.illustrations = {}
    
    # 用户输入缓存（用于回显）
    if "user_inputs" not in st.session_state:
        st.session_state.user_inputs = {"fragments": "", "style": "", "image_path": None}
    
    # 进度状态标记
    if "is_building" not in st.session_state:
        st.session_state.is_building = False
    
    # 用于控制灵感输入栏是否展开（工作台阶段）
    if "show_input_expander" not in st.session_state:
        st.session_state.show_input_expander = False
    
    # 编辑模式状态 {item_key: is_editing}
    if "editing" not in st.session_state:
        st.session_state.editing = {}
    
    # Store current service instance to check image support
    if "current_service" not in st.session_state:
        st.session_state.current_service = None

init_session_state()

st.set_page_config(layout="wide", page_title="AI 深度写作流", page_icon="✍️")

# ==========================================
# 2. 通用 UI 组件
# ==========================================

def render_config_bar():
    """顶部配置条"""
    with st.container():
        c1, c2, c3 = st.columns([1, 1, 2])
        
        # 1. 服务商选择
        with c1:
            provider = st.selectbox(
                "服务商", 
                ["Google Gemini", "DeepSeek", "Qwen-VL"], 
                key="provider_select"
            )
        
        # 2. 根据服务商分别渲染输入框
        api_key = None
        model_name = ""

        if provider == "Google Gemini":
            with c2:
                # 默认值改为新版 SDK 推荐的 flash 模型
                model_name = st.text_input("模型名称", value="gemini-3-flash-preview", key="model_gemini")
            with c3:
                # 检测环境变量
                env_key = os.getenv("GEMINI_API_KEY")
                if env_key:
                    api_key = st.text_input(
                        "✅ 已通过环境变量配置 Key", 
                        value="", 
                        placeholder="不需要输入 API Key",
                        disabled=True,
                        key="key_gemini"
                    )
                    api_key = env_key  # 使用环境变量中的key
                else:
                    api_key = st.text_input("API Key", type="password", key="key_gemini")

        elif provider == "DeepSeek":
            with c2:
                model_name = st.text_input("模型名称", value="deepseek-chat", key="model_deepseek")
            with c3:
                api_key = st.text_input("API Key", type="password", key="key_deepseek")
        elif provider == "Qwen-VL":
            with c2:
                model_name = st.text_input(
                    "本地模型路径或 HF 仓库 ID",
                    # value="models/Qwen/Qwen3-VL-8B-Instruct",
                    value="models/Qwen/Qwen3-VL-4B-Instruct-FP8",
                    key="Qwen-VL"
                )
            with c3:
                st.text("本地模型：无需 API Key。确保已安装 transformers/accelerate 并有足够资源。")
                api_key = None
        
        return provider, model_name, api_key


# ==========================================
# 3. 灵感输入表单组件（可复用）
# ==========================================

def render_input_form(in_workspace=False, supports_images=False):
    """
    渲染灵感输入表单
    in_workspace: 是否在工作台模式（影响form的key和行为）
    supports_images: 当前模型是否支持图片输入
    """
    form_key = "workspace_input_form" if in_workspace else "initial_input_form"
    
    with st.form(form_key):
        # Image upload section (only if model supports it)
        uploaded_file = None
        if supports_images:
            st.markdown("#### 🖼️ 图片输入 (可选)")
            uploaded_file = st.file_uploader(
                "上传图片以辅助内容生成",
                type=["png", "jpg", "jpeg", "webp"],
                help="支持的格式: PNG, JPG, JPEG, WEBP"
            )
            if uploaded_file:
                # Display preview
                st.image(uploaded_file, caption="已上传图片预览", use_container_width=True)
            st.divider()
        else:
            st.info("💡 提示：当前模型不支持图片输入。切换到 Qwen-VL 模型以启用图片功能。")
            st.divider()
        
        # Text inputs
        fragments = st.text_area(
            "意图碎片 (支持 **粗体** 强调核心观点)", 
            height=200,
            value=st.session_state.user_inputs["fragments"],
            placeholder="例如：写一篇关于**长期主义**的文章..."
        )
        style = st.text_input(
            "风格偏好 / 示例",
            value=st.session_state.user_inputs["style"], 
            placeholder="例如：理性、克制、像《经济学人》..."
        )
        
        submitted = st.form_submit_button("🚀 开始构建", type="primary", use_container_width=True)
        
        return fragments, style, uploaded_file, submitted


# ==========================================
# 4. 构建流程处理
# ==========================================

def handle_build_process(fragments, style, uploaded_file, provider, model, key):
    """处理完整的构建流程：创意简报 -> 大纲 -> 正文"""
    
    # 保存用户输入
    st.session_state.user_inputs["fragments"] = fragments
    st.session_state.user_inputs["style"] = style
    
    # Handle image upload if provided
    image_path = None
    if uploaded_file:
        # Save uploaded file temporarily
        import tempfile
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            image_path = tmp_file.name
        st.session_state.user_inputs["image_path"] = image_path
    
    # 初始化服务
    try:
        service = LLMService(provider, model, key)
        st.session_state.current_service = service
    except Exception as e:
        st.error(f"初始化服务失败: {str(e)}")
        return False
    
    # 显示进度状态
    status_placeholder = st.empty()
    status_placeholder.info("🔄 正在构建……")
    
    # 第1步：生成创意简报
    brief_placeholder = st.empty()
    brief_placeholder.info("📝 正在生成创意简报……")
    
    try:
        blueprint = service.generate_blueprint(fragments, style, image_path=image_path)
    except Exception as e:
        brief_placeholder.error(f"生成创意简报失败: {str(e)}")
        status_placeholder.empty()
        return False
    
    # 创意简报生成成功
    brief_placeholder.success("✅ 创意简报生成完成")
    
    # 保存蓝图并切换到工作台
    st.session_state.blueprint = blueprint
    st.session_state.stage = "workspace"
    st.session_state.drafts = {}  # 清空旧的正文草稿
    
    status_placeholder.empty()
    brief_placeholder.empty()
    
    return True


# ==========================================
# 5. 页面逻辑：阶段一 (初始输入页面)
# ==========================================

def render_input_stage(provider, model, key):
    st.markdown("## 💡 灵感输入")
    
    # Check if model supports images
    supports_images = False
    if provider == "Qwen-VL":
        supports_images = True
    
    fragments, style, uploaded_file, submitted = render_input_form(in_workspace=False, supports_images=supports_images)
    
    if submitted:
        if not fragments:
            st.error("请至少输入一些意图碎片")
            return
        
        # 处理构建流程
        success = handle_build_process(fragments, style, uploaded_file, provider, model, key)
        if success:
            st.rerun()  # 刷新进入工作台


# ==========================================
# 6. 页面逻辑：阶段二 (工作台页面)
# ==========================================

def render_workspace_stage(provider, model, key):
    blueprint = st.session_state.blueprint
    if not blueprint:
        st.error("数据丢失，请返回重新生成")
        if st.button("🔙 返回首页"):
            st.session_state.stage = "input"
            st.rerun()
        return
    
    # Check if model supports images
    supports_images = False
    if st.session_state.current_service:
        supports_images = st.session_state.current_service.supports_images
    
    # --- 可下拉的灵感修改栏 ---
    with st.expander("🔽 点击下拉修改灵感重新生成", expanded=st.session_state.show_input_expander):
        fragments, style, uploaded_file, submitted = render_input_form(in_workspace=True, supports_images=supports_images)
        
        if submitted:
            if not fragments:
                st.error("请至少输入一些意图碎片")
            else:
                # 重新构建（旧内容在新内容生成前保留）
                success = handle_build_process(fragments, style, uploaded_file, provider, model, key)
                if success:
                    st.session_state.show_input_expander = False
                    st.rerun()
    
    st.divider()
    
    # --- 主工作区：三栏布局 ---
    st.markdown("### 📝 写作工作台")
    st.divider()
    
    col_brief_container, col_workspace_container = st.columns([1, 2.7])
    
    # === 左栏：创意简报 ===
    with col_brief_container:
        st.markdown("#### 📋 创意简报")
        brief = blueprint.brief
        
        # Show uploaded image if present
        if st.session_state.user_inputs.get("image_path"):
            with st.expander("🖼️ 查看参考图片", expanded=False):
                st.image(st.session_state.user_inputs["image_path"], use_container_width=True)
        
        # 定位靶心
        render_editable_brief_item("targeting", "🎯 定位靶心", brief.targeting, brief)
        
        # 核心洞察
        render_editable_brief_item("insight", "💡 核心洞察", brief.insight, brief)
        
        # 价值跨越 - 当前状态
        render_editable_brief_item("current_state", "🌈 当前状态", 
                                    brief.transformation.current_state, brief)
        
        # 价值跨越 - 期望状态
        render_editable_brief_item("desired_state", "✨ 期望状态", 
                                    brief.transformation.desired_state, brief)
        
        # 沟通策略 - 钩子类型
        render_editable_brief_item("hook_type", "🎣 钩子类型", 
                                    brief.strategy.hook_type, brief)
        
        # 沟通策略 - 沟通语气
        render_editable_brief_item("tone", "🎭 沟通语气", 
                                    brief.strategy.tone, brief)
        
        st.divider()
        
        # 重新生成所有大纲按钮
        if st.button("🔄 重新生成所有大纲", use_container_width=True, type="secondary"):
            regenerate_all_outlines(provider, model, key)
    
    # === 标题栏 ===
    with col_workspace_container:
        h_col1, h_col2 = st.columns([1.5, 1.2])
        with h_col1:
            st.markdown("#### 📑 大纲")
        with h_col2:
            st.markdown("#### 📄 正文")

        # 找到第一个未生成正文的索引
        first_pending_idx = None
        for idx, section in enumerate(blueprint.outline):
            if section.id not in st.session_state.drafts:
                first_pending_idx = idx
                break
        
        # === 在循环内部创建列 ===
        for idx, section in enumerate(blueprint.outline):
            # 每一行都是独立的 columns，确保高度对齐
            c_outline, c_text = st.columns([1.5, 1.2])
            
            with c_outline:
                render_outline_section(section, idx, provider, model, key)
                
            with c_text:
                should_generate = (first_pending_idx is not None and idx == first_pending_idx)
                render_text_section(section, idx, provider, model, key, should_generate=should_generate)
            
            # 每一行结束后加分割线
            st.divider()

# ==========================================
# 7. 创意简报编辑组件
# ==========================================

def render_editable_brief_item(field_key, title, current_value, brief):
    """渲染可编辑的创意简报项目"""
    edit_key = f"brief_{field_key}"
    
    st.markdown(f"**{title}**")
    
    # 检查是否处于编辑模式
    if st.session_state.editing.get(edit_key, False):
        # 编辑模式
        new_value = st.text_area(
            f"编辑 {title}",
            value=current_value,
            height=100,
            key=f"edit_area_{edit_key}",
            label_visibility="collapsed"
        )
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("✅ 保存", key=f"save_{edit_key}", use_container_width=True):
                # 更新值
                update_brief_field(field_key, new_value, brief)
                st.session_state.editing[edit_key] = False
                st.rerun()
        with col2:
            if st.button("❌ 取消", key=f"cancel_{edit_key}", use_container_width=True):
                st.session_state.editing[edit_key] = False
                st.rerun()
    else:
        # 显示模式
        st.write(current_value)
        if st.button("✏️ 编辑", key=f"edit_btn_{edit_key}", use_container_width=True):
            st.session_state.editing[edit_key] = True
            st.rerun()


def update_brief_field(field_key, new_value, brief):
    """更新创意简报字段"""
    if field_key == "targeting":
        brief.targeting = new_value
    elif field_key == "insight":
        brief.insight = new_value
    elif field_key == "current_state":
        brief.transformation.current_state = new_value
    elif field_key == "desired_state":
        brief.transformation.desired_state = new_value
    elif field_key == "hook_type":
        brief.strategy.hook_type = new_value
    elif field_key == "tone":
        brief.strategy.tone = new_value


# ==========================================
# 8. 大纲编辑和生成组件
# ==========================================

def render_outline_section(section, idx, provider, model, key):
    """渲染单个大纲节"""
    section_key = f"outline_{section.id}"
    edit_key = f"edit_{section_key}"
    
    with st.container(border=True):
        # 检查是否处于编辑模式
        if st.session_state.editing.get(edit_key, False):
            # 编辑模式
            new_title = st.text_input("标题", value=section.title, 
                                      key=f"edit_title_{section_key}")
            new_intent = st.text_input("意图", value=section.intent, 
                                       key=f"edit_intent_{section_key}")
            new_points = st.text_area("关键点（每行一个）", 
                                      value="\n".join(section.key_points),
                                      height=100,
                                      key=f"edit_points_{section_key}")
            
            col1, col2 = st.columns(2)
            with col1:
                if st.button("✅ 保存", key=f"save_{edit_key}", use_container_width=True):
                    # 更新大纲
                    section.title = new_title
                    section.intent = new_intent
                    section.key_points = [p.strip() for p in new_points.split("\n") if p.strip()]
                    st.session_state.editing[edit_key] = False
                    st.rerun()
            with col2:
                if st.button("❌ 取消", key=f"cancel_{edit_key}", use_container_width=True):
                    st.session_state.editing[edit_key] = False
                    st.rerun()
        else:
            # 显示模式
            c_btn, c_content = st.columns([1, 10])
            
            with c_btn:
                # 重新生成按钮 (小三角)
                if st.button("▶", key=f"regen_outline_{section.id}", help="重写本节大纲"):
                     regenerate_single_outline(idx, provider, model, key)
            
            with c_content:
                st.markdown(f"**{section.id}. {section.title}**")
                st.caption(f"💡 {section.intent}")
                st.markdown("**关键点:**")
                for p in section.key_points:
                    st.text(f"• {p}")

            if st.button("✏️ 编辑", key=f"edit_btn_{edit_key}", use_container_width=True):
                st.session_state.editing[edit_key] = True
                st.rerun()


def regenerate_single_outline(idx, provider, model, key):
    """重新生成单节大纲（占位功能）"""
    st.toast(f"重新生成第 {idx+1} 节大纲功能开发中...")


def regenerate_all_outlines(provider, model, key):
    """重新生成所有大纲"""
    try:
        service = LLMService(provider, model, key)
        fragments = st.session_state.user_inputs["fragments"]
        style = st.session_state.user_inputs["style"]
        image_path = st.session_state.user_inputs.get("image_path")
        
        with st.spinner("🔄 正在重新生成大纲..."):
            new_blueprint = service.generate_blueprint(fragments, style, image_path=image_path)
        
        if new_blueprint:
            # 保留创意简报，只更新大纲
            st.session_state.blueprint.outline = new_blueprint.outline
            # 清空正文草稿
            st.session_state.drafts = {}
            st.success("✅ 大纲重新生成完成")
            st.rerun()
    except Exception as e:
        st.error(f"重新生成大纲失败: {str(e)}")


# ==========================================
# 9. 正文编辑和生成组件
# ==========================================

def render_text_section(section, idx, provider, model, key, should_generate=False):
    """渲染单节正文"""
    section_key = section.id
    edit_key = f"edit_text_{section_key}"
    
    with st.container(border=True):
        # 检查是否已有正文
        if section_key not in st.session_state.drafts:
            # 只有轮到这一段才生成，否则显示占位符
            if should_generate:
                # 自动生成正文
                with st.spinner(f"✍️ 正在撰写第 {idx+1} 节..."):
                    try:
                        service = LLMService(provider, model, key)
                        generated_text = service.generate_section_text(
                            section=section,
                            brief=st.session_state.blueprint.brief,
                            section_idx=idx
                        )
                        st.session_state.drafts[section_key] = generated_text
                        st.rerun()
                    except Exception as e:
                        st.error(f"生成第 {idx+1} 节失败: {str(e)}")
                        st.session_state.drafts[section_key] = f"生成失败：{str(e)}"
            else:
                # 显示占位符
                st.info(f"⏳ 等待生成第 {idx+1} 节...")
        else:
            current_text = st.session_state.drafts[section_key]
            
            # 检查是否处于编辑模式
            if st.session_state.editing.get(edit_key, False):
                # 编辑模式
                new_text = st.text_area(
                    "编辑内容",
                    value=current_text,
                    height=300,
                    key=f"edit_area_{edit_key}",
                    label_visibility="collapsed"
                )
                
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("✅ 保存", key=f"save_{edit_key}", use_container_width=True):
                        st.session_state.drafts[section_key] = new_text
                        st.session_state.editing[edit_key] = False
                        st.rerun()
                with col2:
                    if st.button("❌ 取消", key=f"cancel_{edit_key}", use_container_width=True):
                        st.session_state.editing[edit_key] = False
                        st.rerun()
            else:
                # 显示模式
                c_btn, c_content = st.columns([1, 10])
                
                with c_btn:
                     # 重新生成按钮 (小三角)
                    if st.button("▶", key=f"regen_text_{section.id}", help="重写本段正文"):
                        regenerate_single_text(section, idx, provider, model, key)
                
                with c_content:
                    st.markdown(current_text)
                
                # Display illustration if exists
                if section_key in st.session_state.illustrations:
                    image_path = st.session_state.illustrations[section_key]
                    if os.path.exists(image_path):
                        st.image(image_path, caption=f"第 {idx+1} 节配图", use_container_width=True)

                # Action buttons row
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("✏️ 编辑", key=f"edit_btn_{edit_key}", use_container_width=True):
                        st.session_state.editing[edit_key] = True
                        st.rerun()
                with col2:
                    # Image generation button
                    if section_key not in st.session_state.illustrations:
                        if st.button("🎨 生成配图", key=f"gen_img_{section_key}", use_container_width=True):
                            generate_section_illustration(section, idx, provider, model, key)
                    else:
                        if st.button("🔄 重新生成配图", key=f"regen_img_{section_key}", use_container_width=True):
                            generate_section_illustration(section, idx, provider, model, key)


def regenerate_single_text(section, idx, provider, model, key):
    """重新生成单节正文"""
    section_key = section.id
    try:
        service = LLMService(provider, model, key)
        
        with st.spinner(f"✍️ 正在重新撰写第 {idx+1} 节..."):
            generated_text = service.generate_section_text(
                section=section,
                brief=st.session_state.blueprint.brief,
                section_idx=idx
            )
            st.session_state.drafts[section_key] = generated_text
            st.success(f"✅ 第 {idx+1} 节重新生成完成")
            st.rerun()
    except Exception as e:
        st.error(f"重新生成第 {idx+1} 节失败: {str(e)}")


def generate_section_illustration(section, idx, provider, model, key):
    """生成单节配图"""
    section_key = section.id
    
    # Check if text exists
    if section_key not in st.session_state.drafts:
        st.warning("请先生成正文内容再生成配图")
        return
    
    try:
        service = LLMService(provider, model, key)
        
        # Check if SD is available
        if not hasattr(service, 'sd_model_path') or not service.sd_model_path:
            st.error("❌ Stable Diffusion 未加载。请设置 SD_MODEL 环境变量并重启应用。")
            return
        
        with st.spinner(f"🎨 正在为第 {idx+1} 节生成配图..."):
            section_text = st.session_state.drafts[section_key]
            image_path = service.generate_illustration(
                section=section,
                brief=st.session_state.blueprint.brief,
                section_text=section_text
            )
            
            if image_path:
                st.session_state.illustrations[section_key] = image_path
                st.success(f"✅ 第 {idx+1} 节配图生成完成")
                st.rerun()
            else:
                st.error("配图生成失败")
    except Exception as e:
        st.error(f"生成配图失败: {str(e)}")


# ==========================================
# 10. 主程序入口
# ==========================================

def main():
    # 1. 顶部始终显示配置
    provider, model, key = render_config_bar()
    
    # 2. 根据状态分发视图
    if st.session_state.stage == "input":
        render_input_stage(provider, model, key)
    else:
        render_workspace_stage(provider, model, key)

if __name__ == "__main__":
    main()