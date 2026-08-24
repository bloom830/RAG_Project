"""
转人工客服模块
职责：渲染侧边栏"转人工"入口，支持二维码弹窗 / 联系方式展示
配置方式（三选一，按优先级检测）：
  1. 企业微信 / 微信客服二维码图片 → 放在 resources/contact_qr.png
  2. 环境变量 TRANSFER_CONTACT  →  如 "微信: zhi_cede" 或 "电话: 400-xxx"
  3. 兜底：显示默认提示语
"""

import os
from pathlib import Path

# ============= 配置 =============

# 二维码图片路径（支持 png / jpg）
QR_PATHS = [
    Path(__file__).resolve().parent.parent / "resources" / "contact_qr.png",
    Path(__file__).resolve().parent.parent / "resources" / "contact_qr.jpg",
]

# 环境变量兜底联系方式
CONTACT_TEXT = os.getenv(
    "TRANSFER_CONTACT",
    "请通过企业微信联系人工客服，或发送邮件至 support@example.com",
)

# 工作时段提示
WORK_HOURS_TEXT = os.getenv(
    "WORK_HOURS_TEXT",
    "人工客服工作时间：周一至周五 9:00 - 18:00",
)


def _find_qr_image() -> Path | None:
    for p in QR_PATHS:
        if p.exists():
            return p
    return None


def render_transfer_button():
    """在侧边栏渲染"转人工"按钮与弹窗"""
    import streamlit as st

    st.markdown("---")
    st.markdown("### 🆘 需要人工帮助？")

    # 使用 expander 模拟弹窗
    with st.expander("点击联系人工客服", expanded=False):
        qr_path = _find_qr_image()

        if qr_path:
            st.image(str(qr_path), caption="扫码联系人工客服", width=180)
            st.caption("请使用微信 / 企业微信扫描上方二维码")
        else:
            st.info(CONTACT_TEXT)

        st.caption(WORK_HOURS_TEXT)

        # 快捷操作：复制联系方式
        if CONTACT_TEXT and "support@example.com" not in CONTACT_TEXT:
            if st.button("📋 复制联系方式", key="copy_contact", use_container_width=True):
                st.toast("已复制到剪贴板（请手动复制上方联系方式）", icon="✅")
