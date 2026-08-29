import ssl
import sys

# ====================================================================
# 🛡️ 绝对优先级的免疫补丁：在所有库加载之前，先修复 Windows 证书 Bug
# ====================================================================
orig_create_default_context = ssl.create_default_context

def safe_create_default_context(*args, **kwargs):
    try:
        return orig_create_default_context(*args, **kwargs)
    except ssl.SSLError:
        return ssl.SSLContext()

ssl.create_default_context = safe_create_default_context
# ====================================================================

# 补丁打完之后，我们再安全地导入 Streamlit
from streamlit.web import cli

if __name__ == '__main__':
    print("🚀 正在通过防爆启动器拉起 DeePFAS-KAN Web Server...")
    # 模拟命令行输入 streamlit run app.py
    sys.argv = ["streamlit", "run", "app.py"]
    sys.exit(cli.main())