import sys
import os

site_packages = [p for p in sys.path if "site-packages" in p]
print("Site packages:", site_packages)
found = []
for sp in site_packages:
    for root, dirs, files in os.walk(sp):
        for f in files:
            if f.endswith(".py"):
                p = os.path.join(root, f)
                try:
                    with open(p, "r", encoding="utf-8", errors="ignore") as fh:
                        content = fh.read()
                        if "divinedevops" in content or "vector_inspector" in content:
                            found.append((p, "divinedevops" in content, "vector_inspector" in content))
                except Exception:
                    pass
print("Found in site-packages:", found)

# Also search entire C:\Users\ADMIN\Downloads\Llm_gateway
workspace = r"C:\Users\ADMIN\Downloads\Llm_gateway"
ws_found = []
for root, dirs, files in os.walk(workspace):
    if ".venv" in root or ".git" in root or ".cache" in root:
        continue
    for f in files:
        if f.endswith((".py", ".env", ".toml", ".yaml", ".yml", ".json", ".ini")):
            p = os.path.join(root, f)
            try:
                with open(p, "r", encoding="utf-8", errors="ignore") as fh:
                    content = fh.read()
                    if "divinedevops" in content or "vector_inspector" in content:
                        ws_found.append((p, "divinedevops" in content, "vector_inspector" in content))
            except Exception:
                pass
print("Found in workspace:", ws_found)
