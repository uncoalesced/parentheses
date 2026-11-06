# Engineered by uncoalesced
from pathlib import Path

p = Path(r"E:\Parentheses\parentheses-0.9\scripts\eval_dravidian.py")
content = p.read_text(encoding="utf-8")

old_line = 'ids += [tok.token_to_id(f"<{lang}>")] + body'
new_block = '''lang_id = tok.token_to_id(f"<{lang}>")
            if lang_id is not None:
                ids += [lang_id] + body
                score += [False] + [True] * len(body)
            else:
                ids += body
                score += [True] * len(body)'''

# Note: The original code has:
# ids += [tok.token_to_id(f"<{lang}>")] + body
# score += [False] + [True] * len(body)

target = 'ids += [tok.token_to_id(f"<{lang}>")] + body\n            score += [False] + [True] * len(body)'
if target not in content:
    target = 'ids += [tok.token_to_id(f"<{lang}>")] + body\r\n            score += [False] + [True] * len(body)'

assert target in content, "Target block not found"
p.write_text(content.replace(target, new_block), encoding="utf-8")
print("[+] Successfully patched eval_dravidian.py")
