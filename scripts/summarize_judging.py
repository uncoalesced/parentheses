import sys, os, json
from collections import defaultdict
import statistics

sys.stdout.reconfigure(encoding='utf-8')

def analyze():
    # Load inputs
    with open('out/queries_v1_base.jsonl', 'r', encoding='utf-8') as f:
        base_inputs = [json.loads(line) for line in f]
    with open('out/queries_v1_sft.jsonl', 'r', encoding='utf-8') as f:
        sft_inputs = [json.loads(line) for line in f]
        
    has_v2 = os.path.exists('out/queries_v1_sft_v2.jsonl') and os.path.exists('out/judged_sft_v2_gemini-3.8-flash.jsonl')
    if has_v2:
        with open('out/queries_v1_sft_v2.jsonl', 'r', encoding='utf-8') as f:
            v2_inputs = [json.loads(line) for line in f]
        with open('out/judged_sft_v2_gemini-3.8-flash.jsonl', 'r', encoding='utf-8') as f:
            v2_judged = [json.loads(line) for line in f]
            
    # Load judged
    with open('out/judged_base_gemini-3.8-flash.jsonl', 'r', encoding='utf-8') as f:
        base_judged = [json.loads(line) for line in f]
    with open('out/judged_sft_gemini-3.8-flash.jsonl', 'r', encoding='utf-8') as f:
        sft_judged = [json.loads(line) for line in f]
        
    score_keys = ['fluency', 'language', 'instruction', 'accuracy', 'stopping', 'tool_choice']
    
    runs_to_check = [(base_inputs, base_judged), (sft_inputs, sft_judged)]
    if has_v2:
        runs_to_check.append((v2_inputs, v2_judged))
        
    for inp, judged in runs_to_check:
        assert len(judged) == 612
        inp_ids = [x['id'] for x in inp]
        judged_ids = [x['id'] for x in judged]
        assert inp_ids == judged_ids, "ID mismatch!"
        for idx, (item, j_item) in enumerate(zip(inp, judged)):
            for k in score_keys:
                assert k in j_item, f"Missing key {k} in item {idx}"
            if item.get('bucket') != 'tool':
                assert j_item['tool_choice'] is None, f"tool_choice not null for non-tool {item['id']}"
            else:
                assert j_item['tool_choice'] is not None, f"tool_choice null for tool {item['id']}"
            words = j_item.get('comment', '').split()
            assert len(words) <= 25, f"Comment too long ({len(words)} words) for {item['id']}"
            
    print("All validation checks PASSED.")
    
    def calc_stats(inputs, judged):
        stats_by_lang = defaultdict(lambda: defaultdict(list))
        for inp, j in zip(inputs, judged):
            lang = inp['lang']
            for k in ['fluency', 'language', 'instruction', 'stopping']:
                val = j[k]
                if val is not None:
                    stats_by_lang[lang][k].append(val)
        return stats_by_lang

    base_stats = calc_stats(base_inputs, base_judged)
    sft_stats = calc_stats(sft_inputs, sft_judged)
    if has_v2:
        v2_stats = calc_stats(v2_inputs, v2_judged)
    
    langs = ['kan', 'tam', 'tel', 'mal', 'eng', 'tcy', 'kff', 'kxv', 'nit', 'hin']
    
    md_lines = []
    md_lines.append("# Phase 3 Query Evaluation: Gemini 3.8 Flash Judge Report")
    md_lines.append("")
    md_lines.append("**Judge**: `gemini-3.8-flash` | **Protocol**: `eval/JUDGE_RUBRIC.md`")
    runs_str = "Base vs SFT-v1 vs SFT-v2" if has_v2 else "Base vs SFT-v1"
    md_lines.append(f"**Evaluated**: {runs_str} (612 items per run)")
    md_lines.append("")
    md_lines.append("## 1. Score Summary by Language & Run")
    md_lines.append("")
    md_lines.append("| Language | Run | Fluency (med / mean) | Language (med / mean) | Instruction (med / mean) | Stopping (med / mean) |")
    md_lines.append("| :--- | :--- | :--- | :--- | :--- | :--- |")
    
    for lang in langs:
        b_s = base_stats[lang]
        s_s = sft_stats[lang]
        
        b_flu_med, b_flu_mean = statistics.median(b_s['fluency']), statistics.mean(b_s['fluency'])
        b_lan_med, b_lan_mean = statistics.median(b_s['language']), statistics.mean(b_s['language'])
        b_ins_med, b_ins_mean = statistics.median(b_s['instruction']), statistics.mean(b_s['instruction'])
        b_stp_med, b_stp_mean = statistics.median(b_s['stopping']), statistics.mean(b_s['stopping'])
        
        s_flu_med, s_flu_mean = statistics.median(s_s['fluency']), statistics.mean(s_s['fluency'])
        s_lan_med, s_lan_mean = statistics.median(s_s['language']), statistics.mean(s_s['language'])
        s_ins_med, s_ins_mean = statistics.median(s_s['instruction']), statistics.mean(s_s['instruction'])
        s_stp_med, s_stp_mean = statistics.median(s_s['stopping']), statistics.mean(s_s['stopping'])
        
        md_lines.append(f"| **{lang}** | Base (v4) | {b_flu_med:.1f} / {b_flu_mean:.2f} | {b_lan_med:.1f} / {b_lan_mean:.2f} | {b_ins_med:.1f} / {b_ins_mean:.2f} | {b_stp_med:.1f} / {b_stp_mean:.2f} |")
        md_lines.append(f"| | SFT-v1 | {s_flu_med:.1f} / {s_flu_mean:.2f} | {s_lan_med:.1f} / {s_lan_mean:.2f} | {s_ins_med:.1f} / {s_ins_mean:.2f} | {s_stp_med:.1f} / {s_stp_mean:.2f} |")
        
        if has_v2:
            v_s = v2_stats[lang]
            v_flu_med, v_flu_mean = statistics.median(v_s['fluency']), statistics.mean(v_s['fluency'])
            v_lan_med, v_lan_mean = statistics.median(v_s['language']), statistics.mean(v_s['language'])
            v_ins_med, v_ins_mean = statistics.median(v_s['instruction']), statistics.mean(v_s['instruction'])
            v_stp_med, v_stp_mean = statistics.median(v_s['stopping']), statistics.mean(v_s['stopping'])
            md_lines.append(f"| | **SFT-v2** | {v_flu_med:.1f} / {v_flu_mean:.2f} | {v_lan_med:.1f} / {v_lan_mean:.2f} | {v_ins_med:.1f} / {v_ins_mean:.2f} | {v_stp_med:.1f} / {v_stp_mean:.2f} |")
        
    md_lines.append("")
    md_lines.append("## 2. Phase 3 Bar: Median Fluency ≥ 3 on Dravidian Core")
    md_lines.append("")
    header = "| Language | Base 41M-v4 Median Fluency | SFT-v1 Median Fluency |"
    divider = "| :--- | :--- | :--- |"
    if has_v2:
        header += " SFT-v2 Median Fluency | Bar Status |"
        divider += " :--- | :--- |"
    else:
        header += " Bar Status (≥ 3.0) |"
        divider += " :--- |"
    md_lines.append(header)
    md_lines.append(divider)
    
    core_langs = ['kan', 'tam', 'tel', 'mal']
    for lang in core_langs:
        b_med = statistics.median(base_stats[lang]['fluency'])
        s_med = statistics.median(sft_stats[lang]['fluency'])
        if has_v2:
            v_med = statistics.median(v2_stats[lang]['fluency'])
            status = "PASS" if v_med >= 3.0 else "FAIL"
            md_lines.append(f"| **{lang}** | {b_med:.1f} | {s_med:.1f} | **{v_med:.1f}** | **{status}** |")
        else:
            status = "PASS" if s_med >= 3.0 and b_med >= 3.0 else ("SFT PASS, BASE FAIL" if s_med >= 3.0 else "FAIL")
            md_lines.append(f"| **{lang}** | {b_med:.1f} | {s_med:.1f} | **{status}** |")

    report_content = "\n".join(md_lines)
    with open('out/swarm/judging_phase3.md', 'w', encoding='utf-8') as rf:
        rf.write(report_content)
        
    print("Report written to out/swarm/judging_phase3.md")

if __name__ == '__main__':
    analyze()
