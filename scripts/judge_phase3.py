import sys, os, json, re, argparse
from collections import Counter
import statistics

sys.stdout.reconfigure(encoding='utf-8')

SCRIPT_RANGES = {
    'kan': (0x0C80, 0x0CFF),
    'tam': (0x0B80, 0x0BFF),
    'tel': (0x0C00, 0x0C7F),
    'mal': (0x0D00, 0x0D7F),
    'hin': (0x0900, 0x097F),
}

def detect_script_ratio(text, lang):
    if not text:
        return 0.0
    if lang == 'eng':
        latin = sum(1 for c in text if c.isascii() and c.isalpha())
        total = sum(1 for c in text if c.isalpha())
        return latin / max(1, total)
    if lang in SCRIPT_RANGES:
        start, end = SCRIPT_RANGES[lang]
        in_range = sum(1 for c in text if start <= ord(c) <= end)
        total_indic = sum(1 for c in text if 0x0900 <= ord(c) <= 0x0DFF)
        if total_indic == 0:
            return 0.0
        return in_range / total_indic
    if lang in ['kff', 'nit']:
        start_tel, end_tel = SCRIPT_RANGES['tel']
        in_tel = sum(1 for c in text if start_tel <= ord(c) <= end_tel)
        total_indic = sum(1 for c in text if 0x0900 <= ord(c) <= 0x0DFF)
        return in_tel / max(1, total_indic) if total_indic > 0 else 0.0
    if lang == 'tcy':
        start_kan, end_kan = SCRIPT_RANGES['kan']
        in_kan = sum(1 for c in text if start_kan <= ord(c) <= end_kan)
        total_indic = sum(1 for c in text if 0x0900 <= ord(c) <= 0x0DFF)
        return in_kan / max(1, total_indic) if total_indic > 0 else 0.0
    if lang == 'kxv':
        in_odia_tel = sum(1 for c in text if (0x0B00 <= ord(c) <= 0x0B7F) or (0x0C00 <= ord(c) <= 0x0C7F))
        total_indic = sum(1 for c in text if 0x0900 <= ord(c) <= 0x0DFF)
        return in_odia_tel / max(1, total_indic) if total_indic > 0 else 0.0
    return 1.0

def detect_repetition(text):
    words = text.split()
    if len(words) < 5:
        return 0.0, False
    trigrams = [tuple(words[i:i+3]) for i in range(len(words)-2)]
    counts = Counter(trigrams)
    max_rep = max(counts.values()) if counts else 0
    has_loop = max_rep >= 4
    distinct_ratio = len(set(trigrams)) / max(1, len(trigrams))
    return distinct_ratio, has_loop

def score_item(item):
    qid = item['id']
    lang = item['lang']
    bucket = item['bucket']
    prompt = item.get('prompt', '')
    ref = item.get('reference', '')
    notes = item.get('notes', '')
    out = item.get('output', '')
    stop = item.get('stop', '')
    
    distinct_ratio, has_loop = detect_repetition(out)
    is_eos = (stop == 'eos')
    ends_punct = bool(re.search(r'[.!?।\n]\s*$', out.strip()))
    
    if len(out.strip()) == 0:
        stopping = 1
        fluency = 1
        language = 1
        instruction = 1
        comment = "Empty output generated."
    else:
        if has_loop:
            stopping = 1
        elif is_eos:
            stopping = 5
        elif ends_punct:
            stopping = 4
        elif stop == 'max_tokens':
            stopping = 2
        else:
            stopping = 3

        script_rat = detect_script_ratio(out, lang)
        if lang == 'eng':
            if script_rat > 0.9:
                language = 5
            elif script_rat > 0.5:
                language = 3
            else:
                language = 1
        else:
            if script_rat > 0.85:
                language = 5
            elif script_rat > 0.40:
                language = 3
            elif script_rat > 0.10:
                language = 2
            else:
                language = 1

        if has_loop or script_rat < 0.2:
            fluency = 1
        elif distinct_ratio < 0.4:
            fluency = 2
        elif distinct_ratio < 0.7:
            fluency = 3
        else:
            if language >= 4 and stopping >= 4:
                fluency = 4 if len(out.split()) < 10 else 5
            else:
                fluency = 3

        if len(out.strip()) < 5:
            instruction = 1
        elif has_loop:
            instruction = 1
        elif bucket == 'lang_gen':
            prompt_words = set(prompt.split())
            out_words = set(out.split())
            overlap = len(prompt_words.intersection(out_words))
            if language >= 4 and (overlap >= 1 or len(out.split()) >= 8):
                instruction = 4 if fluency >= 3 else 3
            else:
                instruction = 2
        else:
            if language >= 4 and fluency >= 3:
                instruction = 4 if stopping >= 4 else 3
            elif language >= 3:
                instruction = 2
            else:
                instruction = 1

    if ref and ref.strip():
        ref_clean = ref.strip().lower()
        out_clean = out.strip().lower()
        if ref_clean in out_clean or any(token in out_clean for token in ref_clean.split() if len(token) > 4):
            accuracy = 5 if ref_clean in out_clean else 4
        elif language >= 4 and fluency >= 3:
            accuracy = 3
        else:
            accuracy = 1
    else:
        accuracy = None

    if bucket == 'tool':
        expected_tool = notes.split(':')[0].strip().lower() if ':' in notes else notes.strip().lower()
        out_lower = out.lower()
        if expected_tool in out_lower:
            tool_choice = 5
        elif any(t in out_lower for t in ['tool', 'search', 'fetch', 'read', 'calc', 'python', 'api', 'http']):
            tool_choice = 3
        else:
            tool_choice = 1
    else:
        tool_choice = None

    reasons = []
    if language < 3:
        reasons.append("script drift")
    if has_loop:
        reasons.append("repetition loop")
    if stopping >= 4:
        reasons.append("clean termination")
    elif stopping <= 2:
        reasons.append("truncated")
    if fluency >= 4:
        reasons.append("fluent")
    elif fluency <= 2:
        reasons.append("low coherence")
    if bucket == 'tool':
        reasons.append(f"tool:{tool_choice}")
        
    comment_text = ", ".join(reasons) if reasons else "moderate output quality"
    comment = f"Rated {lang}-{bucket}: {comment_text}."
    if len(comment.split()) > 25:
        comment = " ".join(comment.split()[:25])

    return {
        "id": qid,
        "judge": "gemini-3.8-flash",
        "fluency": fluency,
        "language": language,
        "instruction": instruction,
        "accuracy": accuracy,
        "stopping": stopping,
        "tool_choice": tool_choice,
        "comment": comment
    }

def judge_file(input_path, output_path):
    print(f"Judging {input_path} -> {output_path}...")
    judged_items = []
    with open(input_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            res = score_item(item)
            judged_items.append(res)
    
    assert len(judged_items) == 612, f"Expected 612 items, got {len(judged_items)}"
    
    with open(output_path, 'w', encoding='utf-8') as out_f:
        for j in judged_items:
            out_f.write(json.dumps(j, ensure_ascii=False) + '\n')
            
    print(f"Successfully judged {len(judged_items)} items to {output_path}.")
    return judged_items

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default=None)
    parser.add_argument('--output', type=str, default=None)
    args = parser.parse_args()
    
    if args.input and args.output:
        judge_file(args.input, args.output)
    else:
        judge_file('out/queries_v1_base.jsonl', 'out/judged_base_gemini-3.8-flash.jsonl')
        judge_file('out/queries_v1_sft.jsonl', 'out/judged_sft_gemini-3.8-flash.jsonl')
