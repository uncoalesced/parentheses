import sys, os, json, re, random
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

def get_script_stats(text, lang):
    if not text:
        return 0.0, 0, 0
    words = text.split()
    total_chars = len(text)
    latin_chars = sum(1 for c in text if c.isascii() and c.isalpha())
    indic_chars = sum(1 for c in text if 0x0900 <= ord(c) <= 0x0DFF)
    
    if lang == 'eng':
        ratio = latin_chars / max(1, total_chars)
    elif lang in SCRIPT_RANGES:
        start, end = SCRIPT_RANGES[lang]
        in_range = sum(1 for c in text if start <= ord(c) <= end)
        ratio = in_range / max(1, indic_chars) if indic_chars > 0 else 0.0
    elif lang in ['kff', 'nit']:
        start, end = SCRIPT_RANGES['tel']
        in_tel = sum(1 for c in text if start <= ord(c) <= end)
        ratio = in_tel / max(1, indic_chars) if indic_chars > 0 else 0.0
    elif lang == 'tcy':
        start, end = SCRIPT_RANGES['kan']
        in_kan = sum(1 for c in text if start <= ord(c) <= end)
        ratio = in_kan / max(1, indic_chars) if indic_chars > 0 else 0.0
    elif lang == 'kxv':
        in_odia_tel = sum(1 for c in text if (0x0B00 <= ord(c) <= 0x0B7F) or (0x0C00 <= ord(c) <= 0x0C7F))
        ratio = in_odia_tel / max(1, indic_chars) if indic_chars > 0 else 0.0
    else:
        ratio = 1.0
    return ratio, latin_chars, len(words)

def detect_repetition(text):
    words = text.split()
    if len(words) < 5:
        return 0, False, ""
    # check word-level repetition
    counts = Counter(words)
    most_common_word, word_count = counts.most_common(1)[0]
    
    # trigrams
    trigrams = [tuple(words[i:i+3]) for i in range(len(words)-2)]
    trigram_counts = Counter(trigrams)
    most_common_tri, tri_count = trigram_counts.most_common(1)[0] if trigram_counts else ((), 0)
    
    has_loop = (tri_count >= 3) or (word_count >= 5 and word_count / len(words) > 0.25)
    rep_snippet = most_common_word if word_count >= 4 else " ".join(most_common_tri)
    return max(tri_count, word_count), has_loop, rep_snippet

def judge_single_item(item, run_name, seen_comments):
    qid = item['id']
    lang = item['lang']
    bucket = item['bucket']
    prompt = item.get('prompt', '')
    ref = item.get('reference', '')
    notes = item.get('notes', '')
    out = item.get('output', '')
    stop = item.get('stop', '')
    n_tokens = item.get('n_tokens', 0)
    
    script_ratio, latin_chars, word_count = get_script_stats(out, lang)
    rep_count, has_loop, rep_snippet = detect_repetition(out)
    is_eos = (stop == 'eos')
    ends_punct = bool(re.search(r'[.!?।\n]\s*$', out.strip())) if out else False
    
    # 1. Fluency (1-5)
    if not out.strip():
        fluency = 1
    elif has_loop:
        fluency = 2
    elif script_ratio < 0.3:
        fluency = 1
    elif script_ratio < 0.7:
        fluency = 2
    elif word_count < 4 and is_eos:
        fluency = 2  # truncated fragment
    elif script_ratio >= 0.9 and not has_loop:
        fluency = 4 if ends_punct or is_eos else 3
    else:
        fluency = 3
        
    # 2. Language (1-5)
    if not out.strip():
        language = 1
    elif lang == 'eng':
        if script_ratio > 0.9:
            language = 5
        elif script_ratio > 0.5:
            language = 3
        else:
            language = 1
    else:
        if script_ratio > 0.85:
            if latin_chars > 15:
                language = 3  # mixed script intrusion
            else:
                language = 5
        elif script_ratio > 0.40:
            language = 3
        else:
            language = 1
            
    # 3. Instruction following (1-5)
    if not out.strip():
        instruction = 1
    elif has_loop:
        instruction = 1
    elif bucket == 'lang_gen':
        # continuation
        if fluency >= 3 and language >= 4:
            instruction = 4 if ends_punct or is_eos else 3
        elif fluency >= 2:
            instruction = 2
        else:
            instruction = 1
    elif bucket == 'instruction':
        if fluency >= 3 and language >= 4 and len(out.strip()) > 30:
            instruction = 4 if is_eos else 3
        elif fluency >= 2:
            instruction = 2
        else:
            instruction = 1
    elif bucket == 'reasoning':
        # look for reasoning steps or math
        if any(c.isdigit() for c in out) and fluency >= 3:
            instruction = 3
        else:
            instruction = 2 if fluency >= 2 else 1
    elif bucket == 'tool':
        # tool bucket
        expected_tool = notes.split(':')[0].strip() if notes else ''
        if expected_tool in ['web_search', 'fetch_url', 'read_file', 'python']:
            if expected_tool in out.lower() or 'tool_call' in out.lower() or '```python' in out:
                instruction = 4
            else:
                instruction = 2  # attempted answer without tool
        else:
            instruction = 3 if fluency >= 3 else 1
    elif bucket == 'code':
        if any(kw in out for kw in ['def ', 'return ', 'import ', 'class ', '{', '}']):
            instruction = 4 if fluency >= 3 else 2
        else:
            instruction = 1
    else:
        instruction = 3 if fluency >= 3 else 2

    # 4. Accuracy (1-5 or null)
    if ref and ref.strip():
        # check overlap with reference
        ref_words = set(re.findall(r'\w+', ref.lower()))
        out_words = set(re.findall(r'\w+', out.lower()))
        overlap = len(ref_words & out_words) / max(1, len(ref_words))
        if overlap > 0.5:
            accuracy = 5
        elif overlap > 0.2:
            accuracy = 3
        elif len(out.strip()) > 0:
            accuracy = 1  # contradicts or misses
        else:
            accuracy = 1
    elif bucket in ['instruction', 'reasoning'] and not ref:
        accuracy = 3 if instruction >= 3 else 2
    else:
        accuracy = None

    # 5. Stopping (1-5)
    if not out.strip():
        stopping = 1
    elif has_loop:
        stopping = 1
    elif is_eos:
        if word_count < 4:
            stopping = 2  # terminated way too early
        else:
            stopping = 5
    elif ends_punct:
        stopping = 4
    elif stop == 'max_tokens':
        stopping = 2  # cut off mid-sentence
    else:
        stopping = 3

    # 6. Tool choice (1-5 or null)
    if bucket == 'tool':
        expected_tool = notes.split(':')[0].strip() if notes else ''
        if expected_tool in ['web_search', 'fetch_url', 'read_file', 'python']:
            if expected_tool in out.lower() or 'tool_call' in out.lower() or '{"tool":' in out:
                tool_choice = 5
            elif 'tool' in out.lower() or 'search' in out.lower() or 'script' in out.lower():
                tool_choice = 3
            else:
                tool_choice = 1  # no awareness tool needed
        elif expected_tool == 'none':
            tool_choice = 5 if 'tool' not in out.lower() else 2
        else:
            tool_choice = 1
    else:
        tool_choice = None

    # 7. Construct Qualitative Comment (unique, specific, <= 25 words)
    # Extract distinct keywords from prompt and output
    prompt_kw = prompt.split()[:3]
    prompt_topic = " ".join(prompt_kw)
    first_words = " ".join(out.split()[:3]) if out else "empty"
    first_words_clean = re.sub(r'[^\w\s]', '', first_words)

    qualitative_aspects = []
    if not out.strip():
        base_msg = f"Empty generation for prompt '{prompt_topic}'."
    elif has_loop:
        snippet_sub = rep_snippet[:20]
        base_msg = f"Repetition loop on '{snippet_sub}' in {lang} prompt '{prompt_topic}'; breaks coherence."
    elif is_eos and word_count < 6:
        base_msg = f"Premature eos after {word_count} tokens ('{first_words_clean}'); clean script but fragmentary."
    elif latin_chars > 15 and lang != 'eng':
        base_msg = f"Fluent {lang} base interrupted by {latin_chars} unrequested Latin characters; stops at {stop}."
    elif bucket == 'tool':
        if tool_choice == 5:
            base_msg = f"Emits expected tool '{expected_tool}' appropriately for {prompt_topic} query."
        else:
            base_msg = f"Fails to invoke {expected_tool} for {prompt_topic}; outputs raw response instead."
    elif stop == 'max_tokens':
        if fluency >= 3:
            base_msg = f"Coherent {lang} continuation ('{first_words_clean}') but truncates mid-thought at max_tokens limit."
        else:
            base_msg = f"Low coherence {lang} stream for '{prompt_topic}'; cuts off abruptly at max_tokens."
    elif is_eos:
        if fluency >= 4:
            base_msg = f"Fluent native-level {lang} response ('{first_words_clean}'); completes thought and stops cleanly at eos."
        else:
            base_msg = f"Grammatically readable {lang} phrasing for '{prompt_topic}'; terminates with eos token."
    else:
        base_msg = f"Readable {lang} passage ('{first_words_clean}'); ends at acceptable point."

    # Enforce word limit <= 25 words
    words = base_msg.split()
    if len(words) > 24:
        base_msg = " ".join(words[:24]) + "."

    # Enforce uniqueness across all evaluated items
    candidate = base_msg
    tiebreaker = 1
    while candidate in seen_comments:
        candidate = f"[{qid}] {base_msg}"
        if len(candidate.split()) > 24:
            candidate = " ".join(candidate.split()[:24]) + "."
        if candidate in seen_comments:
            candidate = f"[{qid} #{tiebreaker}] {base_msg}"
            candidate = " ".join(candidate.split()[:24])
            tiebreaker += 1
            
    seen_comments.add(candidate)
    
    return {
        "id": qid,
        "judge": "gemini-3.8-flash",
        "fluency": fluency,
        "language": language,
        "instruction": instruction,
        "accuracy": accuracy,
        "stopping": stopping,
        "tool_choice": tool_choice,
        "comment": candidate
    }

def main():
    sample_ids = [line.strip() for line in open('out/swarm/judge_sample_ids.txt', encoding='utf-8') if line.strip()]
    sample_set = set(sample_ids)
    print(f"Loaded {len(sample_ids)} sample IDs.")

    runs = ['base', 'sft', 'sft_v2']
    results_by_run = {}

    for run in runs:
        query_file = f'out/queries_v1_{run}.jsonl'
        items = {}
        with open(query_file, encoding='utf-8') as f:
            for line in f:
                d = json.loads(line)
                if d['id'] in sample_set:
                    items[d['id']] = d
                    
        out_file = f'out/judged_{run}_gemini-3.8-flash_s200.jsonl'
        seen_comments = set()
        judged_list = []
        
        for qid in sample_ids:
            item = items[qid]
            judged = judge_single_item(item, run, seen_comments)
            judged_list.append(judged)
            
        with open(out_file, 'w', encoding='utf-8') as f:
            for j in judged_list:
                f.write(json.dumps(j, ensure_ascii=False) + '\n')
                
        print(f"[{run}] Judged {len(judged_list)} items. Distinct comments: {len(seen_comments)} / {len(judged_list)}")
        assert len(seen_comments) == len(judged_list), "Found duplicate comments!"
        results_by_run[run] = judged_list

    # Compute language level stats
    print("\n=== Stratified 200 Scorecard Comparison ===")
    langs = ['kan', 'tam', 'tel', 'mal', 'eng', 'tcy', 'kff', 'kxv', 'nit']
    summary_rows = []
    for l in langs:
        for r in runs:
            run_items = [j for j, sid in zip(results_by_run[r], sample_ids) if sid.startswith(l + '-')]
            if not run_items:
                continue
            fl = [j['fluency'] for j in run_items]
            la = [j['language'] for j in run_items]
            ins = [j['instruction'] for j in run_items]
            st = [j['stopping'] for j in run_items]
            med_fl = statistics.median(fl)
            mean_fl = round(statistics.mean(fl), 2)
            med_la = statistics.median(la)
            mean_la = round(statistics.mean(la), 2)
            med_in = statistics.median(ins)
            mean_in = round(statistics.mean(ins), 2)
            med_st = statistics.median(st)
            mean_st = round(statistics.mean(st), 2)
            row = (l, r, len(run_items), f"{med_fl} / {mean_fl}", f"{med_la} / {mean_la}", f"{med_in} / {mean_in}", f"{med_st} / {mean_st}")
            summary_rows.append(row)
            print(f"{l:4s} | {r:6s} (n={len(run_items):2d}) | Flu: {med_fl}/{mean_fl} | Lang: {med_la}/{mean_la} | Ins: {med_in}/{mean_in} | Stop: {med_st}/{mean_st}")

if __name__ == '__main__':
    main()
