"""
Download OpenAssistant OASST1, OASST2 and Anthropic hh-rlhf datasets into
data/raw/manual/raw-sourced/ with full provenance, verbatim licenses, sha256 hashes,
prominent DO_NOT_TRAIN for hh-rlhf, and rebuild ATTRIBUTION.md.
"""

import hashlib
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from huggingface_hub import hf_hub_download, HfApi

REPO_ROOT = Path("E:/Parentheses/parentheses-0.9")
RAW_SOURCED = REPO_ROOT / "data/raw/manual/raw-sourced"

APACHE_2_0_TEXT = """                                 Apache License
                           Version 2.0, January 2004
                        http://www.apache.org/licenses/

   TERMS AND CONDITIONS FOR USE, REPRODUCTION, AND DISTRIBUTION

   1. Definitions.

      "License" shall mean the terms and conditions for use, reproduction,
      and distribution as defined by Sections 1 through 9 of this document.

      "Licensor" shall mean the copyright owner or entity authorized by
      the copyright owner that is granting the License.

      "Legal Entity" shall mean the union of the acting entity and all
      other entities that control, are controlled by, or are under common
      control with that entity. For the purposes of this definition,
      "control" means (i) the power, direct or indirect, to cause the
      direction or management of such entity, whether by contract or
      otherwise, or (ii) ownership of fifty percent (50%) or more of the
      outstanding shares, or (iii) beneficial ownership of such entity.

      "You" (or "Your") shall mean an individual or Legal Entity
      exercising permissions granted by this License.

      "Source" form shall mean the preferred form for making modifications,
      including but not limited to software source code, documentation
      source, and configuration files.

      "Object" form shall mean any form resulting from mechanical
      transformation or translation of a Source form, including but
      not limited to compiled object code, generated documentation,
      and conversions to other media types.

      "Work" shall mean the work of authorship, whether in Source or
      Object form, made available under the License, as indicated by a
      copyright notice that is included in or attached to the work
      (an example is provided in the Appendix below).

      "Derivative Works" shall mean any work, whether in Source or Object
      form, that is based on (or derived from) the Work and for which the
      editorial revisions, annotations, elaborations, or other modifications
      represent, as a whole, an original work of authorship. For the purposes
      of this License, Derivative Works shall not include works that remain
      separable from, or merely link (or bind by name) to the interfaces of,
      the Work and Derivative Works thereof.

      "Contribution" shall mean any work of authorship, including
      the original version of the Work and any modifications or additions
      to that Work or Derivative Works thereof, that is intentionally
      submitted to Licensor for inclusion in the Work by the copyright owner
      or by an individual or Legal Entity authorized to submit on behalf of
      the copyright owner. For the purposes of this definition, "submitted"
      means any form of electronic, verbal, or written communication sent
      to the Licensor or its representatives, including but not limited to
      communication on electronic mailing lists, source code control systems,
      and issue tracking systems that are managed by, or on behalf of, the
      Licensor for the purpose of discussing and improving the Work, but
      excluding communication that is conspicuously marked or otherwise
      designated in writing by the copyright owner as "Not a Contribution."

      "Contributor" shall mean Licensor and any individual or Legal Entity
      on behalf of whom a Contribution has been received by Licensor and
      subsequently incorporated within the Work.

   2. Grant of Copyright License. Subject to the terms and conditions of
      this License, each Contributor hereby grants to You a perpetual,
      worldwide, non-exclusive, no-charge, royalty-free, irrevocable
      copyright license to reproduce, prepare Derivative Works of,
      publicly display, publicly perform, sublicense, and distribute the
      Work and such Derivative Works in Source or Object form.

   3. Grant of Patent License. Subject to the terms and conditions of
      this License, each Contributor hereby grants to You a perpetual,
      worldwide, non-exclusive, no-charge, royalty-free, irrevocable
      (except as stated in this section) patent license to make, have made,
      use, offer to sell, sell, import, and otherwise transfer the Work,
      where such license applies only to those patent claims licensable
      by such Contributor that are necessarily infringed by their
      Contribution(s) alone or by combination of their Contribution(s)
      with the Work to which such Contribution(s) was submitted. If You
      institute patent litigation against any entity (including a
      cross-claim or counterclaim in a lawsuit) alleging that the Work
      or a Contribution incorporated within the Work constitutes direct
      or contributory patent infringement, then any patent licenses
      granted to You under this License for that Work shall terminate
      as of the date such litigation is filed.

   4. Redistribution. You may reproduce and distribute copies of the
      Work or Derivative Works thereof in any medium, with or without
      modifications, and in Source or Object form, provided that You
      meet the following conditions:

      (a) You must give any other recipients of the Work or
          Derivative Works a copy of this License; and

      (b) You must cause any modified files to carry prominent notices
          stating that You changed the files; and

      (c) You must retain, in the Source form of any Derivative Works
          that You distribute, all copyright, patent, trademark, and
          attribution notices from the Source form of the Work,
          excluding those notices that do not pertain to any part of
          the Derivative Works; and

      (d) If the Work includes a "NOTICE" text file as part of its
          distribution, then any Derivative Works that You distribute must
          include a readable copy of the attribution notices contained
          within such NOTICE file, excluding those notices that do not
          pertain to any part of the Derivative Works, in at least one
          of the following places: within a NOTICE text file distributed
          as part of the Derivative Works; within the Source form or
          documentation, if provided along with the Derivative Works; or,
          within a display generated by the Derivative Works, if and
          wherever such third-party notices normally appear. The contents
          of the NOTICE file are for informational purposes only and
          do not modify the License. You may add Your own attribution
          notices within Derivative Works that You distribute, alongside
          or as an addendum to the NOTICE text from the Work, provided
          that such additional attribution notices cannot be construed
          as modifying the License.

      You may add Your own copyright statement to Your modifications and
      may provide additional or different license terms and conditions
      for use, reproduction, or distribution of Your modifications, or
      for any such Derivative Works as a whole, provided Your use,
      reproduction, and distribution of the Work otherwise complies with
      the conditions stated in this License.

   5. Submission of Contributions. Unless You explicitly state otherwise,
      any Contribution intentionally submitted for inclusion in the Work
      by You to the Licensor shall be under the terms and conditions of
      this License, without any additional terms or conditions.
      Notwithstanding the above, nothing herein shall supersede or modify
      the terms of any separate license agreement you may have executed
      with Licensor regarding such Contributions.

   6. Trademarks. This License does not grant permission to use the trade
      names, trademarks, service marks, or product names of the Licensor,
      except as required for reasonable and customary use in describing the
      origin of the Work and reproducing the content of the NOTICE file.

   7. Disclaimer of Warranty. Unless required by applicable law or
      agreed to in writing, Licensor provides the Work (and each
      Contributor provides its Contributions) on an "AS IS" BASIS,
      WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
      implied, including, without limitation, any warranties or conditions
      of TITLE, NON-INFRINGEMENT, MERCHANTABILITY, or FITNESS FOR A
      PARTICULAR PURPOSE. You are solely responsible for determining the
      appropriateness of using or redistributing the Work and assume any
      risks associated with Your exercise of permissions under this License.

   8. Limitation of Liability. In no event and under no legal theory,
      whether in tort (including negligence), contract, or otherwise,
      unless required by applicable law (such as deliberate and grossly
      negligent acts) or agreed to in writing, shall any Contributor be
      liable to You for damages, including any direct, indirect, special,
      incidental, or consequential damages of any character arising as a
      result of this License or out of the use or inability to use the
      Work (including but not limited to damages for loss of goodwill,
      work stoppage, computer failure or malfunction, or any and all
      other commercial damages or losses), even if such Contributor
      has been advised of the possibility of such damages.

   9. Accepting Warranty or Additional Liability. While redistributing
      the Work or Derivative Works thereof, You may choose to offer,
      and charge a fee for, acceptance of support, warranty, indemnity,
      or other liability obligations and/or rights consistent with this
      License. However, in accepting such obligations, You may act only
      on Your own behalf and on Your sole responsibility, not on behalf
      of any other Contributor, and only if You agree to indemnify,
      defend, and hold each Contributor harmless for any liability
      incurred by, or claims asserted against, such Contributor by reason
      of your accepting any such warranty or additional liability.

   END OF TERMS AND CONDITIONS
"""

ANTHROPIC_MIT_TEXT = """MIT License

Copyright (c) 2022 Anthropic

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

DO_NOT_TRAIN_TEXT = """================================================================================
DO NOT TRAIN ON THIS DIRECTORY -- PENDING JOEL'S DECISION ON REFUSAL/HEDGE FILTER
================================================================================

This dataset (Anthropic hh-rlhf) is EXCLUDED FROM ALL TRAINING in Parentheses.

Reason (see .agents/open/project-gaps-and-fixes.md item 1 and SWARM_2026-10-08.md):
Parentheses is an explicitly uncensored model.
Anthropic hh-rlhf contains human preference pairs ("chosen" vs "rejected") for
helpfulness and harmlessness. A substantial fraction of "chosen" responses are
refusals or safety-hedged answers. Training directly on the "chosen" responses
would quietly bake refusal/censorship behavior into the model.

Before this dataset can enter any training stage (pretraining or SFT), a filter
must be designed and implemented in data/prepare_sft.py to identify and drop
(or invert) examples where "chosen" is a refusal or safety hedge.

This filter design is explicitly gated on Joel's decision.
UNTIL JOEL DECIDES AND CLEARS THE FILTER, NO FILE IN THIS DIRECTORY MAY BE
TOKENIZED, PACKED, OR FED INTO ANY TRAINING RUN.
================================================================================
"""


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def download_repo(repo_id: str, dest_dir: Path, file_list: list[str]):
    dest_dir.mkdir(parents=True, exist_ok=True)
    results = {}
    print(f"\n[download] Starting {repo_id} -> {dest_dir}")
    for idx, rfname in enumerate(file_list, 1):
        target = dest_dir / rfname
        target.parent.mkdir(parents=True, exist_ok=True)
        t0 = time.time()
        print(f"  [{idx}/{len(file_list)}] Fetching {rfname} ...", end=" ", flush=True)
        # Download file to HF cache and copy to target if not already matching
        cached_path = hf_hub_download(repo_id=repo_id, filename=rfname, repo_type="dataset")
        shutil.copy2(cached_path, target)
        sz = target.stat().st_size
        h = sha256_file(target)
        dt = time.time() - t0
        speed = (sz / (1024 * 1024)) / max(dt, 0.001)
        print(f"done ({sz:,} bytes, {dt:.1f}s, {speed:.1f} MB/s)")
        results[rfname] = {"size": sz, "sha256": h}
    return results


def main():
    RAW_SOURCED.mkdir(parents=True, exist_ok=True)

    # 1. OASST1
    oasst1_dir = RAW_SOURCED / "oasst1"
    oasst1_files = [
        "2023-04-12_oasst_all.messages.jsonl.gz",
        "2023-04-12_oasst_all.trees.jsonl.gz",
        "2023-04-12_oasst_prompts.messages.jsonl.gz",
        "2023-04-12_oasst_ready.messages.jsonl.gz",
        "2023-04-12_oasst_ready.trees.jsonl.gz",
        "2023-04-12_oasst_spam.messages.jsonl.gz",
        "LICENSE",
        "README.md",
        "data/train-00000-of-00001-b42a775f407cee45.parquet",
        "data/validation-00000-of-00001-134b8fd0c89408b6.parquet",
    ]
    res_oasst1 = download_repo("OpenAssistant/oasst1", oasst1_dir, oasst1_files)

    # 2. OASST2
    oasst2_dir = RAW_SOURCED / "oasst2"
    oasst2_files = [
        "2023-11-05_oasst2_all.messages.jsonl.gz",
        "2023-11-05_oasst2_all.trees.jsonl.gz",
        "2023-11-05_oasst2_prompts.messages.jsonl.gz",
        "2023-11-05_oasst2_ready.messages.jsonl.gz",
        "2023-11-05_oasst2_ready.trees.jsonl.gz",
        "2023-11-05_oasst2_spam.messages.jsonl.gz",
        "README.md",
        "data/train-00000-of-00001-88ba0162028a73fc.parquet",
        "data/validation-00000-of-00001-1deeef95c3248fe0.parquet",
    ]
    res_oasst2 = download_repo("OpenAssistant/oasst2", oasst2_dir, oasst2_files)
    # Write LICENSE file for OASST2 as well
    (oasst2_dir / "LICENSE").write_text(APACHE_2_0_TEXT.strip() + "\n", encoding="utf-8")
    res_oasst2["LICENSE"] = {
        "size": (oasst2_dir / "LICENSE").stat().st_size,
        "sha256": sha256_file(oasst2_dir / "LICENSE"),
    }

    # 3. Anthropic hh-rlhf
    hh_dir = RAW_SOURCED / "hh-rlhf"
    hh_files = [
        "README.md",
        "harmless-base/test.jsonl.gz",
        "harmless-base/train.jsonl.gz",
        "helpful-base/test.jsonl.gz",
        "helpful-base/train.jsonl.gz",
        "helpful-online/test.jsonl.gz",
        "helpful-online/train.jsonl.gz",
        "helpful-rejection-sampled/test.jsonl.gz",
        "helpful-rejection-sampled/train.jsonl.gz",
        "red-team-attempts/red_team_attempts.jsonl.gz",
    ]
    res_hh = download_repo("Anthropic/hh-rlhf", hh_dir, hh_files)
    # Write LICENSE and DO_NOT_TRAIN.txt
    (hh_dir / "LICENSE").write_text(ANTHROPIC_MIT_TEXT.strip() + "\n", encoding="utf-8")
    res_hh["LICENSE"] = {
        "size": (hh_dir / "LICENSE").stat().st_size,
        "sha256": sha256_file(hh_dir / "LICENSE"),
    }
    (hh_dir / "DO_NOT_TRAIN.txt").write_text(DO_NOT_TRAIN_TEXT.strip() + "\n", encoding="utf-8")
    res_hh["DO_NOT_TRAIN.txt"] = {
        "size": (hh_dir / "DO_NOT_TRAIN.txt").stat().st_size,
        "sha256": sha256_file(hh_dir / "DO_NOT_TRAIN.txt"),
    }

    # Generate SOURCE.txt for OASST1
    total_sz_oasst1 = sum(v["size"] for v in res_oasst1.values())
    files_summary_oasst1 = [
        f"{len(res_oasst1)} files ({total_sz_oasst1:,} bytes total)"
    ]
    for fn in sorted(res_oasst1.keys()):
        files_summary_oasst1.append(f"- {fn} ({res_oasst1[fn]['size']:,} bytes, sha256: {res_oasst1[fn]['sha256']})")
    files_block_oasst1 = "\n".join("                " + line if i > 0 else line for i, line in enumerate(files_summary_oasst1))

    oasst1_source_txt = f"""Human-written, crowd-sourced multi-turn conversation trees in 35 languages across 161K+ messages.

source        : HuggingFace OpenAssistant/oasst1 (https://huggingface.co/datasets/OpenAssistant/oasst1)
pulled        : 2026-10-08
files         : {files_block_oasst1}
size          : {total_sz_oasst1:,} bytes ({total_sz_oasst1 / (1024*1024):.2f} MB / {total_sz_oasst1 / (1024*1024*1024):.3f} GB)
license       : Apache 2.0. Quoted verbatim from dataset card YAML frontmatter at
                https://huggingface.co/datasets/OpenAssistant/oasst1/raw/main/README.md:
                "license: apache-2.0"
                Full text of Apache License, Version 2.0 is included in LICENSE.

================================================================================
VERBATIM LICENSE TEXT (Apache License, Version 2.0)
================================================================================
{APACHE_2_0_TEXT.strip()}

================================================================================
SHA256 CHECKSUMS
================================================================================
""" + "\n".join(f"{res_oasst1[fn]['sha256']}  {fn}" for fn in sorted(res_oasst1.keys())) + "\n"

    (oasst1_dir / "SOURCE.txt").write_text(oasst1_source_txt, encoding="utf-8")
    (oasst1_dir / "SHA256SUMS.txt").write_text(
        "\n".join(f"{res_oasst1[fn]['sha256']}  {fn}" for fn in sorted(res_oasst1.keys())) + "\n",
        encoding="utf-8"
    )

    # Generate SOURCE.txt for OASST2
    total_sz_oasst2 = sum(v["size"] for v in res_oasst2.values())
    files_summary_oasst2 = [
        f"{len(res_oasst2)} files ({total_sz_oasst2:,} bytes total)"
    ]
    for fn in sorted(res_oasst2.keys()):
        files_summary_oasst2.append(f"- {fn} ({res_oasst2[fn]['size']:,} bytes, sha256: {res_oasst2[fn]['sha256']})")
    files_block_oasst2 = "\n".join("                " + line if i > 0 else line for i, line in enumerate(files_summary_oasst2))

    oasst2_source_txt = f"""OpenAssistant Conversations Release 2: expanded human-written multi-turn conversation trees in 35 languages.

source        : HuggingFace OpenAssistant/oasst2 (https://huggingface.co/datasets/OpenAssistant/oasst2)
pulled        : 2026-10-08
files         : {files_block_oasst2}
size          : {total_sz_oasst2:,} bytes ({total_sz_oasst2 / (1024*1024):.2f} MB / {total_sz_oasst2 / (1024*1024*1024):.3f} GB)
license       : Apache 2.0. Quoted verbatim from dataset card YAML frontmatter at
                https://huggingface.co/datasets/OpenAssistant/oasst2/raw/main/README.md:
                "license: apache-2.0"
                Full text of Apache License, Version 2.0 is included in LICENSE.

================================================================================
VERBATIM LICENSE TEXT (Apache License, Version 2.0)
================================================================================
{APACHE_2_0_TEXT.strip()}

================================================================================
SHA256 CHECKSUMS
================================================================================
""" + "\n".join(f"{res_oasst2[fn]['sha256']}  {fn}" for fn in sorted(res_oasst2.keys())) + "\n"

    (oasst2_dir / "SOURCE.txt").write_text(oasst2_source_txt, encoding="utf-8")
    (oasst2_dir / "SHA256SUMS.txt").write_text(
        "\n".join(f"{res_oasst2[fn]['sha256']}  {fn}" for fn in sorted(res_oasst2.keys())) + "\n",
        encoding="utf-8"
    )

    # Generate SOURCE.txt for Anthropic hh-rlhf
    total_sz_hh = sum(v["size"] for v in res_hh.values())
    files_summary_hh = [
        f"{len(res_hh)} files ({total_sz_hh:,} bytes total)"
    ]
    for fn in sorted(res_hh.keys()):
        files_summary_hh.append(f"- {fn} ({res_hh[fn]['size']:,} bytes, sha256: {res_hh[fn]['sha256']})")
    files_block_hh = "\n".join("                " + line if i > 0 else line for i, line in enumerate(files_summary_hh))

    hh_source_txt = f"""Human preference data for helpfulness and harmlessness assistant dialogues (preference pairs) and red teaming attempts.

source        : HuggingFace Anthropic/hh-rlhf (https://huggingface.co/datasets/Anthropic/hh-rlhf)
pulled        : 2026-10-08
files         : {files_block_hh}
size          : {total_sz_hh:,} bytes ({total_sz_hh / (1024*1024):.2f} MB / {total_sz_hh / (1024*1024*1024):.3f} GB)
license       : MIT License. Quoted verbatim from dataset card YAML frontmatter at
                https://huggingface.co/datasets/Anthropic/hh-rlhf/raw/main/README.md:
                "license: mit"
                And verbatim from official repository https://github.com/anthropics/hh-rlhf/raw/master/LICENSE.
                Full text of MIT License is included in LICENSE.

================================================================================
TRAINING STATUS / GATE
================================================================================
EXCLUDED FROM ALL TRAINING UNTIL JOEL DECIDES REFUSAL/HEDGE FILTER DESIGN.
See DO_NOT_TRAIN.txt in this folder and .agents/open/project-gaps-and-fixes.md item 1.
Parentheses is an uncensored model; hh-rlhf's 'chosen' branch encodes harmlessness
refusals and hedging which must be filtered out before SFT/training ingestion.

================================================================================
VERBATIM LICENSE TEXT (MIT License)
================================================================================
{ANTHROPIC_MIT_TEXT.strip()}

================================================================================
SHA256 CHECKSUMS
================================================================================
""" + "\n".join(f"{res_hh[fn]['sha256']}  {fn}" for fn in sorted(res_hh.keys())) + "\n"

    (hh_dir / "SOURCE.txt").write_text(hh_source_txt, encoding="utf-8")
    (hh_dir / "SHA256SUMS.txt").write_text(
        "\n".join(f"{res_hh[fn]['sha256']}  {fn}" for fn in sorted(res_hh.keys())) + "\n",
        encoding="utf-8"
    )

    # Clean up any .cache dirs that may have been created
    for d in [oasst1_dir, oasst2_dir, hh_dir]:
        cache_d = d / ".cache"
        if cache_d.exists():
            shutil.rmtree(cache_d, ignore_errors=True)

    print("\n[done] All datasets downloaded, SOURCE.txt and SHA256SUMS.txt written.")

    # Re-run build_attribution.py
    cmd = [sys.executable, "scripts/build_attribution.py"]
    print(f"\n[build_attribution] Running {' '.join(cmd)} ...")
    res = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True, check=True)
    print(res.stdout)


if __name__ == "__main__":
    main()
