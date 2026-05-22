# `scripts/` — runbooks for vtp-eval

| Script | Purpose |
|--------|---------|
| `vast/onstart.sh` | On-start for Vast.ai instances. Clones repo + installs Figure 3 deps. |
| `text_visual_attention/ui.sh` | Launch the Gradio web UI for picking image + words interactively. |
| `text_visual_attention/list_samples.sh` | Surface candidate COCO images + their full annotation labels. |
| `text_visual_attention/run.sh` | Run the Figure 3 reproduction on a chosen candidate. |

---

## 🟢 Reproducing Figure 3 of LearnPruner on Vast.ai

End-to-end runbook for renting a Vast instance and producing `text_visual_attention_reproduction.png`. **Total cost: ~$0.05–0.10**.

### 1. Rent the instance

1. Open <https://cloud.vast.ai/create/>.
2. **Template:** `PyTorch (Vast)` — image `vastai/pytorch:latest`.
3. **Search filters** (left pane):
   - GPU VRAM ≥ **24 GB** (LLaVA-1.5-7B fp16 peaks ~16 GB; need headroom)
   - Disk Space ≥ **50 GB** (LLaVA weights ~14 GB + COCO annotations ~250 MB + buffer)
   - Verified hosts only (toggle "Show Secure Cloud Only")
4. **GPU pick:** RTX 3090 / 4090 / A5000 / L4 / A100 — any 24 GB+ card works; pick the cheapest available (~$0.20–0.50/h).
5. **Disk:** set to **50 GB** in the right pane slider.
6. **Launch Mode:** *Jupyter-python notebook + SSH* (the default).
7. **Environment Variables** (add three rows):

   | Key | Value |
   |-----|-------|
   | `HF_HOME` | `/workspace/.cache/huggingface` |
   | `JUPYTER_DIR` | `/workspace` |
   | `DATA_DIRECTORY` | `/workspace/` |

8. **On-start Script:** paste the **entire contents** of [`scripts/vast/onstart.sh`](../scripts/vast/onstart.sh) into the box.

   The script will:
   - `git clone https://github.com/AnKun10/vtp-eval.git` into `/workspace/vtp-eval`
   - `git pull --ff-only` (so re-rents pick up new commits)
   - `bash install/text_visual_attention.sh` (transformers 4.49 + accelerate + matplotlib + pandas + pyyaml + `pip install -e .`)
   - Persist `HF_HOME` and auto-`cd /workspace/vtp-eval` for SSH sessions
   - Sanity-check the GPU and verify the `vtp_eval.insight.text_visual_attention` import

9. Click **RENT**. The on-start runs once at first boot (~30–60 s).

### 2. SSH in

1. After the instance shows **Running**, click it → **Direct SSH**. Copy the command:
   ```
   ssh -p <PORT> root@<HOST>
   ```
2. **Either** paste it into a Claude Code chat (Claude runs it through the Bash tool), **or** run it yourself in a local terminal.

   Vast needs your local SSH public key on file: **Account → Keys → +Add Key**. Paste the contents of `~/.ssh/id_ed25519.pub` (or `id_rsa.pub`) before the first SSH attempt.

3. Confirm on-start finished cleanly:
   ```bash
   tail -n 20 /workspace/onstart.log
   ```
   The log should end with `=== onstart finished: ...` and `GPU OK: <gpu name>`. If it halts mid-stream, re-run it — every step is idempotent:
   ```bash
   bash /workspace/vtp-eval/scripts/vast/onstart.sh
   ```

### 3. Pick a candidate image + target words

```bash
cd /workspace/vtp-eval      # (also automatic on SSH login)
bash scripts/text_visual_attention/list_samples.sh
```

This downloads COCO val2017 annotations (one-time, ~241 MB), then prints a table like:

```
====================================================================================
idx id        size          categories (area-ranked)
------------------------------------------------------------------------------------
0   324158    (640, 426)    person(120341), bicycle(85220), backpack(...), ...
1   ...
...
====================================================================================
```

and saves `outputs/coco_candidates.png` (thumbnail grid you can open in the Vast file browser).

**Read the printed categories**, decide which image you want, and pick 1–3 category names to track as target words.

### 4. Run the full reproduction

```bash
bash scripts/text_visual_attention/run.sh <idx> <word1> [word2] [word3]
```

Examples:

```bash
# Two single-word categories from candidate #0
bash scripts/text_visual_attention/run.sh 0 person bicycle

# A multi-word category — quote it
bash scripts/text_visual_attention/run.sh 3 person "sports ball"

# Three words on candidate #5
bash scripts/text_visual_attention/run.sh 5 dog frisbee person
```

On the first call, LLaVA-1.5-7B downloads (~14 GB, ~5 min). Subsequent runs reuse the HF cache and take ~30 s.

Outputs land in `/workspace/outputs/`:

| File | Content |
|------|---------|
| `text_visual_attention_reproduction.png` | **The Figure 3 reproduction** — 2D grid: rows = target words, cols = shallow / middle / deep layers |
| `text_visual_attention_metrics.csv` | Entropy, max-share, top-5% mass per (word, depth) |
| `text_visual_attention_metrics.png` | Bar chart of the metrics |
| `chosen_image.jpg` | The exact image used |
| `coco_candidates.png` | Thumbnail grid from step 3 |
| `candidates.json` | Machine-readable candidate list |

### 5. Verify the insight

`text_visual_attention_metrics.csv` should show, for **every** target word:

| token  | depth   | layer | entropy ↓ | max_share ↑ | top5pct_mass ↑ |
|--------|---------|-------|-----------|-------------|----------------|
| word_A | shallow | 2     | …         | …           | …              |
| word_A | middle  | 12    | **min**   | **max**     | **max**        |
| word_A | deep    | 30    | …         | …           | …              |

Visually, the **middle column** of `text_visual_attention_reproduction.png` should highlight the actual object location (per COCO annotation) for each word, while the shallow / deep columns are diffuse or collapsed onto sink tokens.

### 6. Retrieve outputs

**Option A — Vast web file browser:** instance page → *Files* tab → `/workspace/outputs/`.

**Option B — `scp` from your local machine:**
```bash
scp -P <PORT> -r root@<HOST>:/workspace/outputs ./figure3_outputs
```

**Option C — tarball:**
```bash
# inside SSH
tar czf /workspace/figure3_outputs.tgz -C /workspace outputs
# from your local terminal
scp -P <PORT> root@<HOST>:/workspace/figure3_outputs.tgz .
```

### 7. Destroy the instance

Vast > Instances > the instance > **Destroy**. The model cache lives on the instance volume, so destroying it means the next rent re-downloads LLaVA (~5 min). For repeated experiments, **Stop** instead — the volume is preserved and re-running on-start is a no-op.

---

## 🖥️ Interactive UI workflow

Same pipeline as the CLI, but with a single-page Gradio web UI for picking the image and target words by clicking instead of copy-pasting indices and words across shells.

### 1. Open an SSH local-forward tunnel

From your laptop (not inside the instance):

```bash
ssh -p <VAST_SSH_PORT> -L 7860:localhost:7860 root@<VAST_HOST>
```

The `-L 7860:localhost:7860` flag forwards `localhost:7860` on your laptop to `localhost:7860` on the Vast instance. The Gradio server binds `0.0.0.0` inside the instance but isn't exposed publicly — the SSH tunnel is the only access path.

### 2. Launch the UI on the instance

Inside the SSH session:

```bash
cd /workspace/vtp-eval
bash scripts/text_visual_attention/ui.sh
```

Gradio prints `Running on local URL: http://0.0.0.0:7860`. Leave the script running.

### 3. Use the UI

Open `http://localhost:7860` in your laptop's browser. Then:

1. Click **Load candidates** — 25 COCO thumbnails appear with annotation labels as captions.
2. Click a thumbnail — the *Selected* line updates and the target-words checkboxes populate from that image's full category list.
3. Tick 1–3 word checkboxes — the query box auto-fills (you can edit it). The Run button enables.
4. Click **Run Figure 3** — status messages stream as LLaVA loads (one-time, ~5 min) and the forward pass runs (~10 s).
5. Outputs render inline AND save to `/workspace/outputs/`:
   - `text_visual_attention_reproduction.png` — the heatmap grid
   - `text_visual_attention_metrics.png` — the entropy / top-5% bar chart
   - `text_visual_attention_metrics.csv` — the underlying metrics table

The model stays loaded between runs, so repeat experiments (different image or different words) finish in ~10 s.

### Troubleshooting (UI)

| Symptom | Fix |
|---------|-----|
| `OSError: [Errno 98] Address already in use` | A previous Gradio session is still bound. `pkill -f "vtp_eval.insight.text_visual_attention.ui"`, then re-launch. |
| Browser shows "connection refused" at `localhost:7860` | The SSH tunnel dropped or the UI isn't running. Check `bash scripts/text_visual_attention/ui.sh` is still printing in the SSH session, and that your local `ssh -L 7860:...` is alive. |
| UI loads but Run button stays grey | You picked an image but no words yet (or vice versa). Tick at least one checkbox. |
| Status reads "Words not found as contiguous tokens" | You edited the query and removed one of the words. Either add it back or untick the missing word. |

---


## Troubleshooting

| Symptom | Fix |
|---------|-----|
| `onstart.log` halts mid-stream | Re-run `bash scripts/vast/onstart.sh`. Idempotent. |
| `torch.cuda.is_available()` is `False` | Instance does not expose the GPU. Destroy and pick another host. |
| CUDA OOM during forward pass | < 24 GB VRAM. Rent a bigger card. |
| `Words not found as contiguous tokens` | Auto-built query already mentions each `--words` entry verbatim. If you pass `--query` yourself, ensure every word is in it. Multi-word categories must be quoted: `"sports ball"`. |
| `git pull` fails on re-boot | Local edits diverged from origin. Inspect with `git status`, then either commit or `git stash`. The on-start logs a warning but continues. |
| Cost overrun | One reproduction is ~7 min of GPU. If you're not actively SSH'd in, **Stop** (not Destroy) to pause billing while keeping the volume. |
