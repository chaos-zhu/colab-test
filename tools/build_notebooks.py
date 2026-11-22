# -*- coding: utf-8 -*-
"""產生兩本 Colab 筆記本（在 repo 根目錄執行：python tools/build_notebooks.py）
  H3_Basic_ComfyUI.ipynb   基礎：ComfyUI＋H3 模型組，不帶工作流
  H3_Digital_Human.ipynb   數字人：基礎＋工作流＋上傳照片／貼提示詞／生成
  H3_Toybox.ipynb          H3 Toybox 網頁前端
  H3_JZL_Nodes.ipynb       基礎＋ ComfyUI-JZL-MiniMax-H3 短劇導演台完整工作流（本地 llama-server＋Qwen3.5-9B）
筆記本是產物，要改請改這支再重新產生。"""
import json, io, os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
api = json.load(io.open('workflows/H3_digital_human_api.json', encoding='utf-8'))
fix_src = io.open('tools/userdata_slash_fix.py', encoding='utf-8').read()
model_cache_src = io.open('tools/model_cache.py', encoding='utf-8').read()
ui_wf = io.open('workflows/H3_digital_human.json', encoding='utf-8').read()
sample_prompt = api['7']['inputs']['prompt']
api['7']['inputs']['prompt'] = ''


def md(s):
    return {'cell_type': 'markdown', 'metadata': {}, 'source': s}


def code(s):
    return {'cell_type': 'code', 'metadata': {'cellView': 'form'}, 'execution_count': None, 'outputs': [], 'source': s}


TQ = '"' * 3
cells = [
    md("""# MiniMax H3 數字人｜Colab 版（一張照片＋一段提示詞 → 會說話的影片）

由上往下依序執行。**執行階段請選 L4 GPU**（A100 也可以但很耗點數；T4 跑不動）。

| 步驟 | 做什麼 | 時間（主人 L4 實測） |
|---|---|---|
| ① 檢查＋安裝 | GPU／驅動不符就停下，不安裝；通過才裝 ComfyUI＋節點 | 約 1 分鐘 |
| ② 啟動 | 背景啟動 ComfyUI，工作流放進清單 | 約 1 分鐘 |
| ③ 開介面 | Colab 內建代理（推薦）；③-b 是 cloudflare 備援 | 數秒 |
| ④ 下載模型 | 約 37 GB（ComfyUI 確定正常才下載） | 約 1 分鐘 |
| ⑤ 上傳照片 | 人物照一張 | — |
| ⑥ 貼提示詞 | 用 H3 提示詞 skill 產生後整段貼上 | — |
| ⑦ 生成 | 10 秒影片 | 約 4 分鐘 |

只想用介面操作節點：跑 ①②③④ 就好。只想用表單生成影片：③ 可以跳過。

**Hugging Face 權杖（建議）**：左側欄「🔑 密鑰」新增 `HF_TOKEN`（請用**唯讀**權杖），並開啟「筆記本存取權」。

**交給桌面版 AI 生成**：跑 ①②③-b④，把 ③-b 的網址交給裝了 `h3-colab-digital-human` skill 的 AI（skill 在 repo 的 `skills/`）。

模型：`Minimax-h3_Singularity_x_10Eros-Max_PRISM`（int8，加速已融合，6 步）＋官方 Qwen3-VL 文字編碼器與 VAE。授權依照各來源模型。

最新版：<https://github.com/chaos-zhu/colab-test>"""),
    code("#@title ① 檢查環境＋安裝 ComfyUI（GPU 或驅動不符就停下，不會安裝）\nFIX_SRC = " + repr(fix_src) + """
import torch, shutil, subprocess, sys, os
from pathlib import Path
if not torch.cuda.is_available():
    raise RuntimeError('沒有 GPU：請到「執行階段 → 變更執行階段類型」選 L4 GPU，再重跑這格')
cap = torch.cuda.get_device_capability(0)
drv = subprocess.run(['nvidia-smi', '--query-gpu=driver_version', '--format=csv,noheader'], capture_output=True, text=True).stdout.strip()
print('GPU:', torch.cuda.get_device_name(0), '| capability', cap, '| VRAM', round(torch.cuda.get_device_properties(0).total_memory / 2**30, 1), 'GiB')
print('驅動:', drv, '| 目前 PyTorch CUDA:', torch.version.cuda, '| 可用磁碟', round(shutil.disk_usage('/content').free / 2**30, 1), 'GiB')
if cap < (8, 0):
    raise RuntimeError(f'這張卡（capability {cap}）跑不動 int8 模型，沒有安裝任何東西。請換 L4（或 A100）再重跑。')
if int(drv.split('.')[0]) < 580:
    raise RuntimeError(f'驅動 {drv} 不支援 CUDA 13（需要 580 以上），沒有安裝任何東西。請換一台執行階段（L4）再試。')
print('✅ 環境檢查通過，開始安裝')
# HF 權杖先放進環境變數：之後啟動的 ComfyUI（含 ListHelper 的模型下載節點）都會沿用
try:
    from google.colab import userdata
    os.environ['HF_TOKEN'] = userdata.get('HF_TOKEN'); print('已讀到 HF_TOKEN')
except Exception:
    print('沒有設定 HF_TOKEN，之後改用匿名下載（人多時可能被限速）')
COMFY = Path('/content/ComfyUI')
if not COMFY.exists():
    subprocess.run(['git', 'clone', '--depth', '1', 'https://github.com/Comfy-Org/ComfyUI.git', str(COMFY)], check=True)
if not (torch.version.cuda or '').startswith('13'):
    print('安裝 CUDA 13 版 PyTorch（int8 ConvRot 權重需要）…')
    subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'torch', 'torchvision', 'torchaudio',
                    '--index-url', 'https://download.pytorch.org/whl/cu130'], check=True)
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', '-r', str(COMFY / 'requirements.txt'), 'huggingface_hub', 'hf_xet'], check=True)
# 預裝 ListHelper 節點組（含 ModelDownloader：在介面裡從 Hugging Face 抓模型）
LH = COMFY / 'custom_nodes' / 'ComfyUI-ListHelper'
if not LH.exists():
    subprocess.run(['git', 'clone', '--depth', '1', 'https://github.com/dseditor/ComfyUI-ListHelper.git', str(LH)], check=True)
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', '-r', str(LH / 'requirements.txt')], check=True)
# Colab 內建代理（③）會把網址的 %2F 解開，工作流清單點開會 404 → 裝一個補路由的小節點
FIX = COMFY / 'custom_nodes' / 'colab_userdata_slash_fix'
FIX.mkdir(parents=True, exist_ok=True)
(FIX / '__init__.py').write_text(FIX_SRC, encoding='utf-8')
# 節點管理器（ComfyUI-Manager 已併入官方，用 pip 套件＋啟動旗標 --enable-manager）
if (COMFY / 'manager_requirements.txt').exists():
    subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', '-r', str(COMFY / 'manager_requirements.txt')], check=True)
out = subprocess.run([sys.executable, '-c', 'import torch;print(torch.__version__, torch.version.cuda, torch.cuda.is_available())'],
                     capture_output=True, text=True)
print('ComfyUI 會用的 PyTorch：', out.stdout.strip(), out.stderr[-300:])
"""),
    code("#@title ② 啟動 ComfyUI（背景執行）＋把 H3 數字人工作流放進清單\nUI_WORKFLOW = " + repr(ui_wf) + """
import subprocess, time, urllib.request, sys
# --enable-cors-header：經 cloudflare 通道開介面時，瀏覽器的 Origin（trycloudflare 網址）
# 跟轉進來的 Host（127.0.0.1）對不上，ComfyUI 預設會回 403「存取遭拒」；開這個旗標才放行。
if 'server' in globals() and server.poll() is None:
    server.terminate(); server.wait(); print('已關閉舊的 ComfyUI，重新啟動')
# --enable-compress-response-body：介面要抓一大包節點清單和前端檔案，壓縮後走通道快很多。
# 節點管理器安全等級：normal- ＝可從清單裝節點，也可用 Git 網址安裝（本機模式下）。
# 這台 Colab 用完即丟；但 HF_TOKEN 在環境變數裡，自訂節點讀得到 → HF 權杖請用「唯讀」的。
cfg = COMFY / 'user' / '__manager' / 'config.ini'
cfg.parent.mkdir(parents=True, exist_ok=True)
if not cfg.exists():
    cfg.write_text('[default]\\nsecurity_level = normal-\\n', encoding='utf-8')
# 工作流放進清單（③、③-b 都看得到）；檔名用英文，避免網址編碼在代理上出問題
wf_dir = COMFY / 'user' / 'default' / 'workflows'
wf_dir.mkdir(parents=True, exist_ok=True)
(wf_dir / 'H3_digital_human.json').write_text(UI_WORKFLOW, encoding='utf-8')
old_cn = wf_dir / 'H3數字人.json'
if old_cn.exists(): old_cn.unlink()
Path('/content/H3_digital_human.json').write_text(UI_WORKFLOW, encoding='utf-8')
log = open('/content/comfyui.log', 'w')
server = subprocess.Popen([sys.executable, 'main.py', '--listen', '127.0.0.1', '--port', '8188', '--enable-cors-header', '--enable-manager', '--enable-compress-response-body'],
                          cwd=str(COMFY), stdout=log, stderr=subprocess.STDOUT)
for i in range(180):
    try:
        urllib.request.urlopen('http://127.0.0.1:8188/system_stats', timeout=2)
        print('ComfyUI 已啟動'); break
    except Exception:
        time.sleep(2)
else:
    print(open('/content/comfyui.log').read()[-3000:])
    raise RuntimeError('ComfyUI 沒有啟動成功，看上面的紀錄')
"""),
    code("""#@title ③ 開啟 ComfyUI 介面（推薦：Colab 內建代理，快、穩、網址不外流）
# 執行後下面會出現一個連結；顯示成 localhost 也沒關係，直接點它就會開新分頁。
# ⚠️ 開分頁的瀏覽器必須登入「跟這本 Colab 同一個 Google 帳號」；帳號不同就改用下一格 ③-b。
from google.colab import output
output.serve_kernel_port_as_window(8188)
# 如果左側工作流清單打不開（404），按下面下載 JSON，直接拖進 ComfyUI 畫面就好
from google.colab import files
files.download('/content/H3_digital_human.json')
"""),
    code("""#@title ③-b（備援）用 cloudflare 網址開介面：瀏覽器登入的 Google 帳號和 Colab 不同時用這格
# 會產生一個 trycloudflare 網址，點開就是 ComfyUI；也是交給桌面版 AI 操作時要給的網址。
# ⚠️ 網址沒有密碼，拿到的人都能用你的 GPU —— 不要貼到公開的地方。
import subprocess, re, time, json, urllib.request
if not Path('/content/cloudflared').exists():
    subprocess.run(['wget', '-q', '-O', '/content/cloudflared',
                    'https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64'], check=True)
    subprocess.run(['chmod', '+x', '/content/cloudflared'], check=True)
# 通道的輸出寫進檔案，不要接 PIPE：PIPE 沒人讀會塞滿，通道就卡住，網頁停在載入畫面
if 'tun' in globals() and tun.poll() is None:
    tun.terminate(); tun.wait()
open('/content/cloudflared.log', 'w').close()
tun = subprocess.Popen(['/content/cloudflared', 'tunnel', '--no-autoupdate', '--url', 'http://127.0.0.1:8188'],
                       stdout=open('/content/cloudflared.log', 'a'), stderr=subprocess.STDOUT)
url = None
for i in range(60):
    time.sleep(1)
    m = re.search(r'https://[a-z0-9-]+\.trycloudflare\.com', open('/content/cloudflared.log').read())
    if m:
        url = m.group(0); break
print('👉 ComfyUI 介面：', url or '（沒拿到網址，看 /content/cloudflared.log）')
"""),
    code("""#@title ④ 下載模型（約 37 GB）
#@markdown 預設啟用雲端硬碟快取：已有模型從「我的雲端硬碟/H3_models」複製，缺少的模型從 Hugging Face 下載並備份。首次使用需授權並預留雲端硬碟空間。
USE_DRIVE_CACHE = True  #@param {type:"boolean"}
#@markdown 加裝 ComfyUI 的 Z-Image Turbo 文生圖模板所需模型（多約 20 GB）。
INSTALL_Z_IMAGE_TURBO = False  #@param {type:"boolean"}
from huggingface_hub import hf_hub_download
import shutil, os
""" + model_cache_src + """
CACHE = None
if USE_DRIVE_CACHE:
    from google.colab import drive
    drive.mount('/content/drive')
    CACHE = Path('/content/drive/MyDrive/H3_models'); CACHE.mkdir(parents=True, exist_ok=True)
files = [
    ('OliviaRossi/Minimax-h3_Singularity-x-10Eros-Max_PRISM', 'Minimax-h3_Singularity_x_10Eros-Max_PRISM_int8.safetensors', 'diffusion_models'),
    ('Comfy-Org/MiniMax-H3', 'text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors', 'text_encoders'),
    ('Comfy-Org/MiniMax-H3', 'vae/minimax_h3_video_vae_int8_convrot.safetensors', 'vae'),
    ('Comfy-Org/MiniMax-H3', 'vae/minimax_h3_audio_vae_fp32.safetensors', 'vae'),
]
if INSTALL_Z_IMAGE_TURBO:
    files += [('Comfy-Org/Z-Image-Turbo', 'split_files/' + folder + '/' + name, folder) for folder, name in [
        ('vae', 'ae.safetensors'),
        ('text_encoders', 'qwen_3_4b.safetensors'),
        ('diffusion_models', 'z_image_turbo_bf16.safetensors'),
    ]]
for repo, name, folder in files:
    target = COMFY / 'models' / folder / Path(name).name
    target.parent.mkdir(parents=True, exist_ok=True)
    cached = CACHE / Path(name).name if CACHE else None
    ensure_model(repo, name, target, cached)
    print(folder, target.name, f'{target.stat().st_size / 2**30:.1f} GiB')
"""),
    code("""#@title ⑤ 上傳人物照（一張）
from google.colab import files
from IPython.display import Image as _Img, display
up = files.upload()
IMAGE_NAME = list(up)[0]
(COMFY / 'input').mkdir(exist_ok=True)
(COMFY / 'input' / IMAGE_NAME).write_bytes(up[IMAGE_NAME])
display(_Img(data=up[IMAGE_NAME], width=320))
"""),
    code("#@title ⑥ 貼上提示詞＋設定（執行後下方會出現輸入框）\n"
         "# 執行這格後，下面會出現一個大的輸入框：把 skill 產生的整段提示詞直接貼進去就好，\n"
         "# 不用管引號、換行。貼完不用再執行這格，直接執行 ⑦。\n"
         "import ipywidgets as W\n"
         "from IPython.display import display\n"
         "SAMPLE_PROMPT = " + repr(sample_prompt) + "\n"
         """_keep = prompt_box.value if 'prompt_box' in globals() else SAMPLE_PROMPT   # 重跑這格不會洗掉已貼的內容
prompt_box = W.Textarea(value=_keep, placeholder='把整段提示詞貼在這裡',
                        layout=W.Layout(width='100%', height='320px'))
dur_box = W.IntSlider(value=10, min=5, max=15, step=1, description='秒數')
mp_box = W.FloatText(value=0.4, description='百萬像素')
seed_box = W.IntText(value=-1, description='seed（-1 隨機）')
clear_btn = W.Button(description='清空輸入框')
clear_btn.on_click(lambda _: setattr(prompt_box, 'value', ''))
count = W.HTML()
def _count(change=None):
    n = len(prompt_box.value.split())
    count.value = f'<b>{n}</b> 個英文字（官方建議 350～500）'
prompt_box.observe(_count, names='value'); _count()
display(W.VBox([W.HBox([clear_btn, count]), prompt_box, dur_box, mp_box, seed_box]))
"""),
    code("#@title ⑦ 生成影片\n"
         "import json, random, time, urllib.request\n"
         "from IPython.display import Video, display\n"
         "WORKFLOW = json.loads(" + repr(json.dumps(api, ensure_ascii=False)) + ")\n"
         """p = json.loads(json.dumps(WORKFLOW))
PROMPT = prompt_box.value.strip()
assert PROMPT, '⑥ 的輸入框是空的，請先貼上提示詞'
print('這次送出的提示詞開頭：', PROMPT[:160].replace(chr(10), ' '), '…（共', len(PROMPT.split()), '字）')
DURATION_SECONDS, MEGAPIXELS, SEED = dur_box.value, mp_box.value, seed_box.value
p['7']['inputs']['prompt'] = PROMPT
p['5']['inputs']['image'] = IMAGE_NAME
p['56']['inputs']['value'] = int(DURATION_SECONDS)
p['61']['inputs']['megapixels'] = float(MEGAPIXELS)
p['54']['inputs']['seed'] = SEED if SEED >= 0 else random.randint(0, 2**50)
req = urllib.request.Request('http://127.0.0.1:8188/prompt', data=json.dumps({'prompt': p}).encode(),
                             headers={'Content-Type': 'application/json'})
try:
    pid = json.loads(urllib.request.urlopen(req).read())['prompt_id']
except urllib.error.HTTPError as e:
    raise RuntimeError(e.read().decode()[:2000])
t0 = time.time(); print('已送出，生成中…（seed', p['54']['inputs']['seed'], '）')
while True:
    time.sleep(10)
    h = json.loads(urllib.request.urlopen('http://127.0.0.1:8188/history/' + pid).read())
    if pid in h:
        st = h[pid]['status']
        if st.get('status_str') == 'error':
            print(json.dumps(st, ensure_ascii=False)[:3000]); raise RuntimeError('生成失敗')
        break
    print(f'  {int(time.time() - t0)} 秒', end='\\r')
vids = [v for o in h[pid]['outputs'].values() for v in o.get('images', []) + o.get('videos', []) if v['filename'].endswith('.mp4')]
out = COMFY / 'output' / vids[0]['subfolder'] / vids[0]['filename']
print(f'完成，{(time.time() - t0) / 60:.1f} 分鐘：', out)
display(Video(str(out), embed=True, width=640))
from google.colab import files; files.download(str(out))
"""),
    code("""#@title 🔍 出問題時跑這格：看 GPU 與 ComfyUI 紀錄
import subprocess
print(subprocess.run(['nvidia-smi'], capture_output=True, text=True).stdout)
log = open('/content/comfyui.log', encoding='utf-8', errors='ignore').read()
keys = ('error', 'Error', 'int8', 'quant', 'fallback', 'out of memory', 'loaded', 'Prompt executed')
print('\\n'.join(l for l in log.splitlines() if any(k in l for k in keys))[-4000:])
"""),
]


def without(cell, *snippets):
    """從共用格子拿掉數字人專用的片段；片段找不到就報錯，避免改了上面卻靜默沒拿掉。"""
    src = cell['source']
    for s in snippets:
        assert s in src, '找不到要拿掉的片段：' + s[:60]
        src = src.replace(s, '')
    return dict(cell, source=src)


HEAD_BASIC = md("""# MiniMax H3｜ComfyUI 基礎版（Colab）

在 Colab 上裝好 ComfyUI 與 MiniMax H3 模型組，**不帶任何工作流**——開好介面後自己拉節點、或拖入自己的工作流學習。
由上往下依序執行。**執行階段請選 L4 GPU**（A100 也可以但很耗點數；T4 跑不動）。

| 步驟 | 做什麼 | 時間（L4 實測） |
|---|---|---|
| ① 檢查＋安裝 | GPU／驅動不符就停下，不安裝；通過才裝 ComfyUI＋節點管理器＋ListHelper | 約 1 分鐘 |
| ② 啟動 | 背景啟動 ComfyUI | 約 1 分鐘 |
| ③ 開介面 | Colab 內建代理（推薦）；③-b 是 cloudflare 備援 | 數秒 |
| ④ 下載模型 | 約 37 GB（ComfyUI 確定正常才下載） | 約 1 分鐘 |

想直接做「會說話的數字人」：改開同一個 repo 的 `H3_Digital_Human.ipynb`。

**Hugging Face 權杖（建議）**：左側欄「🔑 密鑰」新增 `HF_TOKEN`（請用**唯讀**權杖），並開啟「筆記本存取權」。

模型：`Minimax-h3_Singularity_x_10Eros-Max_PRISM`（int8，加速已融合，6 步）＋官方 Qwen3-VL 文字編碼器與 VAE。授權依照各來源模型。

最新版：<https://github.com/chaos-zhu/colab-test>""")

C2_BASIC = without(cells[2],
    '＋把 H3 數字人工作流放進清單\nUI_WORKFLOW = ' + repr(ui_wf),
    """# 工作流放進清單（③、③-b 都看得到）；檔名用英文，避免網址編碼在代理上出問題
wf_dir = COMFY / 'user' / 'default' / 'workflows'
wf_dir.mkdir(parents=True, exist_ok=True)
(wf_dir / 'H3_digital_human.json').write_text(UI_WORKFLOW, encoding='utf-8')
old_cn = wf_dir / 'H3數字人.json'
if old_cn.exists(): old_cn.unlink()
Path('/content/H3_digital_human.json').write_text(UI_WORKFLOW, encoding='utf-8')
""")
C3_BASIC = without(cells[3], """# 如果左側工作流清單打不開（404），按下面下載 JSON，直接拖進 ComfyUI 畫面就好
from google.colab import files
files.download('/content/H3_digital_human.json')
""")

# ---------------------------------------------------------------- H3 Toybox（網頁前端）
# ComfyUI 只當背景引擎、不開它的介面；瀏覽器只開 toybox/ 這個前端（Colab 內建代理）。
TOYBOX = [
    md("""# H3 Toybox｜Colab 版（可愛的 MiniMax H3 網頁前端）

把想像丟進玩具箱，搖一搖就變成影片 🧸 ComfyUI 只在背景當引擎，**不用開 ComfyUI 介面**，所有操作都在 H3 Toybox 網頁裡。

由上往下依序執行。**執行階段請選 L4 GPU**（A100 也可以但很耗點數；T4 跑不動）。

| 步驟 | 做什麼 |
|---|---|
| ① 檢查＋安裝 | GPU／驅動不符就停下；通過才裝 ComfyUI、3 個節點組、H3 Toybox |
| ② 下載模型 | 約 50 GB（含 7 GB 的提示詞改寫模型，可以不勾）；勾「加裝生圖」再多約 10 GB |
| ③ 啟動＋開網頁 | 背景啟動 ComfyUI 與 H3 Toybox，用 Colab 內建代理開網頁；作品預設存到雲端硬碟「H3Toybox」資料夾 |
| ③-b | 瀏覽器的 Google 帳號和 Colab 不同時，改用 cloudflare 網址 |

**網頁裡能做的事**
- 🖼 **圖片**（選用）：文生圖，或拿最多 8 張圖做編輯（換表情、換衣服、換風格、把人放進場景）。用 [P-H3-Image](https://huggingface.co/PrunaAI/P-H3-Image-Comfy) 掛在同一顆 H3 模型上，只多約 10 GB
- 🤖 **短片 Agent**：一句話規劃成多段短片（要先在 ⚙ 設定填 Gemini 金鑰）
- 🧸 **參考生成**：圖片 9 張／影片 3 支／聲音 3 段當參考，提示詞裡用 `@圖片1` 標記
- 🎞 **頭尾格**：只放首格、只放尾格、或兩張都放
- ✏️ **純文字**：只靠一段描述
- 🧵 **延伸影片**：參考影片勾「延伸這支影片」，新片段會從它的最後一格接下去，並自動接回原片
- ✨ **提示詞改寫**：把中文描述擴寫成 H3 規格的英文提示詞（改寫結果預設收起來）。預設用 H3 官方的 9B 改寫模型；也可以在 ⚙ 設定改用 **Gemini** 或 **OpenAI 相容端點**
- ⚡ **快速模式**（預設開）：8 步出片；關掉是 12 步
- 📁 **素材庫資料夾**：把圖片、影片、聲音拖進資料夾整理
- 每支影片都會顯示目前做到哪一步、還要多久

**Hugging Face 權杖（建議）**：左側欄「🔑 密鑰」新增 `HF_TOKEN`（請用**唯讀**權杖），並開啟「筆記本存取權」。

模型：`Minimax-h3_Singularity_x_10Eros-Max_PRISM`（int8）＋官方 Qwen3-VL 文字編碼器與 VAE；LoRA、語意橋、潛空間放大器與改寫模型鏡像在 [dseditor/minimax-h3-studio-assets](https://huggingface.co/dseditor/minimax-h3-studio-assets)（各檔來源見該頁）。授權依照各來源模型。

最新版：<https://github.com/chaos-zhu/colab-test>"""),
    code("""#@title ① 檢查環境＋安裝 ComfyUI、節點與 H3 Toybox（GPU 或驅動不符就停下，不會安裝）
import torch, shutil, subprocess, sys, os
from pathlib import Path
if not torch.cuda.is_available():
    raise RuntimeError('沒有 GPU：請到「執行階段 → 變更執行階段類型」選 L4 GPU，再重跑這格')
cap = torch.cuda.get_device_capability(0)
drv = subprocess.run(['nvidia-smi', '--query-gpu=driver_version', '--format=csv,noheader'], capture_output=True, text=True).stdout.strip()
print('GPU:', torch.cuda.get_device_name(0), '| capability', cap, '| VRAM', round(torch.cuda.get_device_properties(0).total_memory / 2**30, 1), 'GiB')
print('驅動:', drv, '| 可用磁碟', round(shutil.disk_usage('/content').free / 2**30, 1), 'GiB')
if cap < (8, 0):
    raise RuntimeError(f'這張卡（capability {cap}）跑不動 int8 模型，沒有安裝任何東西。請換 L4（或 A100）再重跑。')
if int(drv.split('.')[0]) < 580:
    raise RuntimeError(f'驅動 {drv} 不支援 CUDA 13（需要 580 以上），沒有安裝任何東西。請換一台執行階段（L4）再試。')
if shutil.disk_usage('/content').free < 72 * 2**30:
    raise RuntimeError('磁碟剩不到 72 GiB，放不下約 60 GB 的模型。請換一台執行階段。')
print('✅ 環境檢查通過，開始安裝')
try:
    from google.colab import userdata
    os.environ['HF_TOKEN'] = userdata.get('HF_TOKEN'); print('已讀到 HF_TOKEN')
except Exception:
    print('沒有設定 HF_TOKEN，之後改用匿名下載（人多時可能被限速）')

def clone(url, dest):
    if not Path(dest).exists():
        subprocess.run(['git', 'clone', '--depth', '1', url, str(dest)], check=True)
    else:
        subprocess.run(['git', '-C', str(dest), 'pull', '--ff-only'], check=False)

COMFY = Path('/content/ComfyUI')
clone('https://github.com/Comfy-Org/ComfyUI.git', COMFY)
if not (torch.version.cuda or '').startswith('13'):
    print('安裝 CUDA 13 版 PyTorch（int8 ConvRot 權重需要）…')
    subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'torch', 'torchvision', 'torchaudio',
                    '--index-url', 'https://download.pytorch.org/whl/cu130'], check=True)
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', '-r', str(COMFY / 'requirements.txt'), 'huggingface_hub', 'hf_xet'], check=True)
# 工作流用到的三個節點組（都不會動到 PyTorch）
NODES = {
    'minimax-h3-audio-t8': 'https://github.com/T8mars/comfyui-minimax-h3-audio-T8',        # 語意橋 Semantic Bridge
    'comfyui-SelfLift': 'https://github.com/facok/comfyui-SelfLift',                      # 低解析度→潛空間放大→高解析度
    'ComfyUI-H3-Motion-Context': 'https://github.com/NikoDemon80/ComfyUI-H3-Motion-Context',  # 延伸影片（氣鎖）
}
for name, url in NODES.items():
    clone(url, COMFY / 'custom_nodes' / name)
# 導演 Skill 只是文字檔：放在 custom_nodes 外面，不載入它的節點
clone('https://github.com/T8mars/comfyui-minimax-h3-prompt-enhancer-T8', '/content/t8-enhancer')
# H3 Toybox 本體
clone('https://github.com/chaos-zhu/colab-test.git', '/content/minimax-h3-colab')
APP = Path('/content/minimax-h3-colab/toybox')
print('ffmpeg:', shutil.which('ffmpeg') or '沒有（延伸影片會失敗）')
out = subprocess.run([sys.executable, '-c', 'import torch;print(torch.__version__, torch.version.cuda, torch.cuda.is_available())'],
                     capture_output=True, text=True)
print('ComfyUI 會用的 PyTorch：', out.stdout.strip(), out.stderr[-300:])
print('✅ 安裝完成')
"""),
    code("""#@title ② 下載模型（約 50 GB）
#@markdown 預設啟用雲端硬碟快取：已有模型從「我的雲端硬碟/H3_models」複製，缺少的模型從 Hugging Face 下載並備份。首次使用需授權並預留雲端硬碟空間。
USE_DRIVE_CACHE = True  #@param {type:"boolean"}
#@markdown 不勾就少下載 7 GB：網頁的本地改寫會關掉，可以到 ⚙ 設定改用 Gemini 或 OpenAI 端點改寫。
INCLUDE_REWRITER = True  #@param {type:"boolean"}
#@markdown 🖼 加裝生圖（多約 10 GB）：網頁的「圖片」分頁可以文生圖、拿最多 8 張圖做編輯。掛在同一顆 H3 模型上。
INSTALL_IMAGE = True  #@param {type:"boolean"}
#@markdown 加裝 ComfyUI 的 Z-Image Turbo 文生圖模板所需模型（多約 20 GB；不是 Toybox 圖片分頁）。
INSTALL_Z_IMAGE_TURBO = False  #@param {type:"boolean"}
from huggingface_hub import hf_hub_download
import shutil, time
""" + model_cache_src + """
CACHE = None
if USE_DRIVE_CACHE:
    from google.colab import drive
    drive.mount('/content/drive')
    CACHE = Path('/content/drive/MyDrive/H3_models'); CACHE.mkdir(parents=True, exist_ok=True)
ASSETS = 'dseditor/minimax-h3-studio-assets'
files = [  # (repo, 檔案, 放到 models/ 底下的哪裡)
    ('OliviaRossi/Minimax-h3_Singularity-x-10Eros-Max_PRISM', 'Minimax-h3_Singularity_x_10Eros-Max_PRISM_int8.safetensors', 'diffusion_models'),
    ('Comfy-Org/MiniMax-H3', 'text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors', 'text_encoders'),
    ('Comfy-Org/MiniMax-H3', 'vae/minimax_h3_video_vae_int8_convrot.safetensors', 'vae'),
    ('Comfy-Org/MiniMax-H3', 'vae/minimax_h3_audio_vae_fp32.safetensors', 'vae'),
    # pika flex Ref2VA（2026-11-04 取代 dmad）：影片與 32×32 TTS 共用這一顆
    ('Comfy-Org/MiniMax-H3', 'loras/pika_h3_flex_ref2va_v1_bf16.safetensors', 'loras'),
] + [(ASSETS, f, str(Path(f).parent)) for f in [
    'loras/minimax_h3_turbo_v4_step600_ema_DasiwaREF2VAHybridV1_0_curveproj1025_compat_v001.safetensors',
    'loras/MysticXXX_MMH3-V2.safetensors',
    'loras/H3_Combat_V2.safetensors',
    'loras/Motion_Repair.safetensors',
    'loras/H3_SPEED_slider.safetensors',
    'semantic_bridge/t8_compat/BUNNY_H3_ActionLogic_Bridge_V1_T8_Compat.safetensors',
    'latent_upscale_models/minimax_h3_latent_upscaler_3d_fp16_V2.safetensors',
] + (['text_encoders/minimax_h3_rewriter_ref2va_qwen3.5_9b_w4a8.safetensors'] if INCLUDE_REWRITER else [])] + ([
    ('PrunaAI/P-H3-Image-Comfy', 'loras/p-h3-image_lora_bf16.safetensors', 'loras'),
    ('PrunaAI/P-H3-Image-Comfy', 'text_encoders/p-h3-image_text_encoder_int8_convrot.safetensors', 'text_encoders'),
    ('PrunaAI/P-H3-Image-Comfy', 'vae/p-h3-image_vae_int8_convrot.safetensors', 'vae'),
] if INSTALL_IMAGE else [])
if INSTALL_Z_IMAGE_TURBO:
    files += [('Comfy-Org/Z-Image-Turbo', 'split_files/' + folder + '/' + name, folder) for folder, name in [
        ('vae', 'ae.safetensors'),
        ('text_encoders', 'qwen_3_4b.safetensors'),
        ('diffusion_models', 'z_image_turbo_bf16.safetensors'),
    ]]
t0 = time.time()
for repo, name, folder in files:
    target = COMFY / 'models' / folder / Path(name).name
    target.parent.mkdir(parents=True, exist_ok=True)
    cached = CACHE / Path(name).name if CACHE else None
    ensure_model(repo, name, target, cached)
    print(f'{folder:<28} {target.name[:60]:<60} {target.stat().st_size / 2**30:5.1f} GiB')
# 官方提示詞寫作指南（改寫模型照它寫）
GUIDES = APP / 'guides'; GUIDES.mkdir(exist_ok=True)
for g in ('VIDEO_PROMPT_WRITING_GUIDE_ref_en.md', 'VIDEO_PROMPT_WRITING_GUIDE_base_en.md'):
    got = Path(hf_hub_download(repo_id='MiniMaxAI/MiniMax-H3', filename='docs/' + g, local_dir='/content/hf'))
    shutil.copy(str(got), str(GUIDES / g))
print(f'✅ 下載完成，{(time.time() - t0) / 60:.1f} 分鐘')
"""),
    code("""#@title ③ 啟動 ComfyUI（背景）＋ H3 Toybox，開網頁
# 執行後下面會出現一個連結；顯示成 localhost 也沒關係，直接點它就會開新分頁。
# ⚠️ 開分頁的瀏覽器必須登入「跟這本 Colab 同一個 Google 帳號」；帳號不同就改用下一格 ③-b。
#@markdown **作品存到雲端硬碟**：素材庫、資料夾、生成的影片、生成紀錄都放在「我的雲端硬碟/H3Toybox」，
#@markdown 斷線或不小心關掉執行階段，重開後再跑一次 ①②③ 東西都還在。第一次會跳出授權視窗，按同意即可。
SAVE_TO_DRIVE = True  #@param {type:"boolean"}
#@markdown API 金鑰也存到雲端硬碟（下次不用重貼；但雲端硬碟裡會有一份金鑰檔）
SAVE_API_KEYS = False  #@param {type:"boolean"}
import subprocess, time, urllib.request, json, sys, os, shutil
def wait_http(url, name, log, tries=180):
    for i in range(tries):
        try:
            urllib.request.urlopen(url, timeout=2); print(name, '已啟動'); return
        except Exception:
            time.sleep(2)
    print(open(log, encoding='utf-8', errors='ignore').read()[-3000:])
    raise RuntimeError(name + ' 沒有啟動成功，看上面的紀錄')

for proc in ('server', 'toybox'):
    if proc in globals() and globals()[proc].poll() is None:
        globals()[proc].terminate(); globals()[proc].wait()
# 作品放哪裡：雲端硬碟（授權成功）或這台執行階段的暫存空間
DATA = Path('/content/h3toybox_data')
if SAVE_TO_DRIVE:
    try:
        from google.colab import drive
        drive.mount('/content/drive')
        DATA = Path('/content/drive/MyDrive/H3Toybox')
    except Exception as e:
        print('⚠️ 沒有掛上雲端硬碟（', e, '）：這次的作品只存在暫存空間，關掉執行階段就會消失')
DATA.mkdir(parents=True, exist_ok=True)
# ComfyUI 的輸出資料夾直接設在 DATA/output。不能用連結把 output/H3Toybox 接過去：
# ComfyUI 存檔與 /view 都會解開連結檢查真實位置，指到 output 外面就報
# 「Saving image outside the output folder is not allowed」。
output_root = DATA / 'output'
videos, out_dir = output_root / 'H3Toybox', COMFY / 'output' / 'H3Toybox'
videos.mkdir(parents=True, exist_ok=True)
# 舊版留下的東西搬過來，以前生成的影片才看得到：DATA/videos、舊連結指向的資料夾、機器上的 output/H3Toybox
old_dirs = [DATA / 'videos']
if out_dir.is_symlink():
    old_dirs.append(out_dir.resolve())
    out_dir.unlink()
elif out_dir.is_dir():
    old_dirs.append(out_dir)
for old in old_dirs:
    if old.is_dir() and old.resolve() != videos.resolve():
        for f in old.iterdir():
            if f.is_file() and not (videos / f.name).exists():
                shutil.move(str(f), str(videos / f.name))
# ComfyUI 只當引擎：只聽 127.0.0.1，不對外開介面
server = subprocess.Popen([sys.executable, 'main.py', '--listen', '127.0.0.1', '--port', '8188',
                           '--output-directory', str(output_root)],
                          cwd=str(COMFY), stdout=open('/content/comfyui.log', 'w'), stderr=subprocess.STDOUT)
wait_http('http://127.0.0.1:8188/system_stats', 'ComfyUI', '/content/comfyui.log')
print('🎬 影片寫到：', videos)
# H3 Toybox 的 Colab 設定（保留使用者在網頁設定頁改過的內容，只覆寫這台機器的路徑）
(COMFY / 'input').mkdir(exist_ok=True)
cfg_path = DATA / 'config.json'
cfg = json.loads(cfg_path.read_text(encoding='utf-8')) if cfg_path.exists() else {}
cfg.update({
    'comfy_url': 'http://127.0.0.1:8188',
    'comfy_input_dir': str(COMFY / 'input'),
    'skills_dir': '/content/t8-enhancer/directional_skills',
    'free_after_rewrite': True,   # Colab 記憶體比較小：改寫完就把 9B 模型卸載
})
cfg.pop('guides', None)
# 🖼 圖片分頁：Colab 用 P-H3-Image（掛在 PRISM 上）；有沒有裝，看檔案是不是真的在
PH3 = ('loras/p-h3-image_lora_bf16.safetensors', 'text_encoders/p-h3-image_text_encoder_int8_convrot.safetensors',
       'vae/p-h3-image_vae_int8_convrot.safetensors')
# 🤖 短片 Agent：Colab 上沒有本地大模型，預設改走 Gemini（金鑰在網頁 ⚙ 設定填）
cfg['agent'] = dict(cfg.get('agent') or {}, provider=(cfg.get('agent') or {}).get('provider', 'gemini'))
cfg['image'] = dict(cfg.get('image') or {}, engine='ph3',
                    installed=all((COMFY / 'models' / f).is_file() for f in PH3))
print('🖼 生圖：', '已安裝' if cfg['image']['installed'] else '沒有安裝（② 勾 INSTALL_IMAGE 才有）')
cfg_path.write_text(json.dumps(cfg, ensure_ascii=False, indent=1), encoding='utf-8')
secrets = DATA / 'secrets.json' if SAVE_API_KEYS else Path('/content/h3toybox_secrets.json')
toybox = subprocess.Popen([sys.executable, 'server.py'], cwd=str(APP),
                          env=dict(os.environ, H3TOYBOX_PORT='8190', H3TOYBOX_DATA=str(DATA), H3TOYBOX_SECRETS=str(secrets)),
                          stdout=open('/content/toybox.log', 'w'), stderr=subprocess.STDOUT)
wait_http('http://127.0.0.1:8190/api/config', 'H3 Toybox', '/content/toybox.log', tries=30)
print('📁 作品存放位置：', DATA)
from google.colab import output
output.serve_kernel_port_as_window(8190, anchor_text='🧸 打開 H3 Toybox')
"""),
    code("""#@title ③-b（備援）用 cloudflare 網址開 H3 Toybox：瀏覽器登入的 Google 帳號和 Colab 不同時用這格
# ⚠️ 網址沒有密碼，拿到的人都能用你的 GPU —— 不要貼到公開的地方。
import subprocess, re, time
if not Path('/content/cloudflared').exists():
    subprocess.run(['wget', '-q', '-O', '/content/cloudflared',
                    'https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64'], check=True)
    subprocess.run(['chmod', '+x', '/content/cloudflared'], check=True)
# 通道的輸出寫進檔案，不要接 PIPE：PIPE 沒人讀會塞滿，通道就卡住
if 'tun' in globals() and tun.poll() is None:
    tun.terminate(); tun.wait()
open('/content/cloudflared.log', 'w').close()
tun = subprocess.Popen(['/content/cloudflared', 'tunnel', '--no-autoupdate', '--url', 'http://127.0.0.1:8190'],
                       stdout=open('/content/cloudflared.log', 'a'), stderr=subprocess.STDOUT)
url = None
for i in range(60):
    time.sleep(1)
    m = re.search(r'https://[a-z0-9-]+\\.trycloudflare\\.com', open('/content/cloudflared.log').read())
    if m:
        url = m.group(0); break
print('👉 H3 Toybox：', url or '（沒拿到網址，看 /content/cloudflared.log）')
"""),
    code("""#@title 🔍 出問題時跑這格：看 GPU、ComfyUI 與 H3 Toybox 的紀錄
import subprocess
print(subprocess.run(['nvidia-smi'], capture_output=True, text=True).stdout)
print(subprocess.run(['free', '-g'], capture_output=True, text=True).stdout)
log = open('/content/comfyui.log', encoding='utf-8', errors='ignore').read()
keys = ('error', 'Error', 'Traceback', 'out of memory', 'SelfLift timing] high_resolution total', 'Prompt executed', 'IMPORT FAILED')
print('\\n'.join(l for l in log.splitlines() if any(k in l for k in keys))[-4000:])
print('---- H3 Toybox ----')
print(open('/content/toybox.log', encoding='utf-8', errors='ignore').read()[-2000:])
"""),
]

# ---------------------------------------------------------------- JZL MiniMax H3 节点包
# ComfyUI＋H3 模型组＋ wjluoxiao/ComfyUI-JZL-MiniMax-H3，开箱即用「短剧导演台」完整链路：
# 本地 LLM（llama-server＋Qwen3.5-9B）拆解剧本 → 逐段 ref2va 生成 → 拼接成片。
LLM_REPO = 'unsloth/Qwen3.5-9B-GGUF'
LLM_FILE, LLM_MMPROJ, LLM_CHAT_HANDLER = 'Qwen3.5-9B-Q4_K_M.gguf', 'mmproj-F16.gguf', 'Qwen3.5'
SAMPLE_STORY = ('雨夜，老城区一间快要打烊的旧书店。年轻店主林夏正在收拾书架，门铃忽然响起，'
                '一位浑身湿透的白发老人走进来，把一本没有书名的旧书放在柜台上，轻声说：“这本书，三十年前是你外婆卖给我的。”'
                '林夏翻开扉页，看到外婆熟悉的字迹，眼眶慢慢红了。老人笑着说：“现在，它该回家了。”'
                '他转身走进雨里，林夏追到门口，街上只剩下路灯和雨声。')
DIRECTOR_MANAGER = {   # 只写和默认值不同的部分；插件会递归补齐其余键
    'save': {'mode': '拼接保存', 'auto_merge_delete': False},
    'assets': {'images': [], 'videos': [], 'audios': []},
    'enhance': {
        'llm_backend': '本地模型 [local]', 'force_offload': True, 'enabled': False, 'prompt_lang': '中文 [ZH]',
        # 加载 mmproj（视觉模块）时插件要求 chat_handler 不能是 None；llama-server 下它只是「启用视觉」的开关，
        # 选和模型对应的非思考版（Thinking 版会先长篇推理，拆解剧本慢很多）
        'llm': {'model': LLM_FILE, 'mmproj': LLM_MMPROJ, 'chat_handler': LLM_CHAT_HANDLER, 'backend': 'llama-server',
                'n_ctx': 32768, 'max_tokens': 16384},
    },
    'sample_decode': {'steps': 6, 'sampler': 'res_multistep', 'scheduler': 'simple', 'decode_cleanup': '卸载显存模型'},
}
DIRECTOR_API = {
    '1': {'class_type': 'UNETLoader', 'inputs': dict(api['1']['inputs']), '_meta': {'title': '主模型'}},
    '2': {'class_type': 'CLIPLoader', 'inputs': dict(api['2']['inputs']), '_meta': {'title': 'CLIP'}},
    '3': {'class_type': 'VAELoader', 'inputs': dict(api['3']['inputs']), '_meta': {'title': '视觉VAE'}},
    '4': {'class_type': 'VAELoader', 'inputs': dict(api['4']['inputs']), '_meta': {'title': '音频VAE'}},
    '5': {'class_type': 'BlockSparseAttention', 'inputs': dict(api['59']['inputs'], model=['1', 0]), '_meta': {'title': 'BlockSparseAttention'}},
    '10': {'class_type': 'JZL_MiniMaxAssetManagerShortDrama', 'inputs': {
        'run_mode': '🛠️ 故事拆解模式', 'display_info': '', 'aspect_ratio': '16:9 (Widescreen)', 'megapixels': 0.4,
        'duration': '固定5秒', 'scale_factor': 1.0, 'upscale_scale': 1.0, 'video_count': 3, 'story_style': '',
        'story_name': 'colab_demo', 'internal_prompt': SAMPLE_STORY,
        'manager_settings': json.dumps(DIRECTOR_MANAGER, ensure_ascii=False),
        'model': ['5', 0], 'clip': ['2', 0], 'vae': ['3', 0], 'audio_vae': ['4', 0]},
        '_meta': {'title': 'JZL - 🤖 MiniMax-H3短剧导演台'}},
}

JZL = [
    md("""# MiniMax H3 × ComfyUI-JZL-MiniMax-H3｜Colab 版（短剧导演台，开箱即用）

在 Colab 上跑 [wjluoxiao/ComfyUI-JZL-MiniMax-H3](https://github.com/wjluoxiao/ComfyUI-JZL-MiniMax-H3) 的完整工作流：
**一段故事（＋可选的角色／场景／道具图）→ 本地 LLM 拆成分镜剧本 → 逐段生成有声视频 → 自动拼接成片**。
由上往下依序执行。**运行时请选 L4 GPU**（A100 也可以但很耗点数；T4 跑不动 int8 模型）。

| 步骤 | 做什么 | 时间（L4 预估） |
|---|---|---|
| ① 检查＋安装 | GPU／驱动不符就停下；通过才装 ComfyUI、节点管理器、JZL 节点包、VideoHelperSuite | 约 1～2 分钟 |
| ② 本地 LLM 运行时 | 编译 CUDA 版 `llama-server`（只编当前显卡架构）；编好缓存到云端硬盘，下次直接还原 | 首次约 5～15 分钟 |
| ③ 启动 | 后台启动 ComfyUI，确认 JZL 节点已注册 | 约 1 分钟 |
| ④ 开界面 | Colab 内置代理（推荐）；④-b 是 cloudflare 备用 | 数秒 |
| ⑤ 下载模型 | H3 模型组约 37 GB＋ Qwen3.5-9B（GGUF 约 6 GB）＋可选潜空间放大模型 | 数分钟 |
| ⑥（可选）导入工作流 | 上传自己的 JZL 工作流 JSON，放进左侧清单，并检查缺哪些节点和模型 | — |
| ⑦（可选）素材 | 上传角色／场景／道具图，设定名称；故事里写名称就会当参考 | — |
| ⑧ 故事＋设置 | 贴故事、选风格／画幅／每段时长／段数；LLM 用本地或在线 API | — |
| ⑨ 生成 | 导演台：LLM 拆解 → 逐段生成 → 拼接，完成后播放并下载 | 每 5 秒一段约 2～3 分钟 |

不想等编译：② 选 `skip`，⑧ 的 LLM 改选「在线API」，填 Gemini／OpenAI 兼容端点（DeepSeek、Qwen、GLM、Kimi…）的 Key。

**Hugging Face 令牌（建议）**：左侧栏「🔑 密钥」新增 `HF_TOKEN`（请用**只读**令牌），并开启「笔记本访问权限」。

模型：`Minimax-h3_Singularity_x_10Eros-Max_PRISM`（int8，加速已融合，6 步）＋官方 Qwen3-VL 文本编码器与 VAE；
LLM：[unsloth/Qwen3.5-9B-GGUF](https://huggingface.co/unsloth/Qwen3.5-9B-GGUF)（Q4_K_M＋mmproj）。授权依照各来源模型；节点包为 MIT 授权。

最新版：<https://github.com/chaos-zhu/colab-test>"""),
    code("#@title ① 检查环境＋安装 ComfyUI 与 JZL 节点包（GPU 或驱动不符就停下，不会安装）\nFIX_SRC = " + repr(fix_src) + """
import torch, shutil, subprocess, sys, os
from pathlib import Path
if not torch.cuda.is_available():
    raise RuntimeError('没有 GPU：请到「运行时 → 更改运行时类型」选 L4 GPU，再重跑这格')
cap = torch.cuda.get_device_capability(0)
drv = subprocess.run(['nvidia-smi', '--query-gpu=driver_version', '--format=csv,noheader'], capture_output=True, text=True).stdout.strip()
print('GPU:', torch.cuda.get_device_name(0), '| capability', cap, '| VRAM', round(torch.cuda.get_device_properties(0).total_memory / 2**30, 1), 'GiB')
print('驱动:', drv, '| 当前 PyTorch CUDA:', torch.version.cuda, '| 可用磁盘', round(shutil.disk_usage('/content').free / 2**30, 1), 'GiB')
if cap < (8, 0):
    raise RuntimeError(f'这张卡（capability {cap}）跑不动 int8 模型，没有安装任何东西。请换 L4（或 A100）再重跑。')
if int(drv.split('.')[0]) < 580:
    raise RuntimeError(f'驱动 {drv} 不支持 CUDA 13（需要 580 以上），没有安装任何东西。请换一台运行时（L4）再试。')
print('✅ 环境检查通过，开始安装')
# HF 令牌放进环境变量：之后启动的 ComfyUI 与下载都会沿用
try:
    from google.colab import userdata
    os.environ['HF_TOKEN'] = userdata.get('HF_TOKEN'); print('已读到 HF_TOKEN')
except Exception:
    print('没有设置 HF_TOKEN，之后改用匿名下载（人多时可能被限速）')

def clone(url, dest):
    if not Path(dest).exists():
        subprocess.run(['git', 'clone', '--depth', '1', url, str(dest)], check=True)
    else:
        subprocess.run(['git', '-C', str(dest), 'pull', '--ff-only'], check=False)

def pip(*args):
    subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', *args], check=True)

COMFY = Path('/content/ComfyUI')
clone('https://github.com/Comfy-Org/ComfyUI.git', COMFY)   # JZL 需要 ComfyUI ≥ 0.29（官方 H3 支持），直接用最新版
if not (torch.version.cuda or '').startswith('13'):
    print('安装 CUDA 13 版 PyTorch（int8 ConvRot 权重需要；torchaudio 一起装，版本不匹配会让 JZL 整包加载失败）…')
    pip('torch', 'torchvision', 'torchaudio', '--index-url', 'https://download.pytorch.org/whl/cu130')
pip('-r', str(COMFY / 'requirements.txt'), 'huggingface_hub', 'hf_xet')
# 节点管理器（ComfyUI-Manager 已并入官方，用 pip 包＋启动参数 --enable-manager）
if (COMFY / 'manager_requirements.txt').exists():
    pip('-r', str(COMFY / 'manager_requirements.txt'))
# JZL 节点包＋ VideoHelperSuite（导演台「💾 视频保存分配」的分段输出接 VHS Video Combine 存 mp4）
NODES = {
    'ComfyUI-JZL-MiniMax-H3': 'https://github.com/wjluoxiao/ComfyUI-JZL-MiniMax-H3.git',
    'ComfyUI-VideoHelperSuite': 'https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite.git',
}
for name, url in NODES.items():
    d = COMFY / 'custom_nodes' / name
    clone(url, d)
    if (d / 'requirements.txt').exists():
        pip('-r', str(d / 'requirements.txt'))   # JZL：imageio-ffmpeg（分段落盘／合并用）、Pillow、numpy
JZL = COMFY / 'custom_nodes' / 'ComfyUI-JZL-MiniMax-H3'
LLM_REPO, LLM_FILE, LLM_MMPROJ = """ + repr(LLM_REPO) + ', ' + repr(LLM_FILE) + ', ' + repr(LLM_MMPROJ) + """   # 导演台的本地 LLM（⑤ 下载）
# Colab 内置代理（④）会把网址的 %2F 解开，工作流清单点开会 404 → 装一个补路由的小节点
FIX = COMFY / 'custom_nodes' / 'colab_userdata_slash_fix'
FIX.mkdir(parents=True, exist_ok=True)
(FIX / '__init__.py').write_text(FIX_SRC, encoding='utf-8')
import re
ver = re.search(r'(?m)^version\\s*=\\s*"([^"]+)"', (JZL / 'pyproject.toml').read_text(encoding='utf-8'))
print('JZL 节点包版本：', ver.group(1) if ver else '?')
print('ffmpeg:', shutil.which('ffmpeg') or '没有（会改用 imageio-ffmpeg 自带的）')
out = subprocess.run([sys.executable, '-c', 'import torch, torchaudio;print(torch.__version__, torch.version.cuda, torchaudio.__version__, torch.cuda.is_available())'],
                     capture_output=True, text=True)
print('ComfyUI 会用的 torch / CUDA / torchaudio：', out.stdout.strip(), out.stderr[-300:])
print('✅ 安装完成')
"""),
    code("""#@title ② 本地 LLM 运行时 llama-server（导演台拆解剧本用）
#@markdown JZL 用 `llama-server` 子进程跑本地 LLM（llama.cpp b10436，跑完即停、释放显存）。Linux 没有 CUDA 预编译包：
#@markdown - `cuda`（默认）：从源码编译，只编当前显卡的架构；编好缓存到「我的云端硬盘/H3_models/jzl_llama_runtime」，下次直接还原
#@markdown - `cpu`：下载预编译包即可用，但推理很慢
#@markdown - `skip`：不装；⑧ 的 LLM 改选「在线API」
LLAMA_BACKEND = "cuda"  #@param ["cuda", "cpu", "skip"]
USE_DRIVE_CACHE = True  #@param {type:"boolean"}
import subprocess, sys, os, shutil, json, time, torch
ARCH = '%d%d' % torch.cuda.get_device_capability(0)
RT = JZL / 'runtime' / 'linux-x64' / LLAMA_BACKEND
RT_CFG = JZL / 'runtime_config.json'   # 插件自带的是作者 Windows 机器的配置，安装后会被覆盖

def runtime_ok(verbose=False):
    try:
        c = json.loads(RT_CFG.read_text(encoding='utf-8'))
        if c.get('platform') != 'linux' or c.get('backend') != LLAMA_BACKEND:
            return False
        # 和插件一样把运行时目录「加在前面」：Colab 的 GPU 驱动（libcuda.so）靠原有的 LD_LIBRARY_PATH 才找得到
        libs = ':'.join(str(JZL / d) for d in c.get('library_dirs', []))
        env = dict(os.environ, LD_LIBRARY_PATH=libs + ':' + os.environ.get('LD_LIBRARY_PATH', ''))
        r = subprocess.run([str(JZL / c['executable']), '--version'], env=env, capture_output=True, text=True, timeout=120)
        if r.returncode != 0 and verbose:
            print('llama-server --version 失败：', (r.stdout + r.stderr)[-2000:])
        return r.returncode == 0
    except Exception as e:
        if verbose:
            print('llama-server 自检出错：', e)
        return False

cache = None
if LLAMA_BACKEND != 'skip' and USE_DRIVE_CACHE:
    from google.colab import drive
    drive.mount('/content/drive')
    cache = Path('/content/drive/MyDrive/H3_models/jzl_llama_runtime') / (f'cuda_sm{ARCH}' if LLAMA_BACKEND == 'cuda' else 'cpu')
if LLAMA_BACKEND == 'skip':
    print('跳过本地 LLM 运行时：⑧ 请把 LLM 选成「在线API」')
elif runtime_ok():
    print('llama-server 已安装')
else:
    if cache and (cache / 'runtime_config.json').is_file():
        print('从云端硬盘还原 llama-server …')
        shutil.rmtree(RT, ignore_errors=True)
        shutil.copytree(cache / 'runtime', RT)
        shutil.copy(cache / 'runtime_config.json', RT_CFG)
        for f in RT.rglob('*'):
            if f.is_file():
                f.chmod(0o755)   # 云端硬盘不保留执行权限
    if not runtime_ok():
        args = ['--backend', LLAMA_BACKEND, '--force']
        if LLAMA_BACKEND == 'cuda':
            args += ['--build-from-source', '--jobs', str(os.cpu_count() or 4)]
            print(f'开始编译 CUDA 版 llama-server（只编 sm_{ARCH}），大约 5～15 分钟…')
        # 给插件安装器打两个补丁再运行：
        #  1. 插件默认编 8 种 GPU 架构，Colab 上要很久 → 只编当前这张卡
        #  2. llama-server 内嵌的网页聊天界面：构建时会从 HF bucket 下载资源，资源不全时 llama-ui-embed 校验失败、整个编译中断。
        #     JZL 只用它的 API → 关掉下载（LLAMA_USE_PREBUILT_UI=OFF）并清掉源码里残留的 tools/ui/dist，
        #     llama-ui-embed 拿不到资源目录时会生成空资源表（官方支持的「不带网页界面」构建）
        code = ('import sys, shutil, install_runtime as ir\\n'
                '_orig = ir.cuda_cmake_arguments\\n'
                f'ir.cuda_cmake_arguments = lambda *a, **k: _orig(*a, **k) + ["-DCMAKE_CUDA_ARCHITECTURES={ARCH}", "-DLLAMA_USE_PREBUILT_UI=OFF", "-DLLAMA_BUILD_UI=OFF"]\\n'
                '_prep = ir.prepare_source_checkout\\n'
                'def _prep_no_ui(*a, **k):\\n'
                '    src = _prep(*a, **k)\\n'
                '    shutil.rmtree(src / "tools" / "ui" / "dist", ignore_errors=True)\\n'
                '    return src\\n'
                'ir.prepare_source_checkout = _prep_no_ui\\n'
                f'sys.exit(ir.main({args!r}))')
        import re, sysconfig
        scripts = sysconfig.get_path('scripts')
        if LLAMA_BACKEND == 'cuda':
            # Colab 系统自带 CMake 3.28 偏旧，配 CUDA 13 容易出错 → pip 装一份新版，PATH 里排在系统版前面
            subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'cmake>=3.31.8'], check=True)
        env = dict(os.environ, PATH=f'{scripts}:/usr/local/cuda/bin:' + os.environ.get('PATH', ''))
        print('cmake:', subprocess.run(['cmake', '--version'], env=env, capture_output=True, text=True).stdout.split('\\n')[0])
        # 安装器的输出逐行转进这格（直接 subprocess.run 时 Colab 会吞掉它的报错）；完整日志在 /content/llama_build.log
        t0, last_pct = time.time(), -10
        with open('/content/llama_build.log', 'w') as blog:
            proc = subprocess.Popen([sys.executable, '-u', '-c', code], cwd=str(JZL), env=env, text=True,
                                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT, bufsize=1)
            for line in proc.stdout:
                blog.write(line)
                pct = re.match(r'\\[\\s*(\\d+)%\\]', line)
                if pct and int(pct.group(1)) >= last_pct + 10:   # 编译进度每 10% 显示一次
                    last_pct = int(pct.group(1)); print(f'  编译进度 {last_pct}%（{(time.time() - t0) / 60:.1f} 分钟）', flush=True)
                elif not pct and re.search(r'^\\[(INFO|OK|WARN|ERROR)\\]|error|Error|ERROR', line):
                    print(' ', line.rstrip()[:300], flush=True)
            proc.wait()
        if proc.returncode != 0 or not runtime_ok(verbose=True):
            lines = open('/content/llama_build.log', errors='ignore').readlines()
            print('---- 关键行（CMake 报错／UI 资源）----')
            print(''.join(l for l in lines if re.search(r'CMake Error|UI:|embed:|missing required|error:|fatal', l))[-6000:])
            print('---- llama_build.log 最后 60 行 ----')
            print(''.join(lines[-60:]))
            raise RuntimeError('llama-server 编译／安装失败（原因见上面的日志）；可以先把 ② 改选 skip，⑧ 用在线API')
        print(f'编译／安装完成，{(time.time() - t0) / 60:.1f} 分钟')
# 云端硬盘里还没有这份运行时就备份（包括「本次运行时里已经编好、但还没缓存」的情况）
if LLAMA_BACKEND != 'skip':
    if cache and not (cache / 'runtime_config.json').is_file():
        shutil.rmtree(cache, ignore_errors=True)
        cache.mkdir(parents=True)
        shutil.copytree(RT, cache / 'runtime')
        shutil.copy(RT_CFG, cache / 'runtime_config.json')   # 最后写配置：它在 = 缓存完整
        print('已缓存到云端硬盘，下次直接还原：', cache)
    print('✅ llama-server 已就绪')
"""),
    code("#@title ③ 启动 ComfyUI（后台运行）＋确认 JZL 节点已注册\nDIRECTOR_API = " + repr(json.dumps(DIRECTOR_API, ensure_ascii=False, indent=1)) + """
import subprocess, time, urllib.request, sys, json, os
if 'server' in globals() and server.poll() is None:
    server.terminate(); server.wait(); print('已关闭旧的 ComfyUI，重新启动')
# 节点管理器安全等级：normal- ＝可从清单装节点，也可用 Git 网址安装。
# 这台 Colab 用完即丢；但 HF_TOKEN 在环境变量里，自定义节点读得到 → HF 令牌请用「只读」的。
cfg = COMFY / 'user' / '__manager' / 'config.ini'
cfg.parent.mkdir(parents=True, exist_ok=True)
if not cfg.exists():
    cfg.write_text('[default]\\nsecurity_level = normal-\\n', encoding='utf-8')
# --enable-cors-header：走 cloudflare 通道时浏览器 Origin 与 Host 对不上，不开会 403
# --enable-compress-response-body：节点清单与前端文件压缩后走通道快很多
# PYTHONUNBUFFERED：让 JZL 的日志立刻写进 comfyui.log（⑨ 靠它显示进度）
log = open('/content/comfyui.log', 'w')
server = subprocess.Popen([sys.executable, 'main.py', '--listen', '127.0.0.1', '--port', '8188', '--enable-cors-header', '--enable-manager', '--enable-compress-response-body'],
                          cwd=str(COMFY), stdout=log, stderr=subprocess.STDOUT, env=dict(os.environ, PYTHONUNBUFFERED='1'))
for i in range(180):
    try:
        urllib.request.urlopen('http://127.0.0.1:8188/system_stats', timeout=2)
        print('ComfyUI 已启动'); break
    except Exception:
        time.sleep(2)
else:
    print(open('/content/comfyui.log').read()[-3000:])
    raise RuntimeError('ComfyUI 没有启动成功，看上面的日志')
text = open('/content/comfyui.log', encoding='utf-8', errors='ignore').read()
print('\\n'.join(l for l in text.splitlines() if '[JZL MiniMax H3]' in l or 'IMPORT FAILED' in l) or '（日志里没看到 JZL 加载行）')
OBJECT_INFO = json.loads(urllib.request.urlopen('http://127.0.0.1:8188/object_info').read())
print(f"已注册 {sum(k.startswith('JZL_') for k in OBJECT_INFO)} 个 JZL 节点")
if 'JZL_MiniMaxAssetManagerShortDrama' not in OBJECT_INFO:
    raise RuntimeError('JZL 导演台节点没有注册成功：看上面的 [JZL MiniMax H3] 报告，或跑最下面的 🔍 诊断格')

def combo(node, name):
    \"\"\"从 /object_info 取下拉选项（兼容 V1 [[...], {}] 与 V3 ['COMBO', {'options': [...]}]）\"\"\"
    spec = OBJECT_INFO[node]['input'].get('required', {}).get(name) or OBJECT_INFO[node]['input'].get('optional', {}).get(name)
    return list(spec[0]) if isinstance(spec[0], list) else list(spec[1].get('options', []))

# 完整工作流（API 格式）：故事风格填成插件当前的第一个选项；④ 下载后拖进 ComfyUI 画面即可
_wf = json.loads(DIRECTOR_API)
_wf['10']['inputs']['story_style'] = combo('JZL_MiniMaxAssetManagerShortDrama', 'story_style')[0]
DIRECTOR_API = json.dumps(_wf, ensure_ascii=False, indent=1)
Path('/content/JZL_H3_short_drama_api.json').write_text(DIRECTOR_API, encoding='utf-8')
print('✅ JZL 节点包加载正常')
"""),
    code("""#@title ④ 打开 ComfyUI 界面（推荐：Colab 内置代理，快、稳、网址不外流）
# 执行后下面会出现一个链接；显示成 localhost 也没关系，直接点它就会开新标签页。
# ⚠️ 打开标签页的浏览器必须登录「跟这本 Colab 同一个 Google 账号」；账号不同就改用下一格 ④-b。
# 下面会下载「短剧导演台」完整工作流（API 格式）：拖进 ComfyUI 画面就是接好线、配好本地 LLM 的导演台，
# 按 Ctrl+S 存一份就会出现在左侧工作流清单。所有 JZL 节点在节点库「JZL/MiniMax」分类下。
from google.colab import output, files
output.serve_kernel_port_as_window(8188)
files.download('/content/JZL_H3_short_drama_api.json')
"""),
    code("""#@title ④-b（备用）用 cloudflare 网址开界面：浏览器登录的 Google 账号和 Colab 不同时用这格
# ⚠️ 网址没有密码，拿到的人都能用你的 GPU —— 不要贴到公开的地方。
import subprocess, re, time
if not Path('/content/cloudflared').exists():
    subprocess.run(['wget', '-q', '-O', '/content/cloudflared',
                    'https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64'], check=True)
    subprocess.run(['chmod', '+x', '/content/cloudflared'], check=True)
# 通道的输出写进文件，不要接 PIPE：PIPE 没人读会塞满，通道就卡住，网页停在加载画面
if 'tun' in globals() and tun.poll() is None:
    tun.terminate(); tun.wait()
open('/content/cloudflared.log', 'w').close()
tun = subprocess.Popen(['/content/cloudflared', 'tunnel', '--no-autoupdate', '--url', 'http://127.0.0.1:8188'],
                       stdout=open('/content/cloudflared.log', 'a'), stderr=subprocess.STDOUT)
url = None
for i in range(60):
    time.sleep(1)
    m = re.search(r'https://[a-z0-9-]+\\.trycloudflare\\.com', open('/content/cloudflared.log').read())
    if m:
        url = m.group(0); break
print('👉 ComfyUI 界面：', url or '（没拿到网址，看 /content/cloudflared.log）')
"""),
    code("""#@title ⑤ 下载模型（H3 约 37 GB＋本地 LLM 约 6 GB）
#@markdown 默认启用云端硬盘缓存：已有模型从「我的云端硬盘/H3_models」复制，缺少的从 Hugging Face 下载并备份。首次使用需授权并预留云端硬盘空间。
USE_DRIVE_CACHE = True  #@param {type:"boolean"}
#@markdown 本地 LLM：导演台拆解剧本用（② 选了 skip、只用在线 API 时可以不勾）。
INCLUDE_LLM = True  #@param {type:"boolean"}
#@markdown H3 潜空间放大模型（约 0.2 GB）：导演台「二次采样」与「SelfLift 采样」选放大模型时用。
INCLUDE_LATENT_UPSCALER = True  #@param {type:"boolean"}
from huggingface_hub import hf_hub_download, get_hf_file_metadata, hf_hub_url
import shutil, os
""" + model_cache_src + """
CACHE = None
if USE_DRIVE_CACHE:
    from google.colab import drive
    drive.mount('/content/drive')
    CACHE = Path('/content/drive/MyDrive/H3_models'); CACHE.mkdir(parents=True, exist_ok=True)
files = [
    ('OliviaRossi/Minimax-h3_Singularity-x-10Eros-Max_PRISM', 'Minimax-h3_Singularity_x_10Eros-Max_PRISM_int8.safetensors', 'diffusion_models'),
    ('Comfy-Org/MiniMax-H3', 'text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors', 'text_encoders'),
    ('Comfy-Org/MiniMax-H3', 'vae/minimax_h3_video_vae_int8_convrot.safetensors', 'vae'),
    ('Comfy-Org/MiniMax-H3', 'vae/minimax_h3_audio_vae_fp32.safetensors', 'vae'),
]
if INCLUDE_LATENT_UPSCALER:
    files.append(('dseditor/minimax-h3-studio-assets', 'latent_upscale_models/minimax_h3_latent_upscaler_3d_fp16_V2.safetensors', 'latent_upscale_models'))
for repo, name, folder in files:
    target = COMFY / 'models' / folder / Path(name).name
    target.parent.mkdir(parents=True, exist_ok=True)
    cached = CACHE / Path(name).name if CACHE else None
    ensure_model(repo, name, target, cached)
    print(folder, target.name, f'{target.stat().st_size / 2**30:.1f} GiB')
# GGUF 不是 safetensors，用文件大小核对完整性（和 Hugging Face 上的大小一致才算完整）
if INCLUDE_LLM:
    for name in (LLM_FILE, LLM_MMPROJ):
        target = COMFY / 'models' / 'LLM' / name
        target.parent.mkdir(parents=True, exist_ok=True)
        size = get_hf_file_metadata(hf_hub_url(LLM_REPO, name)).size
        cached = CACHE / 'LLM' / name if CACHE else None
        if not (target.is_file() and target.stat().st_size == size):
            if cached and cached.is_file() and cached.stat().st_size == size:
                print('从云端硬盘还原', name, flush=True); shutil.copy(cached, target)
            else:
                got = Path(hf_hub_download(repo_id=LLM_REPO, filename=name, local_dir='/content/hf'))
                os.replace(got, target)
        if cached and not (cached.is_file() and cached.stat().st_size == size):
            print('备份到云端硬盘', name, flush=True); cached.parent.mkdir(exist_ok=True); shutil.copy(target, cached)
        print('LLM', name, f'{target.stat().st_size / 2**30:.1f} GiB')
print('✅ 模型就绪；ComfyUI 界面里按 R 刷新模型清单')
"""),
    code("""#@title ⑥（可选）上传自己的 JZL 工作流 JSON：放进左侧工作流清单，并检查缺少的节点和模型
from google.colab import files
import json, urllib.request
up = files.upload()
info = json.loads(urllib.request.urlopen('http://127.0.0.1:8188/object_info').read())
have = {p.name for p in (COMFY / 'models').rglob('*') if p.is_file()}
FRONTEND_ONLY = {'Note', 'MarkdownNote', 'Reroute', 'PrimitiveNode'}
wf_dir = COMFY / 'user' / 'default' / 'workflows'
wf_dir.mkdir(parents=True, exist_ok=True)
for i, (fname, data) in enumerate(up.items(), 1):
    wf = json.loads(data.decode('utf-8'))
    if 'nodes' in wf:   # 界面格式（含子图）
        subgraphs = (wf.get('definitions') or {}).get('subgraphs') or []
        sub_ids = {s.get('id') for s in subgraphs}
        nodes = wf['nodes'] + [n for s in subgraphs for n in s.get('nodes', [])]
        types = {n.get('type') for n in nodes} - FRONTEND_ONLY - sub_ids
        values = [v for n in nodes for v in (n.get('widgets_values') or []) if isinstance(v, str)]
        safe = ''.join(c if c.isascii() and (c.isalnum() or c in '._-') else '_' for c in Path(fname).stem)   # 英文文件名，避免代理上的网址编码问题
        safe = (safe if any(c.isalnum() for c in safe) else f'jzl_workflow_{i}') + '.json'
        (wf_dir / safe).write_bytes(data)
        print(f'📂 {fname} → 左侧工作流清单：{safe}（刷新页面后可见）')
    else:               # API 格式：清单只认界面格式，直接拖进画面即可
        types = {n.get('class_type') for n in wf.values() if isinstance(n, dict)}
        values = [v for n in wf.values() if isinstance(n, dict) for v in (n.get('inputs') or {}).values() if isinstance(v, str)]
        print(f'📄 {fname} 是 API 格式：请直接拖进 ComfyUI 画面')
    missing = sorted(t for t in types if t and t not in info)
    print('  ❌ 缺少节点：' + '、'.join(missing) + '\\n     → 在界面「节点管理器 → 安装缺少的节点」安装后，重跑 ③' if missing else '  ✅ 节点都在')
    models = {Path(v.replace('\\\\', '/')).name for v in values if v.lower().endswith(('.safetensors', '.gguf', '.ckpt', '.pth'))}
    lack = sorted(m for m in models if m not in have)
    print('  ❌ 缺少模型：' + '、'.join(lack) + '\\n     → 换成 ⑤ 已下载的同类模型，或自行下载到 models/ 对应文件夹' if lack else '  ✅ 引用的模型都在')
"""),
    code("""#@title ⑦（可选）上传参考素材：角色／场景／道具图片
#@markdown 上传后在下方表格设定类型和名称（文件名以「角色_」「场景_」「道具_」开头会自动带入）；
#@markdown ⑧ 的故事里写到这个名称，LLM 拆解时就会把图当参考（也可以写 `@名称` 强制引用）。不上传就是纯文本生成。可重复执行追加素材。
CLEAR_ASSETS = False  #@param {type:"boolean"}
from google.colab import files
import ipywidgets as W
from IPython.display import display
ASSET_DIR = COMFY / 'input' / 'jzl' / 'colab'
ASSET_DIR.mkdir(parents=True, exist_ok=True)
if CLEAR_ASSETS or 'asset_rows' not in globals():
    asset_rows = []
for fname, data in files.upload().items():
    (ASSET_DIR / fname).write_bytes(data)
    stem = Path(fname).stem
    typ = next((t for t in ('角色', '场景', '道具') if stem.startswith(t)), '角色')
    name = stem[len(typ):].lstrip('_- ') if stem.startswith(typ) else stem
    asset_rows.append(dict(path=f'jzl/colab/{fname}', data=data,
        on=W.Checkbox(value=True, indent=False, layout=W.Layout(width='30px')),
        type=W.Dropdown(options=['角色', '场景', '道具'], value=typ, layout=W.Layout(width='80px')),
        name=W.Text(value=name or stem, placeholder='名称（故事里用这个名字）', layout=W.Layout(width='200px')),
        desc=W.Text(placeholder='详细描述（可选，帮助 LLM 理解这张图）', layout=W.Layout(width='420px'))))
display(W.VBox([W.HBox([W.Image(value=r['data'], width=72), r['on'], r['type'], r['name'], r['desc']]) for r in asset_rows]
               or [W.HTML('（没有素材：纯文本生成）')]))
"""),
    code("#@title ⑧ 故事＋设置（执行后下方出现表单；改完不用重跑，直接执行 ⑨）\n"
         "import ipywidgets as W\n"
         "from IPython.display import display\n"
         "SAMPLE_STORY = " + repr(SAMPLE_STORY) + "\n"
         """N = 'JZL_MiniMaxAssetManagerShortDrama'
def _keep(name, default):   # 重跑这格不会清掉已填的内容
    return globals()[name].value if name in globals() else default
L = W.Layout(width='460px'); S = {'description_width': '120px'}
story_box = W.Textarea(value=_keep('story_box', SAMPLE_STORY), placeholder='用自然语言写故事／剧本（不要贴已拆解的 [SHOT_START] 提示词）',
                       layout=W.Layout(width='100%', height='220px'))
run_box = W.Dropdown(options=combo(N, 'run_mode'), value=_keep('run_box', combo(N, 'run_mode')[0]), description='运行模式', layout=L, style=S)
style_box = W.Dropdown(options=combo(N, 'story_style'), value=_keep('style_box', combo(N, 'story_style')[0]), description='故事风格', layout=L, style=S)
name_box = W.Text(value=_keep('name_box', 'colab_demo'), description='故事名称', layout=L, style=S)
ar_box = W.Dropdown(options=combo(N, 'aspect_ratio'), value=_keep('ar_box', '16:9 (Widescreen)'), description='画幅比例', layout=L, style=S)
mp_box = W.FloatText(value=_keep('mp_box', 0.4), description='百万像素（MP）', layout=L, style=S)
dur_box = W.Dropdown(options=combo(N, 'duration'), value=_keep('dur_box', '固定5秒'), description='每段时长', layout=L, style=S)
cnt_box = W.IntText(value=_keep('cnt_box', 3), description='段数（0=剧情决定）', layout=L, style=S)
scale_box = W.FloatSlider(value=_keep('scale_box', 1.0), min=1.0, max=5.0, step=0.1, description='参考图放大', layout=L, style=S)
lang_box = W.Dropdown(options=['中文 [ZH]', '英文 [EN]'], value=_keep('lang_box', '中文 [ZH]'), description='提示词语言', layout=L, style=S)
enh_box = W.Checkbox(value=_keep('enh_box', False), description='提示词增强（LLM 再润色每段，耗时约翻倍）', indent=False)
steps_box = W.IntText(value=_keep('steps_box', 6), description='采样步数', layout=L, style=S)
seed_box = W.IntText(value=_keep('seed_box', -1), description='seed（-1 随机）', layout=L, style=S)
_local_ok = (COMFY / 'models' / 'LLM' / LLM_FILE).is_file() and globals().get('LLAMA_BACKEND', 'skip') != 'skip'
llm_box = W.ToggleButtons(options=['本地模型 [local]', '在线API [api]'], value=_keep('llm_box', '本地模型 [local]' if _local_ok else '在线API [api]'), description='LLM', style=S)
api_provider = W.Dropdown(options=['OpenAI 兼容 (OpenAI/DeepSeek/Qwen/GLM/Kimi/Ollama/vLLM/LM Studio)', 'Anthropic', 'Google Gemini'],
                          value=_keep('api_provider', 'OpenAI 兼容 (OpenAI/DeepSeek/Qwen/GLM/Kimi/Ollama/vLLM/LM Studio)'), description='API 类型', layout=L, style=S)
api_model = W.Text(value=_keep('api_model', 'deepseek-chat'), description='模型名', layout=L, style=S)
api_base = W.Text(value=_keep('api_base', 'https://api.deepseek.com'), description='Base URL', layout=L, style=S)
api_key = W.Password(value=_keep('api_key', ''), description='API Key', layout=L, style=S)
api_panel = W.VBox([api_provider, api_model, api_base, api_key])
def _toggle(change=None):
    api_panel.layout.display = '' if llm_box.value.startswith('在线') else 'none'
llm_box.observe(_toggle, names='value'); _toggle()
display(W.VBox([W.HTML('<b>故事／剧本</b>'), story_box,
                W.HBox([W.VBox([run_box, style_box, name_box, ar_box, mp_box, dur_box]),
                        W.VBox([cnt_box, scale_box, lang_box, steps_box, seed_box, enh_box])]),
                llm_box, api_panel]))
"""),
    code("""#@title ⑨ 生成：短剧导演台（LLM 拆解 → 逐段生成 → 拼接成片）
import json, random, time, urllib.request
from IPython.display import Video, display
p = json.loads(DIRECTOR_API)
d = p['10']['inputs']
m = json.loads(d['manager_settings'])
story = story_box.value.strip()
assert story, '⑧ 的故事是空的'
local = llm_box.value.startswith('本地')
if local and not (COMFY / 'models' / 'LLM' / LLM_FILE).is_file():
    raise RuntimeError('本地 LLM 模型还没下载：跑 ⑤（勾 INCLUDE_LLM），或在 ⑧ 改用在线API')
if local and globals().get('LLAMA_BACKEND', 'skip') == 'skip':
    raise RuntimeError('② 选了 skip，没有 llama-server：回 ② 选 cuda／cpu 再跑，或在 ⑧ 改用在线API')
if not local and not api_key.value.strip():
    raise RuntimeError('⑧ 选了在线API，但 API Key 是空的')
d.update(run_mode=run_box.value, story_style=style_box.value, story_name=name_box.value.strip() or 'colab_demo',
         aspect_ratio=ar_box.value, megapixels=float(mp_box.value), duration=dur_box.value,
         video_count=max(0, min(60, int(cnt_box.value))), scale_factor=float(scale_box.value), internal_prompt=story)
m['enhance'].update(llm_backend=llm_box.value, enabled=bool(enh_box.value), prompt_lang=lang_box.value)
m['enhance']['api'] = {'provider': api_provider.value, 'model': api_model.value.strip(), 'api_key': api_key.value.strip(),
                       'base_url': api_base.value.strip(), 'temperature': 0.6, 'max_tokens': 81920, 'thinking': 'disabled'}
m['sample_decode'].update(steps=int(steps_box.value), seed=max(0, int(seed_box.value)),
                          seed_mode='fixed' if seed_box.value >= 0 else 'randomize')
m['assets']['images'] = [{'type': r['type'].value, 'name': r['name'].value.strip(), 'description': r['desc'].value.strip(),
                          'path': r['path'], 'enabled': True}
                         for r in globals().get('asset_rows', []) if r['on'].value and r['name'].value.strip()]
d['manager_settings'] = json.dumps(m, ensure_ascii=False)
print('素材：', '、'.join(f"{a['type']}「{a['name']}」" for a in m['assets']['images']) or '无（纯文本生成）')
print('LLM：', f'本地 {LLM_FILE}（llama-server）' if local else f'{api_provider.value.split(" ")[0]} / {api_model.value}')
req = urllib.request.Request('http://127.0.0.1:8188/prompt', data=json.dumps({'prompt': p}).encode(),
                             headers={'Content-Type': 'application/json'})
try:
    pid = json.loads(urllib.request.urlopen(req).read())['prompt_id']
except urllib.error.HTTPError as e:
    raise RuntimeError(e.read().decode()[:3000])
t0 = time.time()
logf = open('/content/comfyui.log', encoding='utf-8', errors='ignore'); logf.seek(0, 2)
print('已送出，以下是导演台的实时日志：')
while True:
    time.sleep(5)
    for line in logf.read().splitlines():
        if '[JZL' in line or 'Traceback' in line or 'Error' in line:
            print(' ', line[:300])
    h = json.loads(urllib.request.urlopen('http://127.0.0.1:8188/history/' + pid).read())
    if pid in h:
        st = h[pid]['status']
        if st.get('status_str') == 'error':
            print(json.dumps(st, ensure_ascii=False)[:3000]); raise RuntimeError('生成失败，看上面的日志或跑 🔍')
        break
vids = sorted((f for f in (COMFY / 'output' / 'jzl').rglob('*.mp4') if f.stat().st_mtime > t0), key=lambda f: f.stat().st_mtime)
if not vids:
    raise RuntimeError('这次没有产出视频：看上面的 [JZL-短剧导演台] 日志（运行模式选了「仅提示词／扩写」时本来就不出片）')
final = next((f for f in reversed(vids) if '合并' in f.name), vids[-1])
print(f'完成，{(time.time() - t0) / 60:.1f} 分钟；本次共 {len(vids)} 个视频，成片：', final)
display(Video(str(final), embed=True, width=720))
from google.colab import files; files.download(str(final))
"""),
    code("""#@title 🎬 查看／下载导演台生成的视频（在 ComfyUI 界面里跑的也在这里）
#@markdown 导演台用 ffmpeg 直接把视频写到 `output/jzl/<故事名>/`，不经过 ComfyUI 的「媒体资产」面板，所以面板里看不到。
SHOW_LATEST = 3  #@param {type:"integer"}
DOWNLOAD = False  #@param {type:"boolean"}
from IPython.display import Video, display
from google.colab import files
vids = sorted((COMFY / 'output' / 'jzl').rglob('*.mp4'), key=lambda f: f.stat().st_mtime, reverse=True)
log = open('/content/comfyui.log', encoding='utf-8', errors='ignore').read()
print('\\n'.join(l for l in log.splitlines() if '[JZL-短剧导演台]' in l and any(k in l for k in ('落盘', '成片', '失败', '跳过', '⚠️')))[-3000:]
      or '（日志里没有导演台的落盘记录）')
if not vids:
    print('还没有任何视频：看上面的日志，或跑 🔍')
for f in vids[:max(1, SHOW_LATEST)]:
    print(f'{f.relative_to(COMFY / "output")}  {f.stat().st_size / 2**20:.1f} MB')
    display(Video(str(f), embed=True, width=640))
    if DOWNLOAD:
        files.download(str(f))
"""),
    code("""#@title 🔍 出问题时跑这格：GPU、JZL 加载报告、环境自检、ComfyUI 日志
import subprocess, sys
print(subprocess.run(['nvidia-smi'], capture_output=True, text=True).stdout)
log = open('/content/comfyui.log', encoding='utf-8', errors='ignore').read()
print('---- JZL 日志 ----')
print('\\n'.join(l for l in log.splitlines() if '[JZL' in l)[-4000:] or '（没有 JZL 日志）')
print('---- JZL 环境自检（ffmpeg / PyAV / 输出目录 / 显卡）----')
r = subprocess.run([sys.executable, str(JZL / 'jzl_diagnose.py')], cwd=str(COMFY), capture_output=True, text=True)
print((r.stdout + r.stderr)[-4000:])
print('---- ComfyUI 日志（错误相关）----')
keys = ('error', 'Error', 'Traceback', 'IMPORT FAILED', 'out of memory', 'Prompt executed')
print('\\n'.join(l for l in log.splitlines() if any(k in l for k in keys))[-4000:])
"""),
]

NOTEBOOKS = {
    'H3_Digital_Human.ipynb': cells,
    'H3_Basic_ComfyUI.ipynb': [HEAD_BASIC, cells[1], C2_BASIC, C3_BASIC, cells[4], cells[5], cells[9]],
    'H3_Toybox.ipynb': TOYBOX,
    'H3_JZL_Nodes.ipynb': JZL,
}
for fname, cs in NOTEBOOKS.items():
    nb = {'cells': cs,
          'metadata': {'accelerator': 'GPU', 'colab': {'provenance': [], 'gpuType': 'L4'},
                       'kernelspec': {'name': 'python3', 'display_name': 'Python 3'}},
          'nbformat': 4, 'nbformat_minor': 0}
    io.open(fname, 'w', encoding='utf-8', newline='\n').write(json.dumps(nb, ensure_ascii=False, indent=1) + '\n')
    # 自檢：每個 code cell 都要能編譯
    for i, c in enumerate(cs):
        if c['cell_type'] == 'code':
            compile(c['source'], f'{fname}:cell{i}', 'exec')
    print('ok', fname, 'cells =', len(cs))
