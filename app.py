import base64
import os
import queue
import shutil
import subprocess
import threading
import time
import tempfile
import json
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
from typing import Dict, List, Optional
from urllib.parse import urlparse
import tkinter as tk
from tkinter import filedialog, messagebox

import requests
from PIL import Image
from byteplussdkarkruntime import Ark
from dotenv import load_dotenv

load_dotenv()

MODEL_ID = "ep-20251014141518-c8xhz" ""
SAMPLE_MODEL_ID = "ep-20251224135439-xlh46"
SEEDANCE_MODELS = {
    "Seedance1.0": MODEL_ID,
    "Seedance1.5": SAMPLE_MODEL_ID,
}
DEFAULT_EXTRA_PARAMS = "--resolution 480p --duration 12 --camerafixed false"
DEFAULT_SAVE_DIR = "/Users/shunkurokawa/Desktop/16PJ/生成動画"
POLL_INTERVAL = 1
POLL_TIMEOUT = 600  # seconds
TASK_FRAME_COUNT = 3

I2I_MODEL_ID = "ep-20251208110124-9jp7r"
I2I_MAX_REFERENCES = 14
I2I_DEFAULT_SAVE_DIR = "/Users/shunkurokawa/Desktop/16PJ/生成動画"
I2I_DEFAULT_PROMPT = "masterpiece, best quality, ultra-detailed, photorealistic, 8k, sharp focus"

client = Ark(
    base_url="https://ark.ap-southeast.bytepluses.com/api/v3",
    api_key=os.environ.get("ARK_API_KEY"),
)

executor = ThreadPoolExecutor(max_workers=3)
log_queue = queue.Queue()
jobs_lock = threading.Lock()
active_jobs = 0

root: Optional[tk.Tk] = None
log_text: Optional[tk.Text] = None
save_dir_var: Optional[tk.StringVar] = None
task_frames: List["TaskFrame"] = []


def serialize_response(obj) -> str:
    if obj is None:
        return "null"
    try:
        if hasattr(obj, "model_dump"):
            payload = obj.model_dump()
        elif hasattr(obj, "to_dict"):
            payload = obj.to_dict()
        elif hasattr(obj, "dict"):
            payload = obj.dict()
        elif isinstance(obj, dict):
            payload = obj
        else:
            payload = getattr(obj, "__dict__", str(obj))
    except Exception:  # noqa: BLE001
        payload = str(obj)
    return json.dumps(payload, ensure_ascii=True, default=str)


def log_response_json(log_func, label: str, obj) -> None:
    log_func(f"{label} JSON: {serialize_response(obj)}")


def run_sample_demo(
    log_func,
    prompt_text: str,
    reference_input: str,
    save_dir: str,
    last_frame_input: str = "",
) -> None:
    """Run a simple content generation example, save the result, and stream logs to the UI."""
    try:
        ref_value = reference_input.strip()
        last_value = last_frame_input.strip()
        text_value = prompt_text.strip()
        if not ref_value:
            raise ValueError("参照画像のURLまたはbase64を入力してください。")
        if not text_value:
            raise ValueError("プロンプトを入力してください。")
        if not save_dir:
            raise ValueError("保存先ディレクトリを入力してください。")

        try:
            os.makedirs(save_dir, exist_ok=True)
        except OSError as exc:
            raise RuntimeError(f"保存先ディレクトリを作成できません: {exc}") from exc

        if os.path.isfile(ref_value):
            image_url_value = create_jpeg_data_url(ref_value)
        elif ref_value.startswith(("http://", "https://", "data:")):
            image_url_value = ref_value
        else:
            image_url_value = f"data:image/png;base64,{ref_value}"

        last_url_value = ""
        if last_value:
            if os.path.isfile(last_value):
                last_url_value = create_jpeg_data_url(last_value)
            elif last_value.startswith(("http://", "https://", "data:")):
                last_url_value = last_value
            else:
                last_url_value = f"data:image/png;base64,{last_value}"

        job_label = "Seedance1.5"
        log_func("リクエスト送信中...")
        content_payload = [
            {"type": "text", "text": text_value},
            {"type": "image_url", "image_url": {"url": image_url_value}, "role": "first_frame"},
        ]
        if last_url_value:
            content_payload.append(
                {"type": "image_url", "image_url": {"url": last_url_value}, "role": "last_frame"}
            )
        create_result = client.content_generation.tasks.create(
            model=SAMPLE_MODEL_ID,
            content=content_payload,
            return_last_frame=True,
        )
        task_id = getattr(create_result, "id", None)
        if not task_id:
            raise RuntimeError("タスクIDを取得できませんでした。")

        log_func(f"タスク作成完了: {task_id}")
        while True:
            get_result = client.content_generation.tasks.get(task_id=task_id)
            status = getattr(get_result, "status", "unknown")
            if status == "succeeded":
                content_obj = getattr(get_result, "content", None)
                video_url = getattr(content_obj, "video_url", None)
                log_func("タスク成功")
                log_response_json(log_func, "最終レスポンス", get_result)
                if not video_url:
                    raise RuntimeError("動画URLを取得できませんでした。")
                log_func(f"動画URL: {video_url}")
                output_path = os.path.join(save_dir, f"{task_id}.mp4")
                download_video(video_url, output_path, job_label)
                last_frame_url = get_last_frame_url(get_result)
                if last_frame_url:
                    last_frame_path = os.path.join(save_dir, f"{task_id}_last_frame.png")
                    download_last_frame_image(last_frame_url, last_frame_path, job_label)
                save_prompt_text(output_path, text_value, job_label)
                break
            if status == "failed":
                log_func(f"タスク失敗: {get_result.error if hasattr(get_result, 'error') else '詳細不明'}")
                log_response_json(log_func, "失敗レスポンス", get_result)
                break
            log_func(f"ステータス: {status} -> 1秒後に再試行")
            time.sleep(1)
    except Exception as exc:  # noqa: BLE001
        log_func(f"エラー: {exc}")


def log_message(message: str) -> None:
    timestamp = time.strftime("%H:%M:%S")
    log_queue.put(f"[{timestamp}] {message}")


def build_prompt(prompt_text: str, extra_params: str) -> str:
    prompt_text = prompt_text.strip()
    extra_params = extra_params.strip()
    if prompt_text and extra_params:
        return f"{prompt_text}\n{extra_params}"
    return prompt_text or extra_params


def create_data_url(image_path: str) -> str:
    ext = os.path.splitext(image_path)[1][1:].lower()
    if not ext:
        raise ValueError("画像ファイルの拡張子を取得できませんでした。")
    with open(image_path, "rb") as img_file:
        base64_str = base64.b64encode(img_file.read()).decode("utf-8")
    return f"data:image/{ext};base64,{base64_str}"


def create_jpeg_data_url(image_path: str) -> str:
    """Encode image to RGB JPEG data URL (aligns with seedance/Seedream flow)."""
    with Image.open(image_path) as img:
        if img.mode != "RGB":
            img = img.convert("RGB")
        buffer = BytesIO()
        img.save(buffer, format="JPEG")
        buffer.seek(0)
    base64_str = base64.b64encode(buffer.read()).decode("utf-8")
    return f"data:image/jpeg;base64,{base64_str}"


def poll_task(task_id: str, job_label: str):
    start_time = time.time()
    while True:
        result = client.content_generation.tasks.get(task_id=task_id)
        status = getattr(result, "status", "unknown")
        if status == "succeeded":
            log_message(f"[{job_label}] タスク完了")
            return result
        if status == "failed":
            error_info = getattr(result, "error", "不明なエラー")
            raise RuntimeError(f"タスクが失敗しました: {error_info}")
        elapsed = time.time() - start_time
        if elapsed > POLL_TIMEOUT:
            raise TimeoutError(f"タスクが{int(elapsed)}秒経過しても完了しませんでした。")
        log_message(f"[{job_label}] ステータス: {status} -> {POLL_INTERVAL}秒後に再試行")
        time.sleep(POLL_INTERVAL)


def download_video(video_url: str, output_path: str, job_label: str) -> None:
    log_message(f"[{job_label}] 動画をダウンロード中: {output_path}")
    response = requests.get(video_url, stream=True, timeout=60)
    response.raise_for_status()
    with open(output_path, "wb") as file:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                file.write(chunk)
    log_message(f"[{job_label}] 動画の保存完了")


def download_last_frame_image(image_url: str, output_path: str, job_label: str) -> None:
    log_message(f"[{job_label}] last frameをダウンロード中: {output_path}")
    response = requests.get(image_url, stream=True, timeout=60)
    response.raise_for_status()
    with open(output_path, "wb") as file:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                file.write(chunk)
    log_message(f"[{job_label}] last frameの保存完了")


def get_last_frame_url(result) -> str:
    content_obj = getattr(result, "content", None)
    if content_obj is None:
        return ""
    return getattr(content_obj, "last_frame_url", "") or ""


def probe_frame_count(video_path: str) -> Optional[int]:
    ffprobe_path = shutil.which("ffprobe")
    if not ffprobe_path:
        return None
    cmd = [
        ffprobe_path,
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=nb_frames",
        "-of",
        "default=nokey=1:noprint_wrappers=1",
        video_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        return None
    value = result.stdout.strip()
    if value.isdigit():
        return int(value)
    return None


def probe_video_bitrate(video_path: str) -> Optional[int]:
    ffprobe_path = shutil.which("ffprobe")
    if not ffprobe_path:
        return None

    cmd = [
        ffprobe_path,
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=bit_rate",
        "-of",
        "default=nokey=1:noprint_wrappers=1",
        video_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        return None
    value = result.stdout.strip()
    if value.isdigit():
        return int(value)
    return None


def extract_last_frame(video_path: str, job_label: str) -> None:
    ffmpeg_path = shutil.which("ffmpeg")
    if not ffmpeg_path:
        log_message(f"[{job_label}] ffmpegが見つからないためスキップ")
        return

    base = os.path.splitext(video_path)[0]
    out_path = base + "_last_frame_tv709.png"
    vf = "scale=in_range=tv:out_range=pc:in_color_matrix=bt709:out_color_matrix=bt709,format=rgb24"

    common = [ffmpeg_path, "-y", "-v", "error"]
    offsets = ("-0.001", "-0.1", "-0.5")

    def try_extract(offset: str) -> bool:
        extract_cmd = common + [
            "-sseof",
            offset,
            "-i",
            video_path,
            "-vf",
            vf,
            "-frames:v",
            "1",
            out_path,
        ]
        r = subprocess.run(extract_cmd, capture_output=True, text=True, check=False)
        ok = r.returncode == 0 and os.path.exists(out_path) and os.path.getsize(out_path) > 0
        if ok:
            log_message(f"[{job_label}] 保存: {out_path} (offset={offset}, tag={vf})")
        else:
            log_message(f"[{job_label}] 抽出失敗: {r.stderr.strip()}")
        return ok

    for off in offsets:
        if try_extract(off):
            break


def save_prompt_text(video_path: str, prompt_text: str, job_label: str) -> None:
    if not prompt_text:
        return
    base, _ = os.path.splitext(video_path)
    prompt_path = base + ".txt"
    try:
        with open(prompt_path, "w", encoding="utf-8") as file:
            file.write(prompt_text)
        log_message(f"[{job_label}] プロンプトを保存: {prompt_path}")
    except OSError as exc:
        log_message(f"[{job_label}] プロンプト保存に失敗: {exc}")


def convert_video_to_60fps(video_path: str, job_label: str, output_dir: Optional[str] = None) -> str:
    ffmpeg_path = shutil.which("ffmpeg")
    if not ffmpeg_path:
        log_message(f"[{job_label}] ffmpegが見つからないため60fps変換をスキップ")
        return video_path

    base_name, ext = os.path.splitext(os.path.basename(video_path))
    target_dir = output_dir if output_dir else os.path.dirname(video_path) or "."
    os.makedirs(target_dir, exist_ok=True)
    output_path = os.path.join(target_dir, f"{base_name}_60fps{ext or '.mp4'}")
    source_bitrate = probe_video_bitrate(video_path)

    cmd = [
        ffmpeg_path,
        "-y",
        "-i",
        video_path,
        "-vf",
        "minterpolate=fps=60:mi_mode=mci:mc_mode=aobmc:me_mode=bidir:vsbmc=1",
        "-vsync",
        "0",
    ]

    if source_bitrate and source_bitrate > 0:
        bitrate_str = str(source_bitrate)
        cmd.extend(
            [
                "-c:v",
                "libx264",
                "-b:v",
                bitrate_str,
                "-maxrate",
                bitrate_str,
                "-bufsize",
                str(source_bitrate * 2),
            ]
        )
    else:
        cmd.extend(["-c:v", "libx264", "-crf", "18", "-preset", "medium"])

    cmd.extend(["-pix_fmt", "yuv420p", "-c:a", "copy", output_path])

    log_message(f"[{job_label}] 60fpsへ変換中: {output_path}")
    try:
        subprocess.run(cmd, check=True)
        log_message(f"[{job_label}] 60fps変換完了")
        return output_path
    except subprocess.CalledProcessError as exc:
        log_message(f"[{job_label}] 60fps変換に失敗: {exc}. 元の動画を使用します。")
        return video_path


def concatenate_videos(video_paths: List[str], output_path: str, job_label: str) -> str:
    ffmpeg_path = shutil.which("ffmpeg")
    if not ffmpeg_path:
        raise FileNotFoundError("ffmpegが見つからないため結合できません。")

    list_file_path = None
    try:
        with tempfile.NamedTemporaryFile("w", delete=False, suffix=".txt") as list_file:
            list_file_path = list_file.name
            for path in video_paths:
                escaped_path = path.replace("'", "'\\''")
                list_file.write(f"file '{escaped_path}'\n")

        cmd = [
            ffmpeg_path,
            "-y",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            list_file_path,
            "-c",
            "copy",
            output_path,
        ]
        log_message(f"[{job_label}] 動画を結合中: {output_path}")
        subprocess.run(cmd, check=True)
        log_message(f"[{job_label}] 動画結合完了")
        return output_path
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(f"動画結合に失敗しました: {exc}") from exc
    finally:
        if list_file_path and os.path.exists(list_file_path):
            try:
                os.remove(list_file_path)
            except OSError:
                pass


def run_generation_job(
    job_label: str,
    model_id: str,
    first_frame_path: str,
    end_frame_path: str,
    prompt_text: str,
    extra_params: str,
    save_dir: str,
    on_finish,
) -> None:
    try:
        if not os.path.isfile(first_frame_path):
            raise FileNotFoundError(f"画像ファイルが存在しません: {first_frame_path}")
        if end_frame_path and not os.path.isfile(end_frame_path):
            raise FileNotFoundError(f"画像ファイルが存在しません: {end_frame_path}")

        final_prompt = build_prompt(prompt_text, extra_params)
        if not final_prompt:
            raise ValueError("プロンプトが空です。")

        os.makedirs(save_dir, exist_ok=True)
        first_frame_url = create_data_url(first_frame_path)
        last_frame_url = create_data_url(end_frame_path) if end_frame_path else None

        log_message(f"[{job_label}] タスクを作成中")
        content_payload = [
            {"type": "text", "text": final_prompt},
            {"type": "image_url", "image_url": {"url": first_frame_url}, "role": "first_frame"},
        ]
        if last_frame_url:
            content_payload.append(
                {"type": "image_url", "image_url": {"url": last_frame_url}, "role": "last_frame"}
            )

        create_result = client.content_generation.tasks.create(
            model=model_id,
            content=content_payload,
            return_last_frame=True,
        )

        task_id = getattr(create_result, "id", None)
        if not task_id:
            raise RuntimeError("タスクIDを取得できませんでした。")

        log_message(f"[{job_label}] タスクID: {task_id}")
        result = poll_task(task_id, job_label)
        log_response_json(log_message, f"[{job_label}] 最終レスポンス", result)

        video_url = getattr(getattr(result, "content", None), "video_url", None)
        if not video_url:
            raise RuntimeError("動画URLを取得できませんでした。")

        video_path = os.path.join(save_dir, f"{task_id}.mp4")
        download_video(video_url, video_path, job_label)
        last_frame_url = get_last_frame_url(result)
        if last_frame_url:
            last_frame_path = os.path.join(save_dir, f"{task_id}_last_frame.png")
            download_last_frame_image(last_frame_url, last_frame_path, job_label)
        # last frame is downloaded from API when available
        save_prompt_text(video_path, final_prompt, job_label)
        log_message(f"[{job_label}] すべての処理が完了しました")
    except Exception as exc:  # noqa: BLE001
        log_message(f"[{job_label}] エラー: {exc}")
    finally:
        on_finish()


def run_concat_job(video_paths: List[str], save_dir: str) -> None:
    job_label = "ジョブA"
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(save_dir, f"concat_{timestamp}.mp4")
    try:
        log_message(f"[{job_label}] 入力動画: {', '.join(os.path.basename(p) for p in video_paths)}")
        result_path = concatenate_videos(video_paths, output_path, job_label)
        log_message(f"[{job_label}] 結合結果: {result_path}")
    except Exception as exc:  # noqa: BLE001
        log_message(f"[{job_label}] エラー: {exc}")


def run_convert_job(video_path: str, save_dir: str) -> None:
    job_label = "60fps変換"
    try:
        log_message(f"[{job_label}] 入力動画: {os.path.basename(video_path)}")
        result_path = convert_video_to_60fps(video_path, job_label, output_dir=save_dir)
        log_message(f"[{job_label}] 変換結果: {result_path}")
    except Exception as exc:  # noqa: BLE001
        log_message(f"[{job_label}] エラー: {exc}")


class TaskFrame(tk.LabelFrame):
    def __init__(self, master: tk.Misc, index: int) -> None:
        super().__init__(master, text=f"ジョブ{index}")
        self.index = index
        self.job_label = f"ジョブ{index}"
        self.running = False

        self.image_var = tk.StringVar()
        self.end_image_var = tk.StringVar()
        self.model_var = tk.StringVar(value="Seedance1.5")
        self.params_var = tk.StringVar(value=DEFAULT_EXTRA_PARAMS)

        self.columnconfigure(1, weight=1)

        tk.Label(self, text="開始画像").grid(row=0, column=0, padx=(0, 6), pady=4, sticky="w")
        image_entry = tk.Entry(self, textvariable=self.image_var)
        image_entry.grid(row=0, column=1, padx=(0, 6), pady=4, sticky="ew")
        tk.Button(self, text="参照", command=self.browse_image).grid(row=0, column=2, pady=4)

        tk.Label(self, text="終了画像").grid(row=1, column=0, padx=(0, 6), pady=4, sticky="w")
        end_image_entry = tk.Entry(self, textvariable=self.end_image_var)
        end_image_entry.grid(row=1, column=1, padx=(0, 6), pady=4, sticky="ew")
        tk.Button(self, text="参照", command=self.browse_end_image).grid(row=1, column=2, pady=4)

        tk.Label(self, text="プロンプト").grid(row=2, column=0, padx=(0, 6), pady=4, sticky="nw")
        self.prompt_text = tk.Text(self, height=4, wrap="word")
        self.prompt_text.grid(row=2, column=1, columnspan=2, pady=4, sticky="ew")

        tk.Label(self, text="モデル").grid(row=3, column=0, padx=(0, 6), pady=(4, 2), sticky="w")
        model_menu = tk.OptionMenu(self, self.model_var, *SEEDANCE_MODELS.keys())
        model_menu.grid(row=3, column=1, padx=(0, 6), pady=(4, 2), sticky="w")

        tk.Label(self, text="追加パラメータ").grid(row=4, column=0, padx=(0, 6), pady=(4, 2), sticky="w")
        params_entry = tk.Entry(self, textvariable=self.params_var)
        params_entry.grid(row=4, column=1, padx=(0, 6), pady=(4, 2), sticky="ew")

        button_frame = tk.Frame(self)
        button_frame.grid(row=5, column=0, columnspan=3, pady=(4, 6), sticky="e")

        self.start_button = tk.Button(button_frame, text="生成", command=self.start_job)
        self.start_button.pack(side=tk.RIGHT, padx=(6, 0))
        self.clear_button = tk.Button(button_frame, text="入力リセット", command=self.clear)
        self.clear_button.pack(side=tk.RIGHT)

    def browse_image(self) -> None:
        file_path = filedialog.askopenfilename(
            title="画像を選択",
            filetypes=[
                ("画像ファイル", "*.png *.jpg *.jpeg *.webp *.bmp"),
                ("すべてのファイル", "*.*"),
            ],
        )
        if file_path:
            self.image_var.set(file_path)

    def browse_end_image(self) -> None:
        file_path = filedialog.askopenfilename(
            title="終了画像を選択",
            filetypes=[
                ("画像ファイル", "*.png *.jpg *.jpeg *.webp *.bmp"),
                ("すべてのファイル", "*.*"),
            ],
        )
        if file_path:
            self.end_image_var.set(file_path)

    def clear(self) -> None:
        if self.running:
            return
        self.image_var.set("")
        self.end_image_var.set("")
        self.prompt_text.delete("1.0", tk.END)
        self.model_var.set("Seedance1.5")
        self.params_var.set(DEFAULT_EXTRA_PARAMS)

    def get_job_data(self) -> Dict[str, str]:
        prompt = self.prompt_text.get("1.0", tk.END).strip()
        return {
            "label": self.job_label,
            "image": self.image_var.get().strip(),
            "end_image": self.end_image_var.get().strip(),
            "prompt": prompt,
            "model": self.model_var.get().strip(),
            "params": self.params_var.get().strip(),
        }

    def set_running(self, running: bool) -> None:
        self.running = running
        state = tk.DISABLED if running else tk.NORMAL
        self.start_button.config(state=state)
        self.clear_button.config(state=state)

    def start_job(self) -> None:
        if self.running:
            return
        start_generation_for_frame(self)


class ConvertFrame(tk.LabelFrame):
    def __init__(self, master: tk.Misc, title: str = "60fps変換") -> None:
        super().__init__(master, text=title)
        self.video_var = tk.StringVar()

        self.columnconfigure(1, weight=1)

        tk.Label(self, text="動画").grid(row=0, column=0, padx=(0, 6), pady=4, sticky="w")
        video_entry = tk.Entry(self, textvariable=self.video_var)
        video_entry.grid(row=0, column=1, padx=(0, 6), pady=4, sticky="ew")
        tk.Button(self, text="参照", command=self.browse_video).grid(row=0, column=2, pady=4)

        button_frame = tk.Frame(self)
        button_frame.grid(row=1, column=0, columnspan=3, pady=(4, 2), sticky="e")
        tk.Button(button_frame, text="60fps変換", command=self.start_convert).pack(side=tk.RIGHT, padx=(6, 0))
        tk.Button(button_frame, text="クリア", command=self.clear).pack(side=tk.RIGHT)

    def browse_video(self) -> None:
        file_path = filedialog.askopenfilename(
            title="動画を選択",
            filetypes=[
                ("動画ファイル", "*.mp4 *.mov *.m4v *.mkv"),
                ("すべてのファイル", "*.*"),
            ],
        )
        if file_path:
            self.video_var.set(file_path)

    def clear(self) -> None:
        self.video_var.set("")

    def start_convert(self) -> None:
        start_convert_job(self.video_var.get())


class ConcatFrame(tk.LabelFrame):
    def __init__(self, master: tk.Misc, title: str = "動画結合") -> None:
        super().__init__(master, text=title)
        self.video1_var = tk.StringVar()
        self.video2_var = tk.StringVar()

        self.columnconfigure(1, weight=1)

        tk.Label(self, text="動画1").grid(row=0, column=0, padx=(0, 6), pady=4, sticky="w")
        video1_entry = tk.Entry(self, textvariable=self.video1_var)
        video1_entry.grid(row=0, column=1, padx=(0, 6), pady=4, sticky="ew")
        tk.Button(self, text="参照", command=lambda: self.browse_video(self.video1_var)).grid(row=0, column=2, pady=4)

        tk.Label(self, text="動画2").grid(row=1, column=0, padx=(0, 6), pady=4, sticky="w")
        video2_entry = tk.Entry(self, textvariable=self.video2_var)
        video2_entry.grid(row=1, column=1, padx=(0, 6), pady=4, sticky="ew")
        tk.Button(self, text="参照", command=lambda: self.browse_video(self.video2_var)).grid(row=1, column=2, pady=4)

        button_frame = tk.Frame(self)
        button_frame.grid(row=2, column=0, columnspan=3, pady=(4, 2), sticky="e")
        tk.Button(button_frame, text="結合", command=self.start_concat).pack(side=tk.RIGHT, padx=(6, 0))
        tk.Button(button_frame, text="クリア", command=self.clear).pack(side=tk.RIGHT)

    def browse_video(self, target_var: tk.StringVar) -> None:
        file_path = filedialog.askopenfilename(
            title="動画を選択",
            filetypes=[
                ("動画ファイル", "*.mp4 *.mov *.m4v *.mkv"),
                ("すべてのファイル", "*.*"),
            ],
        )
        if file_path:
            target_var.set(file_path)

    def clear(self) -> None:
        self.video1_var.set("")
        self.video2_var.set("")

    def start_concat(self) -> None:
        start_concat_job(self.video1_var.get(), self.video2_var.get())


class SeedreamI2IFrame(tk.LabelFrame):
    def __init__(self, master: tk.Misc, root_ref: tk.Misc, client: Ark) -> None:
        super().__init__(master, text="Seedream 4.5 I2I (最大14枚参照)")
        self.root = root_ref
        self.client = client

        self.image_paths: List[str] = []
        self.log_queue: queue.Queue = queue.Queue()
        self.save_dir_var = tk.StringVar(value=I2I_DEFAULT_SAVE_DIR)
        self.prompt_text = tk.Text(self, height=8, wrap="word")
        self.image_list_text: tk.Text
        self.generate_button: tk.Button
        self.log_text: tk.Text

        self.columnconfigure(0, weight=1)
        self.rowconfigure(1, weight=1)  # 画像リスト
        self.rowconfigure(3, weight=1)  # プロンプト
        self.rowconfigure(6, weight=1)  # ログ

        tk.Label(self, text=f"ローカル画像 (最大{I2I_MAX_REFERENCES}枚)").grid(row=0, column=0, sticky="w")

        image_frame = tk.Frame(self)
        image_frame.grid(row=1, column=0, sticky="nsew", pady=(4, 8))
        image_frame.columnconfigure(0, weight=1)
        image_frame.rowconfigure(0, weight=1)

        self.image_list_text = tk.Text(image_frame, height=4, wrap="word", state="disabled")
        self.image_list_text.grid(row=0, column=0, sticky="nsew")
        image_scroll = tk.Scrollbar(image_frame, command=self.image_list_text.yview)
        image_scroll.grid(row=0, column=1, sticky="ns")
        self.image_list_text.configure(yscrollcommand=image_scroll.set)

        button_column = tk.Frame(image_frame)
        button_column.grid(row=0, column=2, sticky="nsw", padx=(6, 0))
        tk.Button(button_column, text="参照(複数可)", command=self.browse_images).pack(fill="x", pady=(0, 6))
        tk.Button(button_column, text="クリア", command=self.clear_images).pack(fill="x")

        tk.Label(self, text="プロンプト").grid(row=2, column=0, sticky="w")
        self.prompt_text.grid(row=3, column=0, sticky="nsew", pady=(4, 8))
        self.prompt_text.insert("1.0", I2I_DEFAULT_PROMPT)

        save_frame = tk.Frame(self)
        save_frame.grid(row=4, column=0, sticky="ew", pady=(0, 6))
        save_frame.columnconfigure(1, weight=1)
        tk.Label(save_frame, text="保存先").grid(row=0, column=0, sticky="w", padx=(0, 6))
        tk.Entry(save_frame, textvariable=self.save_dir_var).grid(row=0, column=1, sticky="ew", padx=(0, 6))
        tk.Button(save_frame, text="参照", command=self.browse_save_dir).grid(row=0, column=2)

        button_frame = tk.Frame(self)
        button_frame.grid(row=5, column=0, sticky="e", pady=(0, 6))
        self.generate_button = tk.Button(button_frame, text="生成", width=12, command=self.start_generation)
        self.generate_button.pack(side=tk.RIGHT)

        log_frame = tk.LabelFrame(self, text="Seedreamログ")
        log_frame.grid(row=6, column=0, sticky="nsew")
        log_frame.rowconfigure(0, weight=1)
        log_frame.columnconfigure(0, weight=1)
        self.log_text = tk.Text(log_frame, state=tk.DISABLED, wrap="word", height=6)
        log_scroll = tk.Scrollbar(log_frame, command=self.log_text.yview)
        self.log_text.configure(yscrollcommand=log_scroll.set)
        self.log_text.grid(row=0, column=0, sticky="nsew")
        log_scroll.grid(row=0, column=1, sticky="ns")

        self.start_log_loop()

    def start_log_loop(self) -> None:
        if self.root is None:
            return
        self.process_log_queue()

    def log_message(self, message: str) -> None:
        timestamp = time.strftime("%H:%M:%S")
        self.log_queue.put(f"[{timestamp}] {message}")

    def process_log_queue(self) -> None:
        if self.root is None:
            return
        try:
            while True:
                msg = self.log_queue.get_nowait()
                self.log_text.configure(state="normal")
                self.log_text.insert(tk.END, msg + "\n")
                self.log_text.see(tk.END)
                self.log_text.configure(state=tk.DISABLED)
        except queue.Empty:
            pass
        self.root.after(200, self.process_log_queue)

    def browse_save_dir(self) -> None:
        directory = filedialog.askdirectory(title="保存先ディレクトリを選択")
        if directory:
            self.save_dir_var.set(directory)

    def create_data_url(self, image_path: str) -> str:
        with Image.open(image_path) as img:
            if img.mode != "RGB":
                img = img.convert("RGB")
            buffer = BytesIO()
            img.save(buffer, format="JPEG")
            buffer.seek(0)
        base64_str = base64.b64encode(buffer.read()).decode("utf-8")
        return f"data:image/jpeg;base64,{base64_str}"

    def derive_filename(self, image_url: str) -> str:
        parsed = urlparse(image_url)
        base = os.path.basename(parsed.path) or "image.jpeg"
        if "." not in base:
            base = f"{base}.jpeg"
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        return f"{timestamp}_{base}"

    def download_image(self, image_url: str, save_dir: str) -> str:
        os.makedirs(save_dir, exist_ok=True)
        filename = self.derive_filename(image_url)
        output_path = os.path.join(save_dir, filename)
        self.log_message(f"画像をダウンロード: {image_url}")
        response = requests.get(image_url, stream=True, timeout=60)
        response.raise_for_status()
        with open(output_path, "wb") as file:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    file.write(chunk)
        return output_path

    def show_error(self, message: str) -> None:
        if self.root:
            self.root.after(0, lambda: messagebox.showerror("エラー", message))
        else:
            messagebox.showerror("エラー", message)

    def update_image_list_display(self) -> None:
        self.image_list_text.configure(state="normal")
        self.image_list_text.delete("1.0", tk.END)
        if self.image_paths:
            for idx, path in enumerate(self.image_paths, start=1):
                self.image_list_text.insert(tk.END, f"{idx}. {path}\n")
        self.image_list_text.configure(state="disabled")

    def set_image_paths(self, paths: List[str]) -> None:
        unique_paths = list(dict.fromkeys(paths))
        if len(unique_paths) > I2I_MAX_REFERENCES:
            messagebox.showwarning(
                "上限超過",
                f"参照画像は最大{I2I_MAX_REFERENCES}枚までです。先頭{I2I_MAX_REFERENCES}枚のみ使用します。",
            )
        self.image_paths = unique_paths[:I2I_MAX_REFERENCES]
        self.update_image_list_display()

    def browse_images(self) -> None:
        paths = filedialog.askopenfilenames(
            title="参照する画像を選択 (最大14枚)",
            filetypes=[
                ("画像ファイル", "*.png *.jpg *.jpeg *.webp *.bmp"),
                ("すべてのファイル", "*.*"),
            ],
        )
        if paths:
            self.set_image_paths(list(paths))

    def clear_images(self) -> None:
        self.set_image_paths([])

    def run_generation(self, image_path_list: List[str], prompt: str, save_dir: str, on_finish) -> None:
        try:
            self.log_message("BytePlusへリクエスト送信中...")
            request_args = {
                "model": I2I_MODEL_ID,
                "prompt": prompt,
                "sequential_image_generation": "disabled",
                "response_format": "url",
                "size": "2K",
                "stream": False,
                "watermark": False,
            }
            if image_path_list:
                self.log_message(f"{len(image_path_list)}枚の参照画像をエンコード中...")
                data_urls = [self.create_data_url(path) for path in image_path_list]
                request_args["image"] = data_urls
            response = self.client.images.generate(**request_args)

            first_item = response.data[0] if getattr(response, "data", None) else None
            image_url = getattr(first_item, "url", None)
            if not image_url:
                raise RuntimeError("レスポンスから画像URLを取得できませんでした。")

            self.log_message(f"生成完了: {image_url}")
            saved_path = self.download_image(image_url, save_dir)
            self.log_message(f"画像を保存しました: {saved_path}")
        except Exception as exc:  # noqa: BLE001
            self.log_message(f"エラー: {exc}")
            self.show_error(str(exc))
        finally:
            on_finish()

    def set_ui_state(self, enabled: bool) -> None:
        state = tk.NORMAL if enabled else tk.DISABLED
        self.generate_button.config(state=state)

    def start_generation(self) -> None:
        prompt = self.prompt_text.get("1.0", tk.END).strip()
        save_dir = self.save_dir_var.get().strip()
        selected_paths = list(self.image_paths)

        if selected_paths:
            for path in selected_paths:
                if not os.path.isfile(path):
                    messagebox.showerror("ファイル未検出", f"画像ファイルが見つかりません: {path}")
                    return
        if not prompt:
            messagebox.showerror("入力エラー", "プロンプトを入力してください。")
            return
        if not save_dir:
            messagebox.showerror("入力エラー", "保存先ディレクトリを入力してください。")
            return

        self.set_ui_state(False)
        self.log_message("生成を開始します...")

        def on_finish() -> None:
            if self.root:
                self.root.after(0, lambda: self.set_ui_state(True))

        thread = threading.Thread(
            target=self.run_generation,
            args=(selected_paths, prompt, save_dir, on_finish),
            daemon=True,
        )
        thread.start()


class SampleDemoFrame(tk.LabelFrame):
    def __init__(self, master: tk.Misc, root_ref: tk.Misc) -> None:
        super().__init__(master, text="単発I2V実行")
        self.root = root_ref
        self.log_queue: queue.Queue = queue.Queue()
        initial_save = ""
        if save_dir_var is not None:
            try:
                initial_save = save_dir_var.get().strip()
            except Exception:
                initial_save = ""
        if not initial_save:
            initial_save = DEFAULT_SAVE_DIR
        self.save_dir_var = tk.StringVar(value=initial_save)

        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=0)
        self.rowconfigure(1, weight=1)
        self.rowconfigure(2, weight=0)
        self.rowconfigure(3, weight=1)
        self.rowconfigure(4, weight=0)
        self.rowconfigure(5, weight=1)
        self.rowconfigure(6, weight=0)
        self.rowconfigure(7, weight=0)
        self.rowconfigure(8, weight=1)

        ref_header = tk.Frame(self)
        ref_header.grid(row=0, column=0, sticky="ew")
        ref_header.columnconfigure(0, weight=1)
        tk.Label(ref_header, text="参照画像 URL または base64").grid(row=0, column=0, sticky="w")
        tk.Button(ref_header, text="画像参照", command=self.browse_reference).grid(row=0, column=1, padx=(6, 0))

        self.ref_text = tk.Text(self, height=1, wrap="none")
        self.ref_text.grid(row=1, column=0, sticky="nsew", pady=(2, 6))

        last_header = tk.Frame(self)
        last_header.grid(row=2, column=0, sticky="ew")
        last_header.columnconfigure(0, weight=1)
        tk.Label(last_header, text="Last frame (任意) URL または base64").grid(row=0, column=0, sticky="w")
        tk.Button(last_header, text="画像参照", command=self.browse_last_frame).grid(row=0, column=1, padx=(6, 0))

        self.last_frame_text = tk.Text(self, height=1, wrap="none")
        self.last_frame_text.grid(row=3, column=0, sticky="nsew", pady=(2, 6))

        tk.Label(self, text="プロンプト").grid(row=4, column=0, sticky="w")
        self.prompt_text = tk.Text(self, height=4, wrap="word")
        self.prompt_text.grid(row=5, column=0, sticky="nsew", pady=(2, 6))
        self.prompt_text.insert("1.0", DEFAULT_EXTRA_PARAMS)

        save_frame = tk.Frame(self)
        save_frame.grid(row=6, column=0, sticky="ew", pady=(0, 6))
        save_frame.columnconfigure(1, weight=1)
        tk.Label(save_frame, text="保存先").grid(row=0, column=0, sticky="w", padx=(0, 6))
        tk.Entry(save_frame, textvariable=self.save_dir_var).grid(row=0, column=1, sticky="ew", padx=(0, 6))
        tk.Button(save_frame, text="参照", command=self.browse_save_dir).grid(row=0, column=2)

        button_frame = tk.Frame(self)
        button_frame.grid(row=7, column=0, sticky="e", pady=(0, 6))
        self.run_button = tk.Button(button_frame, text="実行", width=12, command=self.start_demo)
        self.run_button.pack(side=tk.RIGHT)

        log_frame = tk.Frame(self)
        log_frame.grid(row=8, column=0, sticky="nsew")
        log_frame.columnconfigure(0, weight=1)
        log_frame.rowconfigure(0, weight=1)

        self.log_text = tk.Text(log_frame, state=tk.DISABLED, wrap="word", height=8)
        log_scroll = tk.Scrollbar(log_frame, command=self.log_text.yview)
        self.log_text.configure(yscrollcommand=log_scroll.set)
        self.log_text.grid(row=0, column=0, sticky="nsew")
        log_scroll.grid(row=0, column=1, sticky="ns")

        self.start_log_loop()

    def start_log_loop(self) -> None:
        if self.root is None:
            return
        self.process_log_queue()

    def log_message(self, message: str) -> None:
        timestamp = time.strftime("%H:%M:%S")
        self.log_queue.put(f"[{timestamp}] {message}")

    def process_log_queue(self) -> None:
        if self.root is None:
            return
        try:
            while True:
                msg = self.log_queue.get_nowait()
                self.log_text.configure(state="normal")
                self.log_text.insert(tk.END, msg + "\n")
                self.log_text.see(tk.END)
                self.log_text.configure(state=tk.DISABLED)
        except queue.Empty:
            pass
        self.root.after(200, self.process_log_queue)

    def set_button_state(self, enabled: bool) -> None:
        state = tk.NORMAL if enabled else tk.DISABLED
        self.run_button.config(state=state)

    def get_reference_input(self) -> str:
        return self.ref_text.get("1.0", tk.END).strip()

    def get_prompt_input(self) -> str:
        return self.prompt_text.get("1.0", tk.END).strip()

    def get_last_frame_input(self) -> str:
        return self.last_frame_text.get("1.0", tk.END).strip()

    def browse_save_dir(self) -> None:
        directory = filedialog.askdirectory(title="保存先ディレクトリを選択")
        if directory:
            self.save_dir_var.set(directory)

    def browse_reference(self) -> None:
        file_path = filedialog.askopenfilename(
            title="参照画像を選択",
            filetypes=[
                ("画像ファイル", "*.png *.jpg *.jpeg *.webp *.bmp"),
                ("すべてのファイル", "*.*"),
            ],
        )
        if not file_path:
            return
        self.ref_text.delete("1.0", tk.END)
        self.ref_text.insert("1.0", file_path)

    def browse_last_frame(self) -> None:
        file_path = filedialog.askopenfilename(
            title="Last frame画像を選択",
            filetypes=[
                ("画像ファイル", "*.png *.jpg *.jpeg *.webp *.bmp"),
                ("すべてのファイル", "*.*"),
            ],
        )
        if not file_path:
            return
        self.last_frame_text.delete("1.0", tk.END)
        self.last_frame_text.insert("1.0", file_path)

    def start_demo(self) -> None:
        self.set_button_state(False)
        ref_input = self.get_reference_input()
        prompt_input = self.get_prompt_input()
        last_frame_input = self.get_last_frame_input()
        save_dir_value = self.save_dir_var.get().strip()
        # fall back to global save_dir_var if local is空
        if not save_dir_value and save_dir_var:
            try:
                save_dir_value = save_dir_var.get().strip()
            except Exception:
                save_dir_value = ""
        if not ref_input:
            messagebox.showerror("入力エラー", "参照画像のURLまたはbase64を入力してください。")
            self.set_button_state(True)
            return
        if not prompt_input:
            messagebox.showerror("入力エラー", "プロンプトを入力してください。")
            self.set_button_state(True)
            return
        if not save_dir_value:
            messagebox.showerror("出力先未設定", "保存先ディレクトリを入力してください。")
            self.set_button_state(True)
            return

        self.log_message("ジョブを開始します...")

        def target() -> None:
            try:
                run_sample_demo(
                    self.log_message,
                    prompt_input,
                    ref_input,
                    save_dir_value,
                    last_frame_input,
                )
            finally:
                if self.root:
                    self.root.after(0, lambda: self.set_button_state(True))
                else:
                    self.set_button_state(True)

        thread = threading.Thread(target=target, daemon=True)
        thread.start()


def process_log_queue() -> None:
    if log_text is None or root is None:
        return
    while True:
        try:
            message = log_queue.get_nowait()
        except queue.Empty:
            break
        log_text.configure(state="normal")
        log_text.insert(tk.END, message + "\n")
        log_text.see(tk.END)
        log_text.configure(state="disabled")
    root.after(200, process_log_queue)


def job_finished(frame: Optional["TaskFrame"]) -> None:
    global active_jobs
    with jobs_lock:
        active_jobs = max(0, active_jobs - 1)
    if root and frame:
        root.after(0, lambda: frame.set_running(False))


def start_concat_job(video1_path: str, video2_path: str) -> None:
    if save_dir_var is None:
        return

    video1_path = video1_path.strip()
    video2_path = video2_path.strip()
    if not video1_path or not video2_path:
        messagebox.showerror("入力エラー", "結合する動画を2つ指定してください。")
        return

    missing = next((path for path in (video1_path, video2_path) if not os.path.isfile(path)), None)
    if missing:
        messagebox.showerror("ファイル未検出", f"動画ファイルが見つかりません: {missing}")
        return

    save_dir = save_dir_var.get().strip()
    if not save_dir:
        messagebox.showerror("出力先未設定", "保存先ディレクトリを入力してください。")
        return

    try:
        os.makedirs(save_dir, exist_ok=True)
    except OSError as exc:
        messagebox.showerror("ディレクトリ作成エラー", f"保存先ディレクトリを作成できません: {exc}")
        return

    log_message("[ジョブA] ジョブを開始します。")
    executor.submit(run_concat_job, [video1_path, video2_path], save_dir)


def start_convert_job(video_path: str) -> None:
    if save_dir_var is None:
        return

    video_path = video_path.strip()
    if not video_path:
        messagebox.showerror("入力エラー", "変換する動画を指定してください。")
        return
    if not os.path.isfile(video_path):
        messagebox.showerror("ファイル未検出", f"動画ファイルが見つかりません: {video_path}")
        return

    save_dir = save_dir_var.get().strip()
    if not save_dir:
        messagebox.showerror("出力先未設定", "保存先ディレクトリを入力してください。")
        return

    try:
        os.makedirs(save_dir, exist_ok=True)
    except OSError as exc:
        messagebox.showerror("ディレクトリ作成エラー", f"保存先ディレクトリを作成できません: {exc}")
        return

    log_message("[60fps変換] ジョブを開始します。")
    executor.submit(run_convert_job, video_path, save_dir)


def start_generation_for_frame(frame: "TaskFrame") -> None:
    global active_jobs
    if save_dir_var is None:
        return

    save_dir = save_dir_var.get().strip()
    if not save_dir:
        messagebox.showerror("出力先未設定", "保存先ディレクトリを入力してください。")
        return

    data = frame.get_job_data()
    has_image = bool(data["image"])
    has_prompt = bool(data["prompt"])
    if not (has_image and has_prompt):
        messagebox.showerror(
            "入力エラー",
            f"{data['label']} の画像またはプロンプトが未入力です。両方を入力してください。",
        )
        return

    try:
        os.makedirs(save_dir, exist_ok=True)
    except OSError as exc:
        messagebox.showerror("ディレクトリ作成エラー", f"保存先ディレクトリを作成できません: {exc}")
        return

    frame.set_running(True)
    model_label = data.get("model") or "Seedance1.5"
    log_message(f"[{data['label']}] ジョブを開始します。モデル: {model_label}")

    with jobs_lock:
        active_jobs += 1

    model_id = SEEDANCE_MODELS.get(model_label, MODEL_ID)
    executor.submit(
        run_generation_job,
        data["label"],
        model_id,
        data["image"],
        data["end_image"],
        data["prompt"],
        data["params"],
        save_dir,
        lambda frm=frame: job_finished(frm),
    )


def on_close() -> None:
    if root is None:
        return
    if messagebox.askokcancel("終了確認", "アプリケーションを終了しますか？"):
        try:
            executor.shutdown(wait=False, cancel_futures=False)
        except TypeError:
            executor.shutdown(wait=False)
        root.destroy()


def build_gui() -> None:
    global root, log_text, save_dir_var, task_frames

    root = tk.Tk()
    root.title("BytePlus I2V + Seedream I2I ツール")
    root.geometry("1500x780")

    save_dir_var = tk.StringVar(value=DEFAULT_SAVE_DIR)
    task_frames = []

    main_frame = tk.Frame(root)
    main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
    for col in range(3):
        main_frame.columnconfigure(col, weight=1, uniform="col")
    main_frame.rowconfigure(0, weight=1)

    left_frame = tk.Frame(main_frame)
    left_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 8))

    middle_frame = tk.Frame(main_frame)
    middle_frame.grid(row=0, column=1, sticky="nsew", padx=(0, 8))

    right_frame = tk.Frame(main_frame)
    right_frame.grid(row=0, column=2, sticky="nsew")

    for idx in range(1, TASK_FRAME_COUNT + 1):
        frame = TaskFrame(left_frame, idx)
        frame.pack(fill=tk.X, expand=True, padx=4, pady=4)
        task_frames.append(frame)

    output_frame = tk.LabelFrame(left_frame, text="出力設定")
    output_frame.pack(fill=tk.X, expand=True, padx=4, pady=6)
    output_frame.columnconfigure(1, weight=1)

    tk.Label(output_frame, text="保存先ディレクトリ").grid(row=0, column=0, padx=(0, 6), pady=4, sticky="w")
    save_entry = tk.Entry(output_frame, textvariable=save_dir_var)
    save_entry.grid(row=0, column=1, padx=(0, 6), pady=4, sticky="ew")

    def browse_save_dir() -> None:
        directory = filedialog.askdirectory(title="保存先ディレクトリを選択")
        if directory:
            save_dir_var.set(directory)

    tk.Button(output_frame, text="参照", command=browse_save_dir).grid(row=0, column=2, pady=4)

    middle_frame.columnconfigure(0, weight=1)
    middle_frame.rowconfigure(2, weight=1)

    convert_frame = ConvertFrame(middle_frame, title="60fps変換")
    convert_frame.grid(row=0, column=0, sticky="ew", pady=(0, 8))

    concat_frame = ConcatFrame(middle_frame, title="ジョブA (動画結合)")
    concat_frame.grid(row=1, column=0, sticky="ew", pady=(0, 8))

    sample_frame = SampleDemoFrame(middle_frame, root)
    sample_frame.grid(row=2, column=0, sticky="nsew", pady=(0, 0))

    right_frame.columnconfigure(0, weight=1)
    right_frame.rowconfigure(0, weight=2)
    right_frame.rowconfigure(1, weight=1)

    seedream_frame = SeedreamI2IFrame(right_frame, root, client)
    seedream_frame.grid(row=0, column=0, sticky="nsew", pady=(0, 8))

    log_frame = tk.LabelFrame(right_frame, text="ログ")
    log_frame.grid(row=1, column=0, sticky="nsew")
    log_frame.columnconfigure(0, weight=1)
    log_frame.rowconfigure(0, weight=1)

    log_text = tk.Text(log_frame, height=12, state=tk.DISABLED, wrap="word")
    log_scroll = tk.Scrollbar(log_frame, command=log_text.yview)
    log_text.configure(yscrollcommand=log_scroll.set)
    log_text.grid(row=0, column=0, sticky="nsew")
    log_scroll.grid(row=0, column=1, sticky="ns")

    root.protocol("WM_DELETE_WINDOW", on_close)
    root.after(200, process_log_queue)


if __name__ == "__main__":
    build_gui()
    if root is not None:
        root.mainloop()
