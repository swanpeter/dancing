import base64
import json
import os
import queue
import threading
import time
from typing import Optional

import tkinter as tk
from tkinter import filedialog, messagebox

import requests
from byteplussdkarkruntime import Ark
from dotenv import load_dotenv

load_dotenv()

SEEDANCE_MODEL_ID = "ep-20251224135439-xlh46"
DEFAULT_EXTRA_PARAMS = "--resolution 480p --duration 12 --camerafixed false"
DEFAULT_SAVE_DIR = "/Users/shunkurokawa/Desktop/16PJ/生成動画"
POLL_INTERVAL = 1
POLL_TIMEOUT = 600

client = Ark(
    base_url="https://ark.ap-southeast.bytepluses.com/api/v3",
    api_key=os.environ.get("ARK_API_KEY"),
)

log_queue = queue.Queue()
root: Optional[tk.Tk] = None
log_text: Optional[tk.Text] = None
save_dir_var: Optional[tk.StringVar] = None


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


def run_generation_job(
    job_label: str,
    first_frame_path: str,
    end_frame_path: str,
    prompt_text: str,
    extra_params: str,
    save_dir: str,
    on_finish,
) -> None:
    try:
        if first_frame_path and not os.path.isfile(first_frame_path):
            raise FileNotFoundError(f"画像ファイルが存在しません: {first_frame_path}")
        if end_frame_path and not os.path.isfile(end_frame_path):
            raise FileNotFoundError(f"画像ファイルが存在しません: {end_frame_path}")

        final_prompt = build_prompt(prompt_text, extra_params)
        if not final_prompt:
            raise ValueError("プロンプトが空です。")

        os.makedirs(save_dir, exist_ok=True)
        first_frame_url = create_data_url(first_frame_path) if first_frame_path else None
        last_frame_url = create_data_url(end_frame_path) if end_frame_path else None

        log_message(f"[{job_label}] タスクを作成中")
        content_payload = [{"type": "text", "text": final_prompt}]
        if first_frame_url:
            content_payload.append(
                {"type": "image_url", "image_url": {"url": first_frame_url}, "role": "first_frame"}
            )
        if last_frame_url:
            content_payload.append(
                {"type": "image_url", "image_url": {"url": last_frame_url}, "role": "last_frame"}
            )

        create_result = client.content_generation.tasks.create(
            model=SEEDANCE_MODEL_ID,
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
        save_prompt_text(video_path, final_prompt, job_label)
        log_message(f"[{job_label}] すべての処理が完了しました")
    except Exception as exc:  # noqa: BLE001
        log_message(f"[{job_label}] エラー: {exc}")
    finally:
        on_finish()


class TaskFrame(tk.LabelFrame):
    def __init__(self, master: tk.Misc) -> None:
        super().__init__(master, text="Seedance1.5")
        self.running = False
        self.image_var = tk.StringVar()
        self.end_image_var = tk.StringVar()
        self.params_var = tk.StringVar(value=DEFAULT_EXTRA_PARAMS)

        self.columnconfigure(1, weight=1)

        tk.Label(self, text="開始画像").grid(row=0, column=0, padx=(0, 6), pady=4, sticky="w")
        tk.Entry(self, textvariable=self.image_var).grid(row=0, column=1, sticky="ew", pady=4)
        tk.Button(self, text="参照", command=self.browse_image).grid(row=0, column=2, padx=(6, 0), pady=4)

        tk.Label(self, text="終了画像 (任意)").grid(row=1, column=0, padx=(0, 6), pady=4, sticky="w")
        tk.Entry(self, textvariable=self.end_image_var).grid(row=1, column=1, sticky="ew", pady=4)
        tk.Button(self, text="参照", command=self.browse_end_image).grid(row=1, column=2, padx=(6, 0), pady=4)

        tk.Label(self, text="プロンプト").grid(row=2, column=0, padx=(0, 6), pady=4, sticky="nw")
        self.prompt_text = tk.Text(self, height=4, wrap="word")
        self.prompt_text.grid(row=2, column=1, columnspan=2, sticky="ew", pady=4)

        tk.Label(self, text="追加パラメータ").grid(row=3, column=0, padx=(0, 6), pady=4, sticky="w")
        tk.Entry(self, textvariable=self.params_var).grid(row=3, column=1, sticky="ew", pady=4)

        self.run_button = tk.Button(self, text="生成", command=self.start_generation)
        self.run_button.grid(row=3, column=2, padx=(6, 0), pady=4, sticky="e")

    def browse_image(self) -> None:
        file_path = filedialog.askopenfilename(
            title="開始画像を選択",
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

    def set_running(self, running: bool) -> None:
        self.running = running
        state = tk.DISABLED if running else tk.NORMAL
        self.run_button.config(state=state)

    def start_generation(self) -> None:
        if save_dir_var is None:
            return
        image_path = self.image_var.get().strip()
        end_image_path = self.end_image_var.get().strip()
        prompt_text = self.prompt_text.get("1.0", tk.END).strip()
        extra_params = self.params_var.get().strip()
        save_dir = save_dir_var.get().strip()

        if not prompt_text:
            messagebox.showerror("入力エラー", "プロンプトを入力してください。")
            return
        if not save_dir:
            messagebox.showerror("入力エラー", "保存先ディレクトリを入力してください。")
            return

        if image_path and not os.path.isfile(image_path):
            messagebox.showerror("ファイル未検出", f"画像ファイルが見つかりません: {image_path}")
            return
        if end_image_path and not os.path.isfile(end_image_path):
            messagebox.showerror("ファイル未検出", f"画像ファイルが見つかりません: {end_image_path}")
            return

        try:
            os.makedirs(save_dir, exist_ok=True)
        except OSError as exc:
            messagebox.showerror("ディレクトリ作成エラー", f"保存先ディレクトリを作成できません: {exc}")
            return

        self.set_running(True)
        log_message("[Seedance1.5] ジョブを開始します。")

        def on_finish() -> None:
            if root:
                root.after(0, lambda: self.set_running(False))
            else:
                self.set_running(False)

        thread = threading.Thread(
            target=run_generation_job,
            args=(
                "Seedance1.5",
                image_path,
                end_image_path,
                prompt_text,
                extra_params,
                save_dir,
                on_finish,
            ),
            daemon=True,
        )
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


def build_gui() -> None:
    global root, log_text, save_dir_var

    root = tk.Tk()
    root.title("Seedance1.5 動画生成")
    root.geometry("900x520")

    save_dir_var = tk.StringVar(value=DEFAULT_SAVE_DIR)

    main_frame = tk.Frame(root)
    main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
    main_frame.columnconfigure(0, weight=1)
    main_frame.rowconfigure(1, weight=1)

    task_frame = TaskFrame(main_frame)
    task_frame.grid(row=0, column=0, sticky="ew", pady=(0, 8))

    output_frame = tk.LabelFrame(main_frame, text="出力設定")
    output_frame.grid(row=1, column=0, sticky="ew")
    output_frame.columnconfigure(1, weight=1)

    tk.Label(output_frame, text="保存先ディレクトリ").grid(row=0, column=0, padx=(0, 6), pady=4, sticky="w")
    save_entry = tk.Entry(output_frame, textvariable=save_dir_var)
    save_entry.grid(row=0, column=1, padx=(0, 6), pady=4, sticky="ew")

    def browse_save_dir() -> None:
        directory = filedialog.askdirectory(title="保存先ディレクトリを選択")
        if directory:
            save_dir_var.set(directory)

    tk.Button(output_frame, text="参照", command=browse_save_dir).grid(row=0, column=2, pady=4)

    log_frame = tk.LabelFrame(main_frame, text="ログ")
    log_frame.grid(row=2, column=0, sticky="nsew", pady=(8, 0))
    log_frame.columnconfigure(0, weight=1)
    log_frame.rowconfigure(0, weight=1)

    log_text = tk.Text(log_frame, height=10, state=tk.DISABLED, wrap="word")
    log_scroll = tk.Scrollbar(log_frame, command=log_text.yview)
    log_text.configure(yscrollcommand=log_scroll.set)
    log_text.grid(row=0, column=0, sticky="nsew")
    log_scroll.grid(row=0, column=1, sticky="ns")

    root.after(200, process_log_queue)


if __name__ == "__main__":
    build_gui()
    if root is not None:
        root.mainloop()
