import base64
import json
import os
import time
import tempfile
from typing import Callable, Optional, Tuple

import requests
import streamlit as st
from byteplussdkarkruntime import Ark
from dotenv import load_dotenv

from streamlit_auth_history_utils import (
    get_secret_value,
    init_history,
    persist_history_to_storage,
    require_basic_login,
    sync_cookie_controller,
)

load_dotenv()

SEEDANCE_MODEL_ID = (
    get_secret_value("SEEDANCE_MODEL_ID")
    or os.environ.get("SEEDANCE_MODEL_ID")
    or "ep-20251224135439-xlh46"
)
BASE_EXTRA_PARAMS = "masterpiece, best quality, ultra-detailed, photorealistic, 8k, sharp focus,"
DEFAULT_SAVE_DIR = os.path.join(tempfile.gettempdir(), "seedance_outputs")
POLL_INTERVAL = 1
POLL_TIMEOUT = 600

client = Ark(
    base_url="https://ark.ap-southeast.bytepluses.com/api/v3",
    api_key=get_secret_value("ARK_API_KEY") or os.environ.get("ARK_API_KEY"),
)


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


def build_prompt(prompt_text: str, extra_params: str) -> str:
    prompt_text = prompt_text.strip()
    extra_params = extra_params.strip()
    if prompt_text and extra_params:
        return f"{prompt_text}\n{extra_params}"
    return prompt_text or extra_params


def data_url_from_bytes(file_bytes: bytes, mime_type: str) -> str:
    b64 = base64.b64encode(file_bytes).decode("utf-8")
    return f"data:{mime_type};base64,{b64}"


def poll_task(task_id: str, job_label: str, log_func: Callable[[str], None]):
    start_time = time.time()
    while True:
        result = client.content_generation.tasks.get(task_id=task_id)
        status = getattr(result, "status", "unknown")
        if status == "succeeded":
            log_func(f"[{job_label}] タスク完了")
            return result
        if status == "failed":
            error_info = getattr(result, "error", "不明なエラー")
            raise RuntimeError(f"タスクが失敗しました: {error_info}")
        elapsed = time.time() - start_time
        if elapsed > POLL_TIMEOUT:
            raise TimeoutError(f"タスクが{int(elapsed)}秒経過しても完了しませんでした。")
        time.sleep(POLL_INTERVAL)


def download_binary(url: str, output_path: str) -> None:
    response = requests.get(url, stream=True, timeout=60)
    response.raise_for_status()
    with open(output_path, "wb") as file:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                file.write(chunk)


def save_prompt_text(video_path: str, prompt_text: str, log_func: Callable[[str], None]) -> None:
    if not prompt_text:
        return
    base, _ = os.path.splitext(video_path)
    prompt_path = base + ".txt"
    try:
        with open(prompt_path, "w", encoding="utf-8") as file:
            file.write(prompt_text)
        log_func(f"プロンプトを保存: {prompt_path}")
    except OSError as exc:
        log_func(f"プロンプト保存に失敗: {exc}")


def run_generation(
    prompt_text: str,
    extra_params: str,
    save_dir: str,
    first_frame: Optional[bytes],
    first_mime: Optional[str],
    last_frame: Optional[bytes],
    last_mime: Optional[str],
    log_func: Callable[[str], None],
) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    final_prompt = build_prompt(prompt_text, extra_params)
    if not final_prompt:
        raise ValueError("プロンプトが空です。")
    os.makedirs(save_dir, exist_ok=True)

    content_payload = [{"type": "text", "text": final_prompt}]
    if first_frame and first_mime:
        first_url = data_url_from_bytes(first_frame, first_mime)
        content_payload.append(
            {"type": "image_url", "image_url": {"url": first_url}, "role": "first_frame"}
        )
    if last_frame and last_mime:
        last_url = data_url_from_bytes(last_frame, last_mime)
        content_payload.append(
            {"type": "image_url", "image_url": {"url": last_url}, "role": "last_frame"}
        )

    create_result = client.content_generation.tasks.create(
        model=SEEDANCE_MODEL_ID,
        content=content_payload,
        return_last_frame=True,
    )
    task_id = getattr(create_result, "id", None)
    if not task_id:
        raise RuntimeError("タスクIDを取得できませんでした。")

    result = poll_task(task_id, "Seedance1.5", log_func)

    video_url = getattr(getattr(result, "content", None), "video_url", None)
    if not video_url:
        raise RuntimeError("動画URLを取得できませんでした。")

    video_path = os.path.join(save_dir, f"{task_id}.mp4")
    log_func(f"動画をダウンロード中: {video_path}")
    download_binary(video_url, video_path)
    log_func("動画の保存完了")

    last_frame_url = getattr(getattr(result, "content", None), "last_frame_url", None)
    last_frame_path = None
    if last_frame_url:
        last_frame_path = os.path.join(save_dir, f"{task_id}_last_frame.png")
        log_func(f"last frameをダウンロード中: {last_frame_path}")
        download_binary(last_frame_url, last_frame_path)
        log_func("last frameの保存完了")

    save_prompt_text(video_path, final_prompt, log_func)
    return video_path, task_id, last_frame_path


def init_session_state() -> None:
    if "history" not in st.session_state:
        st.session_state.history = []


def main() -> None:
    st.set_page_config(page_title="Seedance1.5 動画生成", layout="centered")
    init_session_state()
    sync_cookie_controller()
    require_basic_login()
    init_history()

    st.title("Seedance1.5 動画生成 (Streamlit)")

    with st.sidebar:
        st.header("出力設定")
        save_dir = DEFAULT_SAVE_DIR
        resolution_value = st.selectbox("解像度", ["480", "720", "1080"], index=0)
        aspect_value = st.selectbox("アスペクト比", ["21:9", "16:9", "4:3", "1:1", "3:4", "9:16"], index=1)
        duration_value = st.slider("生成秒数", min_value=4, max_value=12, value=12, step=1)
        camera_fixed_value = st.selectbox("カメラFIXED", ["false", "true"], index=0)
        extra_params = (
            f"{BASE_EXTRA_PARAMS} "
            f"--resolution {resolution_value}p "
            f"--ratio {aspect_value} "
            f"--duration {duration_value} "
            f"--camerafixed {camera_fixed_value}"
        )

    st.subheader("入力")
    prompt_text = st.text_area("プロンプト", height=120)
    col1, col2 = st.columns(2)
    with col1:
        first_file = st.file_uploader(
            "開始画像 (任意)", type=["png", "jpg", "jpeg", "webp", "bmp"]
        )
    with col2:
        last_file = st.file_uploader(
            "終了画像 (任意)", type=["png", "jpg", "jpeg", "webp", "bmp"]
        )

    if st.button("生成", type="primary"):
        if not prompt_text.strip():
            st.error("プロンプトを入力してください。")
            return
        try:
            log_func = lambda _msg: None

            first_bytes = first_file.read() if first_file else None
            first_mime = first_file.type if first_file else None
            last_bytes = last_file.read() if last_file else None
            last_mime = last_file.type if last_file else None

            with st.spinner("生成中..."):
                video_path, task_id, last_frame_path = run_generation(
                    prompt_text=prompt_text,
                    extra_params=extra_params,
                    save_dir=save_dir,
                    first_frame=first_bytes,
                    first_mime=first_mime,
                    last_frame=last_bytes,
                    last_mime=last_mime,
                    log_func=log_func,
                )
            if video_path:
                st.session_state["last_video_path"] = video_path
                if last_frame_path:
                    st.session_state["last_frame_path"] = last_frame_path
                st.session_state.history.insert(
                    0,
                    {
                        "id": task_id or os.path.basename(video_path),
                        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                        "prompt": prompt_text.strip(),
                        "extra_params": extra_params.strip(),
                        "model": SEEDANCE_MODEL_ID,
                        "video_path": video_path,
                        "last_frame_path": last_frame_path,
                    },
                )
                persist_history_to_storage(st.session_state.history)
                st.success(f"完了: {video_path}")
        except Exception as exc:  # noqa: BLE001
            st.error(f"エラー: {exc}")

    last_video_path = st.session_state.get("last_video_path")
    if last_video_path and os.path.isfile(last_video_path):
        st.subheader("ダウンロード")
        with open(last_video_path, "rb") as file:
            st.download_button(
                "動画をダウンロード",
                file,
                file_name=os.path.basename(last_video_path),
            )
        last_frame_path = st.session_state.get("last_frame_path")
        if last_frame_path and os.path.isfile(last_frame_path):
            with open(last_frame_path, "rb") as file:
                st.download_button(
                    "ラストフレームをダウンロード",
                    file,
                    file_name=os.path.basename(last_frame_path),
                )
        st.video(last_video_path)

    history = st.session_state.get("history", [])
    if history:
        st.subheader("履歴")
        for item in history:
            video_path = item.get("video_path") or ""
            last_frame_path = item.get("last_frame_path") or ""
            title = f"{item.get('timestamp', '')} | {os.path.basename(video_path)}"
            with st.expander(title, expanded=False):
                st.text_area(
                    "プロンプト",
                    value=item.get("prompt") or "",
                    height=120,
                    key=f"prompt_{title}",
                )
                if item.get("extra_params"):
                    st.code(item.get("extra_params"), language="text")
                if os.path.isfile(video_path):
                    with open(video_path, "rb") as file:
                        st.download_button(
                            "動画をダウンロード",
                            file,
                            file_name=os.path.basename(video_path),
                            key=f"dl_{title}",
                        )
                    if last_frame_path and os.path.isfile(last_frame_path):
                        with open(last_frame_path, "rb") as file:
                            st.download_button(
                                "ラストフレームをダウンロード",
                                file,
                                file_name=os.path.basename(last_frame_path),
                                key=f"last_{title}",
                            )
                    st.video(video_path)
                else:
                    st.warning("動画ファイルが見つかりません。")


if __name__ == "__main__":
    main()
