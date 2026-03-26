import mimetypes
import os
import tempfile
import time
import json
from urllib.parse import urlparse
from typing import Callable, Optional, Tuple

import streamlit as st
from dotenv import load_dotenv
from google.cloud import storage
from google import genai
from google.genai import types
from google.oauth2 import service_account

from streamlit_auth_history_utils import (
    get_secret_value,
    init_history,
    persist_history_to_storage,
    sync_cookie_controller,
)

load_dotenv()

ENV_VEO_MODEL = (
    get_secret_value("VEO_MODEL_ID")
    or os.environ.get("VEO_MODEL_ID")
)
GEMINI_MODEL_OPTIONS = [
    "veo-3.1-generate-preview",
    "veo-3.1-fast-generate-preview",
]
VERTEX_MODEL_OPTIONS = [
    "veo-3.1-generate-001",
    "veo-3.1-fast-generate-001",
]
MODEL_LABELS = {
    "veo-3.1-fast-generate-preview": "高速モデル",
    "veo-3.1-generate-preview": "高価・高品質モデル",
    "veo-3.1-fast-generate-001": "高速モデル",
    "veo-3.1-generate-001": "高価・高品質モデル",
}
DEFAULT_SAVE_DIR = os.path.join(tempfile.gettempdir(), "veo_outputs")
POLL_INTERVAL = 10
POLL_TIMEOUT = 1800
HIGH_RESOLUTION_OPTIONS = {"1080p"}
GEMINI_API_UNSUPPORTED_OPTIONS = {
    "enhance_prompt",
    "negative_prompt",
    "seed",
}
VERTEX_AI_SCOPES = ["https://www.googleapis.com/auth/cloud-platform"]


def get_api_key() -> Optional[str]:
    return (
        get_secret_value("CX_GEMINI_API_KEY")
        or os.environ.get("CX_GEMINI_API_KEY")
        or get_secret_value("GEMINI_API_KEY")
        or get_secret_value("GOOGLE_API_KEY")
        or os.environ.get("GEMINI_API_KEY")
        or os.environ.get("GOOGLE_API_KEY")
    )


def get_vertex_project() -> Optional[str]:
    return (
        get_secret_value("GOOGLE_CLOUD_PROJECT")
        or os.environ.get("GOOGLE_CLOUD_PROJECT")
        or get_secret_value("GCP_PROJECT")
        or os.environ.get("GCP_PROJECT")
        or get_secret_value("VERTEX_PROJECT_ID")
        or os.environ.get("VERTEX_PROJECT_ID")
        or get_gcp_service_account_project()
    )


def get_vertex_location() -> str:
    return (
        get_secret_value("GOOGLE_CLOUD_LOCATION")
        or os.environ.get("GOOGLE_CLOUD_LOCATION")
        or get_secret_value("VERTEX_LOCATION")
        or os.environ.get("VERTEX_LOCATION")
        or "us-central1"
    )


def get_gcp_service_account_section() -> Optional[dict]:
    try:
        secrets_obj = st.secrets
    except Exception:
        return None
    section = None
    if isinstance(secrets_obj, dict):
        section = secrets_obj.get("gcp_service_account")
    else:
        getter = getattr(secrets_obj, "get", None)
        if callable(getter):
            try:
                section = getter("gcp_service_account")
            except Exception:
                section = None
    if section is None:
        return None
    if isinstance(section, dict):
        return dict(section)
    try:
        return dict(section)
    except Exception:
        return None


def get_gcp_service_account_project() -> Optional[str]:
    section = get_gcp_service_account_section()
    if not section:
        return None
    project_id = section.get("project_id")
    if isinstance(project_id, str) and project_id.strip():
        return project_id.strip()
    raw_json = section.get("service_account_json")
    if isinstance(raw_json, str) and raw_json.strip():
        try:
            payload = json.loads(raw_json)
        except json.JSONDecodeError:
            return None
        project_id = payload.get("project_id")
        if isinstance(project_id, str) and project_id.strip():
            return project_id.strip()
    return None


def get_vertex_credentials():
    section = get_gcp_service_account_section()
    if not section:
        return None
    raw_json = section.get("service_account_json")
    info = None
    if isinstance(raw_json, str) and raw_json.strip():
        try:
            info = json.loads(raw_json)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"service_account_json のJSON解析に失敗しました: {exc}") from exc
    else:
        info = {key: value for key, value in section.items() if key != "bucket_name"}
    if not isinstance(info, dict) or not info.get("client_email") or not info.get("private_key"):
        return None
    return service_account.Credentials.from_service_account_info(
        info,
        scopes=VERTEX_AI_SCOPES,
    )


def parse_bool(value: Optional[object]) -> Optional[bool]:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    return None


def should_use_vertex_ai() -> bool:
    explicit = parse_bool(
        get_secret_value("GOOGLE_GENAI_USE_VERTEXAI")
        or os.environ.get("GOOGLE_GENAI_USE_VERTEXAI")
        or get_secret_value("VEO_USE_VERTEXAI")
        or os.environ.get("VEO_USE_VERTEXAI")
    )
    if explicit is not None:
        return explicit
    return bool(get_vertex_project())


def get_model_options(using_vertex_ai: bool) -> list[str]:
    options = VERTEX_MODEL_OPTIONS if using_vertex_ai else GEMINI_MODEL_OPTIONS
    if ENV_VEO_MODEL:
        return list(dict.fromkeys([ENV_VEO_MODEL, *options]))
    return list(options)


def get_default_model(using_vertex_ai: bool) -> str:
    if ENV_VEO_MODEL:
        return ENV_VEO_MODEL
    if using_vertex_ai:
        return "veo-3.1-fast-generate-001"
    return "veo-3.1-fast-generate-preview"


def build_client() -> genai.Client:
    if should_use_vertex_ai():
        project = get_vertex_project()
        if not project:
            raise RuntimeError("Vertex AI モードには GOOGLE_CLOUD_PROJECT が必要です。")
        credentials = get_vertex_credentials()
        return genai.Client(
            vertexai=True,
            project=project,
            location=get_vertex_location(),
            credentials=credentials,
        )
    api_key = get_api_key()
    if not api_key:
        raise RuntimeError("CX_GEMINI_API_KEY または Vertex AI 設定が未設定です。")
    return genai.Client(api_key=api_key)


def is_vertex_ai_client(client: genai.Client) -> bool:
    return bool(getattr(client, "vertexai", False))


def build_history_params(
    aspect_ratio: str,
    resolution: str,
    duration_seconds: int,
    enhance_prompt: bool,
    seed_value: Optional[int],
    negative_prompt: str,
    has_first_frame: bool,
    has_last_frame: bool,
) -> str:
    params = [
        f"--aspect-ratio {aspect_ratio}",
        f"--resolution {resolution}",
        f"--duration {duration_seconds}",
        f"--enhance-prompt {str(enhance_prompt).lower()}",
    ]
    if seed_value is not None:
        params.append(f"--seed {seed_value}")
    if negative_prompt.strip():
        params.append(f"--negative-prompt {negative_prompt.strip()}")
    if has_first_frame:
        params.append("--first-frame true")
    if has_last_frame:
        params.append("--last-frame true")
    return "\n".join(params)


def normalize_image_mime(file_name: Optional[str], mime_type: Optional[str]) -> str:
    if mime_type:
        return mime_type
    guessed_mime, _ = mimetypes.guess_type(file_name or "")
    return guessed_mime or "image/png"


def make_image(
    file_bytes: Optional[bytes],
    file_name: Optional[str],
    mime_type: Optional[str],
) -> Optional[types.Image]:
    if not file_bytes:
        return None
    return types.Image(
        image_bytes=file_bytes,
        mime_type=normalize_image_mime(file_name, mime_type),
    )


def validate_inputs(
    prompt_text: str,
    first_frame: Optional[bytes],
    last_frame: Optional[bytes],
    resolution: str,
    duration_seconds: int,
) -> None:
    if not prompt_text.strip() and not first_frame:
        raise ValueError("プロンプト、または開始画像のどちらかを入力してください。")
    if last_frame and not first_frame:
        raise ValueError("終了画像を使う場合は開始画像も指定してください。")
    if resolution in HIGH_RESOLUTION_OPTIONS and duration_seconds != 8:
        raise ValueError("1080p は 8 秒生成のみ対応です。")


def save_prompt_text(
    video_path: str,
    prompt_text: str,
    negative_prompt: str,
    params_text: str,
    log_func: Callable[[str], None],
) -> None:
    lines = []
    if prompt_text.strip():
        lines.append("[prompt]")
        lines.append(prompt_text.strip())
        lines.append("")
    if negative_prompt.strip():
        lines.append("[negative_prompt]")
        lines.append(negative_prompt.strip())
        lines.append("")
    if params_text.strip():
        lines.append("[params]")
        lines.append(params_text.strip())
        lines.append("")
    if not lines:
        return

    base, _ = os.path.splitext(video_path)
    prompt_path = base + ".txt"
    try:
        with open(prompt_path, "w", encoding="utf-8") as file:
            file.write("\n".join(lines).rstrip() + "\n")
        log_func(f"プロンプトを保存: {prompt_path}")
    except OSError as exc:
        log_func(f"プロンプト保存に失敗: {exc}")


def poll_operation(
    client: genai.Client,
    operation: types.GenerateVideosOperation,
    log_func: Callable[[str], None],
) -> types.GenerateVideosOperation:
    start_time = time.time()
    current_operation = operation
    while not current_operation.done:
        elapsed = time.time() - start_time
        if elapsed > POLL_TIMEOUT:
            raise TimeoutError(f"タスクが{int(elapsed)}秒経過しても完了しませんでした。")
        log_func("Veo タスクの完了待機中...")
        time.sleep(POLL_INTERVAL)
        current_operation = client.operations.get(current_operation)
    return current_operation


def download_gcs_video(video_uri: str, output_path: str) -> None:
    parsed = urlparse(video_uri)
    if parsed.scheme != "gs" or not parsed.netloc or not parsed.path:
        raise RuntimeError(f"サポート外の動画URIです: {video_uri}")
    client = storage.Client(project=get_vertex_project() or None)
    bucket = client.bucket(parsed.netloc)
    blob = bucket.blob(parsed.path.lstrip("/"))
    blob.download_to_filename(output_path)


def save_generated_video(
    client: genai.Client,
    generated_video: types.GeneratedVideo,
    output_path: str,
) -> None:
    video = generated_video.video
    if video is None:
        raise RuntimeError("生成された動画情報を取得できませんでした。")
    if video.video_bytes:
        video.save(output_path)
        return
    if is_vertex_ai_client(client):
        if video.uri:
            download_gcs_video(video.uri, output_path)
            return
        raise RuntimeError("Vertex AI の動画出力URIを取得できませんでした。")
    client.files.download(file=generated_video)
    video.save(output_path)


def run_generation(
    model: str,
    prompt_text: str,
    negative_prompt: str,
    aspect_ratio: str,
    resolution: str,
    duration_seconds: int,
    enhance_prompt: bool,
    seed_value: Optional[int],
    save_dir: str,
    first_frame: Optional[bytes],
    first_name: Optional[str],
    first_mime: Optional[str],
    last_frame: Optional[bytes],
    last_name: Optional[str],
    last_mime: Optional[str],
    log_func: Callable[[str], None],
) -> Tuple[str, str]:
    validate_inputs(prompt_text, first_frame, last_frame, resolution, duration_seconds)
    os.makedirs(save_dir, exist_ok=True)

    client = build_client()
    is_vertex_ai = is_vertex_ai_client(client)
    source = types.GenerateVideosSource(
        prompt=prompt_text.strip() or None,
        image=make_image(first_frame, first_name, first_mime),
    )
    config_kwargs = dict(
        aspect_ratio=aspect_ratio,
        resolution=resolution,
        duration_seconds=duration_seconds,
        last_frame=make_image(last_frame, last_name, last_mime),
    )
    if is_vertex_ai:
        config_kwargs.update(
            enhance_prompt=enhance_prompt,
            negative_prompt=negative_prompt.strip() or None,
            seed=seed_value,
        )
    config = types.GenerateVideosConfig(**config_kwargs)

    operation = client.models.generate_videos(
        model=model,
        source=source,
        config=config,
    )
    operation = poll_operation(client, operation, log_func)

    if operation.error:
        raise RuntimeError(str(operation.error))

    response = operation.result or operation.response
    generated_videos = getattr(response, "generated_videos", None) or []
    if not generated_videos:
        raise RuntimeError("生成動画を取得できませんでした。")

    operation_id = (operation.name or "").rstrip("/").split("/")[-1] or str(int(time.time()))
    video_path = os.path.join(save_dir, f"{operation_id}.mp4")
    log_func(f"動画を保存中: {video_path}")
    save_generated_video(client, generated_videos[0], video_path)
    log_func("動画の保存完了")

    params_text = build_history_params(
        aspect_ratio=aspect_ratio,
        resolution=resolution,
        duration_seconds=duration_seconds,
        enhance_prompt=enhance_prompt,
        seed_value=seed_value,
        negative_prompt=negative_prompt,
        has_first_frame=first_frame is not None,
        has_last_frame=last_frame is not None,
    )
    save_prompt_text(video_path, prompt_text, negative_prompt, params_text, log_func)
    return video_path, operation_id


def init_session_state() -> None:
    if "history" not in st.session_state:
        st.session_state.history = []


def render_history_item(item: dict) -> None:
    video_path = item.get("video_path") or ""
    title = f"{item.get('timestamp', '')} | {os.path.basename(video_path)}"
    with st.expander(title, expanded=False):
        if item.get("prompt"):
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
            st.video(video_path)
        else:
            st.warning("動画ファイルが見つかりません。")


def main() -> None:
    st.set_page_config(page_title="Veo 動画生成", layout="centered")
    init_session_state()
    sync_cookie_controller()
    init_history()

    st.title("Veo 動画生成 (No Auth)")

    with st.sidebar:
        st.header("出力設定")
        save_dir = DEFAULT_SAVE_DIR
        using_vertex_ai = should_use_vertex_ai()
        model_options = get_model_options(using_vertex_ai)
        default_model = get_default_model(using_vertex_ai)
        model_value = st.selectbox(
            "モデル",
            model_options,
            index=model_options.index(default_model),
            format_func=lambda model_id: MODEL_LABELS.get(model_id, model_id),
        )
        aspect_ratio = st.selectbox("アスペクト比", ["16:9", "9:16"], index=0)
        resolution = st.selectbox("解像度", ["720p", "1080p"], index=0)
        duration_seconds = st.selectbox("生成秒数", [4, 6, 8], index=0)
        enhance_prompt = st.checkbox("プロンプト補強", value=True, disabled=not using_vertex_ai)
        seed_text = st.text_input("Seed (任意)", value="", disabled=not using_vertex_ai)
        negative_prompt = st.text_area(
            "ネガティブプロンプト (任意)",
            height=90,
            disabled=not using_vertex_ai,
        )
        st.caption("1080p は 8 秒のみ対応です。")
        if using_vertex_ai:
            st.caption(f"現在は Vertex AI モードです。location: {get_vertex_location()}")
        else:
            st.caption(
                "現在は Gemini Developer API モードです。未対応: "
                + " / ".join(sorted(GEMINI_API_UNSUPPORTED_OPTIONS))
            )

    st.subheader("入力")
    st.caption("Veo は英語プロンプトのほうが安定しやすいです。")
    prompt_text = st.text_area("プロンプト", height=120)
    col1, col2 = st.columns(2)
    with col1:
        first_file = st.file_uploader(
            "開始画像 (任意)",
            type=["png", "jpg", "jpeg", "webp"],
        )
    with col2:
        last_file = st.file_uploader(
            "終了画像 (任意)",
            type=["png", "jpg", "jpeg", "webp"],
        )

    if st.button("生成", type="primary"):
        try:
            seed_value = None
            if seed_text.strip():
                seed_value = int(seed_text.strip())

            first_bytes = first_file.read() if first_file else None
            last_bytes = last_file.read() if last_file else None

            with st.spinner("生成中..."):
                video_path, operation_id = run_generation(
                    model=model_value,
                    prompt_text=prompt_text,
                    negative_prompt=negative_prompt,
                    aspect_ratio=aspect_ratio,
                    resolution=resolution,
                    duration_seconds=duration_seconds,
                    enhance_prompt=enhance_prompt,
                    seed_value=seed_value,
                    save_dir=save_dir,
                    first_frame=first_bytes,
                    first_name=first_file.name if first_file else None,
                    first_mime=first_file.type if first_file else None,
                    last_frame=last_bytes,
                    last_name=last_file.name if last_file else None,
                    last_mime=last_file.type if last_file else None,
                    log_func=lambda _msg: None,
                )

            params_text = build_history_params(
                aspect_ratio=aspect_ratio,
                resolution=resolution,
                duration_seconds=duration_seconds,
                enhance_prompt=enhance_prompt,
                seed_value=seed_value,
                negative_prompt=negative_prompt,
                has_first_frame=first_bytes is not None,
                has_last_frame=last_bytes is not None,
            )
            st.session_state["last_video_path"] = video_path
            st.session_state.history.insert(
                0,
                {
                    "id": operation_id,
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "prompt": prompt_text.strip(),
                    "extra_params": params_text,
                    "model": model_value,
                    "video_path": video_path,
                },
            )
            persist_history_to_storage(st.session_state.history)
            st.success(f"完了: {video_path}")
        except ValueError as exc:
            st.error(f"入力エラー: {exc}")
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
        st.video(last_video_path)

    history = st.session_state.get("history", [])
    if history:
        st.subheader("履歴")
        for item in history:
            render_history_item(item)


if __name__ == "__main__":
    main()
