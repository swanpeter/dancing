import base64
import datetime
import json
import os
import tempfile
import time
import uuid
from typing import Any, Dict, List, Optional, Tuple

import streamlit as st

try:
    from streamlit_cookies_controller import CookieController
except ImportError:
    CookieController = None

try:
    from streamlit.runtime.secrets import StreamlitSecretNotFoundError
except ImportError:
    StreamlitSecretNotFoundError = Exception


def get_secret_value(key: str) -> Optional[str]:
    try:
        secrets_obj = st.secrets
    except StreamlitSecretNotFoundError:
        return None
    except Exception:
        return None
    try:
        return secrets_obj[key]
    except (KeyError, TypeError, StreamlitSecretNotFoundError):
        pass
    get_method = getattr(secrets_obj, "get", None)
    if callable(get_method):
        try:
            return get_method(key)
        except Exception:
            return None
    return None


def rerun_app() -> None:
    rerun = getattr(st, "rerun", None)
    if callable(rerun):
        rerun()
        return
    experimental_rerun = getattr(st, "experimental_rerun", None)
    if callable(experimental_rerun):
        experimental_rerun()


def _normalize_credential(value: Optional[str]) -> Optional[str]:
    if isinstance(value, str):
        stripped = value.strip()
        if stripped:
            return stripped
    return None


def get_secret_auth_credentials() -> Tuple[Optional[str], Optional[str]]:
    try:
        secrets_obj = st.secrets
    except StreamlitSecretNotFoundError:
        return None, None
    except Exception:
        return None, None

    auth_section: Optional[Dict[str, Any]] = None
    if isinstance(secrets_obj, dict):
        auth_section = secrets_obj.get("auth")
    else:
        auth_section = getattr(secrets_obj, "get", lambda _key, _default=None: None)("auth")

    def _get_from_container(container: object, key: str) -> Optional[Any]:
        if isinstance(container, dict):
            return container.get(key)
        getter = getattr(container, "get", None)
        if callable(getter):
            try:
                return getter(key)
            except TypeError:
                try:
                    return getter(key, None)
                except TypeError:
                    return None
        try:
            return getattr(container, key)
        except AttributeError:
            return None

    def _extract_credential(container: object, keys: Tuple[str, ...]) -> Optional[Any]:
        for key in keys:
            value = _get_from_container(container, key)
            if value is not None:
                return value
        return None

    username = None
    password = None
    if auth_section is not None:
        username = _extract_credential(auth_section, ("username", "id", "user", "name"))
        password = _extract_credential(auth_section, ("password", "pass", "pwd"))

    if username is None:
        username = get_secret_value("USERNAME") or get_secret_value("ID")
    if password is None:
        password = get_secret_value("PASSWORD") or get_secret_value("PASS")

    normalized_username = _normalize_credential(str(username)) if username is not None else None
    normalized_password = _normalize_credential(str(password)) if password is not None else None
    return normalized_username, normalized_password


def get_configured_auth_credentials(
    default_username: str,
    default_password: str,
) -> Tuple[str, str]:
    secret_username, secret_password = get_secret_auth_credentials()
    if secret_username and secret_password:
        return secret_username, secret_password
    return default_username, default_password


def _get_cookie_controller() -> Optional[object]:
    if CookieController is None:
        return None
    controller = st.session_state.get("_cookie_controller")
    if controller is None:
        try:
            controller = CookieController()
        except Exception:
            return None
        st.session_state["_cookie_controller"] = controller
    return controller


def sync_cookie_controller() -> None:
    controller = _get_cookie_controller()
    if controller is None:
        return
    sync_stage = st.session_state.get("_cookies_sync_stage", 0)
    if sync_stage == 0:
        try:
            controller.refresh()
        except Exception:
            return
        st.session_state["_cookies_sync_stage"] = 1
        rerun_app()
        return
    if sync_stage == 1:
        try:
            controller.refresh()
        except Exception:
            return
        st.session_state["_cookies_sync_stage"] = 2


def restore_login_from_cookie(
    cookie_key: str = "logged_in",
    attempts: int = 2,
    sleep_seconds: float = 0.3,
) -> bool:
    controller = _get_cookie_controller()
    if controller is None:
        return False
    for _ in range(max(1, attempts)):
        try:
            controller.refresh()
            if controller.get(cookie_key) == "1":
                return True
        except Exception:
            return False
        time.sleep(sleep_seconds)
    return False


def persist_login_to_cookie(
    value: bool,
    cookie_key: str = "logged_in",
    set_sleep_seconds: float = 0.6,
) -> None:
    controller = _get_cookie_controller()
    if controller is None:
        return
    try:
        if value:
            controller.set(cookie_key, "1")
            time.sleep(set_sleep_seconds)
        else:
            controller.remove(cookie_key)
    except Exception:
        return


def require_basic_login(
    username: Optional[str] = None,
    password: Optional[str] = None,
    *,
    cookie_key: str = "logged_in",
    session_state_key: str = "authenticated",
    username_label: str = "ID",
    password_label: str = "PASS",
    title: str = "ログイン",
    success_message: str = "ログインしました。",
) -> None:
    if session_state_key not in st.session_state:
        st.session_state[session_state_key] = False

    if not st.session_state[session_state_key] and restore_login_from_cookie(cookie_key=cookie_key):
        st.session_state[session_state_key] = True

    if st.session_state[session_state_key]:
        return

    st.title(title)

    if username is None or password is None:
        username, password = get_secret_auth_credentials()

    if not username or not password:
        st.info("ログイン情報が未設定です。管理者に連絡してください。")
        st.stop()
        return

    with st.form("login_form", clear_on_submit=False):
        input_username = st.text_input(username_label)
        input_password = st.text_input(password_label, type="password")
        submitted = st.form_submit_button("ログイン")

    if submitted:
        if input_username == username and input_password == password:
            st.session_state[session_state_key] = True
            persist_login_to_cookie(True, cookie_key=cookie_key)
            st.success(success_message)
            rerun_app()
            return
        st.error("IDまたはPASSが正しくありません。")
    st.stop()


def _get_history_dir(history_dir: Optional[str]) -> str:
    if history_dir:
        return history_dir
    return os.path.join(tempfile.gettempdir(), "streamlit_history")


def _get_history_path(session_id: str, history_dir: Optional[str] = None) -> str:
    history_root = _get_history_dir(history_dir)
    os.makedirs(history_root, exist_ok=True)
    safe_id = "".join(ch for ch in session_id if ch.isalnum() or ch in {"-", "_"})
    return os.path.join(history_root, f"{safe_id}.json")


def get_browser_session_id(
    cookie_key: str = "browser_session_id",
    create: bool = True,
    set_sleep_seconds: float = 0.6,
) -> Optional[str]:
    controller = _get_cookie_controller()
    if controller is None:
        return None
    try:
        controller.refresh()
        session_id = controller.get(cookie_key)
    except Exception:
        session_id = None
    if session_id:
        return str(session_id)
    if not create:
        return None
    new_id = uuid.uuid4().hex
    try:
        controller.set(cookie_key, new_id)
        time.sleep(set_sleep_seconds)
    except Exception:
        return None
    return new_id


def _serialize_history(history: List[Dict[str, object]]) -> List[Dict[str, object]]:
    serialized: List[Dict[str, object]] = []
    for entry in history:
        image_bytes = entry.get("image_bytes")
        if isinstance(image_bytes, (bytes, bytearray, memoryview)):
            image_b64 = base64.b64encode(bytes(image_bytes)).decode("utf-8")
        else:
            image_b64 = None
        serialized.append(
            {
                "id": entry.get("id"),
                "prompt": entry.get("prompt"),
                "model": entry.get("model"),
                "no_text": entry.get("no_text"),
                "extra_params": entry.get("extra_params"),
                "video_path": entry.get("video_path"),
                "last_frame_path": entry.get("last_frame_path"),
                "timestamp": entry.get("timestamp"),
                "image_b64": image_b64,
            }
        )
    return serialized


def _deserialize_history(payload: List[Dict[str, object]]) -> List[Dict[str, object]]:
    history: List[Dict[str, object]] = []
    for entry in payload:
        image_b64 = entry.get("image_b64")
        image_bytes = decode_image_data(image_b64) if image_b64 else None
        history.append(
            {
                "id": entry.get("id"),
                "prompt": entry.get("prompt"),
                "model": entry.get("model"),
                "no_text": entry.get("no_text"),
                "extra_params": entry.get("extra_params"),
                "video_path": entry.get("video_path"),
                "last_frame_path": entry.get("last_frame_path"),
                "timestamp": entry.get("timestamp"),
                "image_bytes": image_bytes,
            }
        )
    return history


def load_history_from_storage(
    history_dir: Optional[str] = None,
    session_cookie_key: str = "browser_session_id",
) -> Optional[List[Dict[str, object]]]:
    session_id = get_browser_session_id(cookie_key=session_cookie_key, create=False)
    if not session_id:
        return None
    history_path = _get_history_path(session_id, history_dir=history_dir)
    if not os.path.exists(history_path):
        return None
    try:
        with open(history_path, "r", encoding="utf-8") as file_handle:
            payload = json.load(file_handle)
    except Exception:
        return None
    if not isinstance(payload, dict):
        return None
    entries = payload.get("history")
    if not isinstance(entries, list):
        return None
    return _deserialize_history(entries)


def persist_history_to_storage(
    history: List[Dict[str, object]],
    history_dir: Optional[str] = None,
    session_cookie_key: str = "browser_session_id",
) -> None:
    session_id = get_browser_session_id(cookie_key=session_cookie_key, create=True)
    if not session_id:
        return
    history_path = _get_history_path(session_id, history_dir=history_dir)
    payload = {
        "updated_at": datetime.datetime.utcnow().isoformat(),
        "history": _serialize_history(history),
    }
    try:
        with open(history_path, "w", encoding="utf-8") as file_handle:
            json.dump(payload, file_handle)
    except Exception:
        return


def clear_history_storage(
    history_dir: Optional[str] = None,
    session_cookie_key: str = "browser_session_id",
) -> None:
    session_id = get_browser_session_id(cookie_key=session_cookie_key, create=False)
    if not session_id:
        return
    history_path = _get_history_path(session_id, history_dir=history_dir)
    try:
        if os.path.exists(history_path):
            os.remove(history_path)
    except Exception:
        return


def init_history(
    session_state_key: str = "history",
    history_dir: Optional[str] = None,
    session_cookie_key: str = "browser_session_id",
) -> None:
    if session_state_key not in st.session_state:
        st.session_state[session_state_key] = []
    if not st.session_state.get("_history_loaded"):
        restored = load_history_from_storage(
            history_dir=history_dir,
            session_cookie_key=session_cookie_key,
        )
        if restored is not None:
            st.session_state[session_state_key] = restored
            st.session_state["_history_loaded"] = True
        else:
            if (
                get_browser_session_id(cookie_key=session_cookie_key, create=False) is not None
                or _get_cookie_controller() is None
            ):
                st.session_state["_history_loaded"] = True


def decode_image_data(data: Optional[object]) -> Optional[bytes]:
    if data is None:
        return None
    if isinstance(data, bytes):
        return data
    if isinstance(data, str):
        try:
            return base64.b64decode(data)
        except (ValueError, TypeError):
            return None
    return None
