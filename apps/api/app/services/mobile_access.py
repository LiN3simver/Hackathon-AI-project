"""Generate the phone entry URL and QR code once per API startup."""

import base64
import ipaddress
import socket
from urllib.parse import urlsplit

import qrcode
from qrcode.image.svg import SvgPathFillImage

from app.core.config import get_settings


def generate_mobile_access() -> dict:
    settings = get_settings()
    url = settings.mobile_web_url.strip()
    if not url:
        # UDP route selection requires no packet to be sent or Internet response.
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as probe:
                probe.connect(("192.0.2.1", 80))
                address = probe.getsockname()[0]
            if ipaddress.ip_address(address).is_loopback:
                raise OSError("only loopback available")
        except OSError:
            return {
                "url": None,
                "qr_data_url": None,
                "message": "未找到可用區網 IP，請設定 GUARDIAN_MOBILE_WEB_URL 後重新啟動 API。",
            }
        # The user web dev server runs on HTTPS so phones can open the camera.
        url = f"https://{address}:{settings.mobile_web_port}"
    parsed = urlsplit(url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname or parsed.username or parsed.password:
        raise ValueError("GUARDIAN_MOBILE_WEB_URL 必須是完整 http/https 前台網址")
    if parsed.query or parsed.fragment:
        raise ValueError("GUARDIAN_MOBILE_WEB_URL 不得包含登入憑證、查詢參數或 fragment")
    if parsed.hostname.lower() == "localhost" or parsed.hostname in {"127.0.0.1", "::1", "0.0.0.0"}:
        raise ValueError("GUARDIAN_MOBILE_WEB_URL 請使用手機可連線的區網 IP 或 HTTPS 網址")
    svg = qrcode.make(url, image_factory=SvgPathFillImage, border=4).to_string()
    return {
        "url": url,
        "qr_data_url": "data:image/svg+xml;base64," + base64.b64encode(svg).decode("ascii"),
        "message": "手機與電腦請連到相同 Wi-Fi。第一次開啟會出現憑證警告（本機自簽憑證），"
        "請確認網址是這台主機後選擇「繼續前往」，即可使用相機。",
    }
