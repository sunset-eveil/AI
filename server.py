import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Lock

from inference import detect_entities

inference_lock = Lock()

class PiiHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        pass

    def send_json(self, status, data):
        body = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def send_error_json(self, status, request_id, code, message):
        self.send_json(status, {
            "request_id": request_id,
            "error": {"code": code, "message": message},
        })

    def do_POST(self):
        if self.path != "/api/v1/pii/detect":
            self.send_error_json(404, None, "NOT_FOUND", "API 경로가 올바르지 않습니다.")
            return

        request_id = None
        try:
            size = int(self.headers.get("Content-Length", "0"))
            if size <= 0:
                raise ValueError("본문 없음")
            if size > 65536:
                self.send_error_json(413, None, "INPUT_TOO_LONG", "요청 본문이 너무 큽니다.")
                return

            data = json.loads(self.rfile.read(size).decode("utf-8"))
            if not isinstance(data, dict):
                raise ValueError("JSON 객체가 아님")

            value = data.get("request_id")
            if isinstance(value, str) and value.strip():
                request_id = value

            text = data.get("text")
            if request_id is None or not isinstance(text, str) or not text.strip():
                raise ValueError("필수 입력이 올바르지 않음")
        except ValueError:
            self.send_error_json(400, request_id, "INVALID_INPUT", "요청 JSON을 확인해 주세요.")
            return

        if not inference_lock.acquire(blocking=False):
            self.send_error_json(503, request_id, "SERVER_BUSY", "다른 요청을 처리 중입니다.")
            return

        try:
            entities = detect_entities(text)
        except Exception as error:
            if isinstance(error, ValueError) and str(error) == "INPUT_TOO_LONG":
                self.send_error_json(413, request_id, "INPUT_TOO_LONG", "256토큰을 초과했습니다.")
            else:
                self.send_error_json(500, request_id, "INFERENCE_FAILED", "추론에 실패했습니다.")
            return
        finally:
            inference_lock.release()

        self.send_json(200, {
            "request_id": request_id,
            "model_version": "eveil-ai-v1",
            "offset_unit": "unicode_codepoint",
            "entities": entities,
        })

if __name__ == "__main__":
    with ThreadingHTTPServer(("127.0.0.1", 8000), PiiHandler) as server:
        print("완료")
        server.serve_forever()