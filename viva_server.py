"""Web server for the coursework prototype and Hugging Face Docker Space."""
from __future__ import annotations

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import mimetypes
import os
from pathlib import Path
import sys
from datetime import datetime, timezone
from urllib.parse import unquote, urlsplit

from model_inference import TrainedPredictor
from pdf_report import build_pdf_report

ROOT = Path(__file__).resolve().parent
HOST = os.environ.get('HOST', '127.0.0.1')
PORT = int(os.environ.get('PORT', '8765'))
ALLOWED = {'index.html', 'styles.css', 'polish.css', 'design.css', 'script.js'}
PREDICTOR = TrainedPredictor()
REJECTION_LOG = ROOT / 'viva_rejections.jsonl'


def log_rejection(filename: str, reason: str):
    record = {'time_utc': datetime.now(timezone.utc).isoformat(),
              'filename': Path(filename).name, 'reason': reason}
    try:
        with REJECTION_LOG.open('a', encoding='utf-8') as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + '\n')
    except OSError as exc:
        print(f'Could not write rejection log: {exc}', file=sys.stderr)


class Handler(BaseHTTPRequestHandler):
    def send_json(self, status: int, value: dict):
        body = json.dumps(value, separators=(',', ':')).encode('utf-8')
        self.send_response(status)
        self.send_header('Content-Type', 'application/json; charset=utf-8')
        self.send_header('Content-Length', str(len(body)))
        self.send_header('Cache-Control', 'no-store')
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        requested = unquote(urlsplit(self.path).path).lstrip('/') or 'index.html'
        if requested not in ALLOWED:
            self.send_error(404)
            return
        file = ROOT / requested
        if not file.is_file():
            self.send_error(404)
            return
        data = file.read_bytes()
        self.send_response(200)
        self.send_header('Content-Type', mimetypes.guess_type(file.name)[0] or 'application/octet-stream')
        self.send_header('Content-Length', str(len(data)))
        self.send_header('X-Content-Type-Options', 'nosniff')
        self.send_header('Cache-Control', 'no-store')
        self.end_headers()
        self.wfile.write(data)

    def do_POST(self):
        endpoint = urlsplit(self.path).path
        if endpoint not in {'/api/predict', '/api/report'}:
            self.send_error(404)
            return
        filename = Path(unquote(self.headers.get('X-Filename', 'upload.png'))).name
        try:
            size = int(self.headers.get('Content-Length', '-1'))
            if not 0 < size <= 128 * 1024 * 1024:
                raise ValueError('Upload must be a nonempty JPEG/PNG under 128 MB.')
            payload = self.rfile.read(size)
            result = PREDICTOR.predict(payload, filename)
            if endpoint == '/api/report':
                pdf = build_pdf_report(result, payload)
                safe_stem = ''.join(character if character.isascii() and (character.isalnum() or character in '-_') else '-'
                                    for character in Path(filename).stem)[:70] or 'retinal-report'
                self.send_response(200)
                self.send_header('Content-Type', 'application/pdf')
                self.send_header('Content-Disposition', f'attachment; filename="{safe_stem}-report.pdf"')
                self.send_header('Content-Length', str(len(pdf)))
                self.send_header('Cache-Control', 'no-store')
                self.send_header('X-Content-Type-Options', 'nosniff')
                self.end_headers()
                self.wfile.write(pdf)
                return
            if result.get('classification_permitted'):
                try:
                    from viva_pipeline import analyze
                    evidence = analyze(payload, filename, include_visuals=True)
                    result['preprocessing_evidence'] = evidence['steps']
                except Exception as exc:
                    print(f'Preprocessing evidence error: {exc}', file=sys.stderr)
                    result['preprocessing_evidence_error'] = (
                        'The prediction completed, but preprocessing evidence previews could not be generated. '
                        'Check the local server output for details.'
                    )
            self.send_json(200, result)
        except ValueError as exc:
            log_rejection(filename, str(exc))
            self.send_json(400, {'error': str(exc)})
        except Exception as exc:
            print(f'Analysis error: {exc}', file=sys.stderr)
            self.send_json(500, {'error': 'The image could not be processed. Check the server output for details.'})


if __name__ == '__main__':
    print(f'Viva prototype: http://{HOST}:{PORT}', flush=True)
    ThreadingHTTPServer((HOST, PORT), Handler).serve_forever()
