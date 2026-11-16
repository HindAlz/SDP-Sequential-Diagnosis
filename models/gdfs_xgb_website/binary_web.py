"""Local binary consultation UI: GDFS-inspired acquisition + XGBoost diagnosis."""
import argparse
import json
import math
import secrets
import sys
import threading
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import os
from urllib.parse import urlparse


class Consultation:
    def __init__(self, engine, initial):
        self.engine = engine
        self.initial = initial
        self.history = [(initial, 1)]
        self.asked = {initial}
        self.turns = []
        self.pending = None
        self.reason = None
        self.ranking = []
        self.initial_ranking = []
        self.created = datetime.now().isoformat()

    def advance(self):
        self.ranking = self.engine.predict(self.history)
        if not self.initial_ranking:
            self.initial_ranking = self.ranking
        if len(self.turns) >= self.engine.max_turn:
            self.reason = 'question_budget'
            self.pending = None
            return
        self.pending, self.reason = self.engine.next_question(self.history, self.asked)

    def answer(self, evidence, value, unknown=False):
        if self.reason or evidence != self.pending:
            raise ValueError('This question is no longer current. Start a new consultation if needed.')
        encoded = None if unknown else self.engine.encode(evidence, value)
        if not unknown:
            self.history.append((evidence, encoded))
        self.asked.add(evidence)
        self.advance_after_answer(evidence, encoded, unknown)

    def advance_after_answer(self, evidence, encoded, unknown):
        # Add a turn before advancing so the question budget counts it.
        event = {'turn': len(self.turns) + 1, 'evidence_id': evidence,
                 'question': self.engine.question(evidence),
                 'answer': self.engine.display_answer(evidence, encoded) if not unknown else 'Unknown / skipped',
                 'encoded_answer': encoded, 'state_updated': not unknown}
        self.turns.append(event)
        self.advance()
        event['top_5'] = self.ranking

    def view(self):
        return {'created': self.created, 'initial_evidence': self.initial,
                'initial_question': self.engine.question(self.initial),
                'initial_answer': 'Yes', 'initial_top_5': self.initial_ranking,
                'turns': self.turns, 'top_5': self.ranking,
                'pending': self.engine.describe(self.pending) if self.pending else None,
                'stopped': self.reason is not None, 'stop_reason': self.reason,
                'max_turn': self.engine.max_turn}


class Engine:
    def __init__(self, args):
        import torch
        import numpy as np
        from website_model import DiagnosisModel
        self.torch = torch
        self.np = np
        self.model = DiagnosisModel(args.model)
        self.max_turn = self.model.budget if args.max_turn is None else args.max_turn
        if not 1 <= self.max_turn <= self.model.budget:
            raise ValueError('Use a question budget between 1 and the trained model budget.')
        self.device = torch.device('cpu')
        self.initials = self.model.features
        print('GDFS-inspired selector + XGBoost loaded on CPU', flush=True)

    def question(self, eid):
        return self.model.questions[eid]

    def describe(self, eid):
        return {'id': eid, 'text': self.question(eid), 'kind': 'binary',
                'choices': [], 'numbers': []}

    def encode(self, eid, value):
        if eid not in self.model.index or type(value) is not bool:
            raise ValueError('Choose Yes or No for a supported binary finding.')
        return int(value)

    def display_answer(self, eid, value):
        return 'Yes' if value else 'No'

    def predict(self, history):
        answers = dict(history)
        return self.model.respond(answers, history[0][0])['top_5']

    def next_question(self, history, asked):
        # Skipped findings remain hidden but cannot be asked again.
        t, np = self.torch, self.np
        n = len(self.model.features)
        values = np.zeros((1, n), dtype=np.float32)
        mask = np.zeros((1, n), dtype=np.float32)
        for name, value in history:
            index = self.model.index[name]
            values[0, index] = value
            mask[0, index] = 1
        blocked = np.zeros((1, n), dtype=bool)
        for name in asked:
            blocked[0, self.model.index[name]] = True
        if blocked.all():
            return None, 'no_remaining_questions'
        inputs = np.concatenate([values, mask], axis=1)
        with t.no_grad():
            scores = self.model.selector(t.from_numpy(inputs))
            scores = scores.masked_fill(t.from_numpy(blocked), float('-inf'))
            index = scores.argmax(dim=1).item()
        return self.model.features[index], None


def make_handler(engine):
    sessions = {}
    lock = threading.Lock()
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'binary_web.html'), 'rb') as f:
        html = f.read()

    class Handler(BaseHTTPRequestHandler):
        def reply(self, value, status=200, content_type='application/json; charset=utf-8'):
            raw = value if isinstance(value, bytes) else json.dumps(value, ensure_ascii=False).encode('utf-8')
            self.send_response(status)
            self.send_header('Content-Type', content_type)
            self.send_header('Content-Length', str(len(raw)))
            self.send_header('Cache-Control', 'no-store')
            self.send_header('X-Content-Type-Options', 'nosniff')
            self.end_headers()
            self.wfile.write(raw)

        def do_GET(self):
            path = urlparse(self.path).path
            if path == '/': return self.reply(html, content_type='text/html; charset=utf-8')
            if path == '/api/meta':
                return self.reply({'device': str(engine.device), 'initials': [engine.describe(e) for e in engine.initials]})
            self.reply({'error': 'Not found'}, 404)

        def do_POST(self):
            # Bind to loopback and accept only requests from this same web origin.
            origin = self.headers.get('Origin')
            if origin and origin != 'http://' + self.headers.get('Host', ''):
                return self.reply({'error': 'Origin rejected'}, 403)
            try:
                length = int(self.headers.get('Content-Length', 0))
                if not 0 < length <= 65536: raise ValueError('Invalid request size')
                body = json.loads(self.rfile.read(length))
                with lock:
                    if self.path == '/api/start':
                        if body.get('initial') not in engine.initials: raise ValueError('Choose an initial symptom')
                        if len(sessions) >= 100: raise ValueError('Restart the server to clear old sessions')
                        session = Consultation(engine, body['initial'])
                        session.advance()
                        sid = secrets.token_urlsafe(24)
                        sessions[sid] = session
                    else:
                        sid = body.get('session')
                        if sid not in sessions: raise ValueError('Session expired; start a new consultation')
                        session = sessions[sid]
                        if self.path == '/api/answer':
                            session.answer(body.get('evidence'), body.get('value'), body.get('unknown') is True)
                        elif self.path == '/api/stop':
                            session.reason = 'user_stopped'
                            session.pending = None
                        else:
                            return self.reply({'error': 'Not found'}, 404)
                    self.reply({'session': sid, 'consultation': session.view()})
            except (ValueError, KeyError, TypeError) as exc:
                self.reply({'error': str(exc)}, 400)
            except Exception as exc:
                import traceback
                traceback.print_exc()
                self.reply({'error': str(exc)}, 500)
    return Handler


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default=os.path.join(os.path.dirname(os.path.abspath(__file__))))
    parser.add_argument('--max-turn', type=int, default=None)
    parser.add_argument('--port', type=int, default=8000)
    args = parser.parse_args()
    engine = Engine(args)
    server = ThreadingHTTPServer(('127.0.0.1', args.port), make_handler(engine))
    print('Open http://127.0.0.1:%d  (Ctrl+C to stop)' % args.port, flush=True)
    try: server.serve_forever()
    except KeyboardInterrupt: pass
    finally: server.server_close()


if __name__ == '__main__':
    main()
