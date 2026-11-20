"""Local two-page consultation website using the current trained clean models."""
import argparse
import json
import os
import secrets
import threading
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse
from cohort import Cohort, PatientBank


class Consultation:
    def __init__(self, engine, initial, initial_value=1, reference=None):
        self.engine = engine
        self.initial = initial
        self.initial_value = initial_value
        self.reference = reference
        self.history = [(initial, initial_value)]
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
                'initial_answer': self.engine.display_answer(self.initial,self.initial_value),
                'reference_case': self.reference, 'initial_top_5': self.initial_ranking,
                'turns': self.turns, 'top_5': self.ranking,
                'pending': self.engine.describe(self.pending) if self.pending else None,
                'stopped': self.reason is not None, 'stop_reason': self.reason,
                'max_turn': self.engine.max_turn}


def make_handler(engine, bank):
    singles, cohorts = {}, {}
    examples = []
    lock = threading.Lock()
    folder = os.path.dirname(os.path.abspath(__file__))
    pages = {}
    for route,name in [('/', 'binary_web.html'),('/cohort','cohort.html')]:
        with open(os.path.join(folder,name),'rb') as f:
            pages[route] = f.read()

    def get_examples():
        if not examples:
            for i,case in enumerate(bank.sample(5,42)):
                positive = []
                for name in engine.features:
                    if case['answers'].get(name,0) == 1:
                        positive.append({'id':name,'text':engine.question(name)})
                examples.append({'id':i,'name':'Patient %02d'%(i+1),
                                 'initial':case['initial'],
                                 'initial_value':case['answers'].get(case['initial'],1),
                                 'initial_question':engine.question(case['initial']),
                                 'positive_findings':positive,'reference':case['reference'],
                                 'record_id':case['record_id']})
        return examples

    class Handler(BaseHTTPRequestHandler):
        def reply(self, value, status=200, html=False):
            raw = value if isinstance(value,bytes) else json.dumps(value,ensure_ascii=False).encode('utf-8')
            self.send_response(status)
            self.send_header('Content-Type','text/html; charset=utf-8' if html else 'application/json; charset=utf-8')
            self.send_header('Content-Length',str(len(raw)))
            self.send_header('Cache-Control','no-store')
            self.send_header('X-Content-Type-Options','nosniff')
            self.end_headers()
            self.wfile.write(raw)

        def do_GET(self):
            path = urlparse(self.path).path
            if path in pages:
                return self.reply(pages[path],html=True)
            if path == '/api/examples':
                try:
                    with lock:
                        return self.reply({'patients':get_examples()})
                except (ValueError,KeyError,TypeError,OSError) as exc:
                    return self.reply({'error':str(exc)},400)
            if path == '/api/meta':
                return self.reply({'device':'cpu','max_turn':engine.max_turn,
                                   'initials':[engine.describe(e) for e in engine.initials],
                                   'cohort_data_configured':bool(bank.folder)})
            self.reply({'error':'Not found'},404)

        def do_POST(self):
            origin = self.headers.get('Origin')
            if origin and origin != 'http://' + self.headers.get('Host',''):
                return self.reply({'error':'Origin rejected'},403)
            try:
                length = int(self.headers.get('Content-Length',0))
                if not 0 < length <= 65536:
                    raise ValueError('Invalid request size.')
                body = json.loads(self.rfile.read(length))
                path = urlparse(self.path).path
                with lock:
                    if path == '/api/cohort/start':
                        if len(cohorts)>=20:
                            raise ValueError('Restart the server to clear old cohort sessions.')
                        count = int(body.get('patients',6))
                        average = int(body.get('average',10))
                        cap = int(body.get('cap',engine.max_turn))
                        minimum = int(body.get('minimum',2))
                        seed = int(body.get('seed',42))
                        if not 2 <= count <= 12:
                            raise ValueError('Choose 2 to 12 patients.')
                        session = Cohort(engine, bank.sample(count,seed),body.get('policy','minimum_uncertainty'),average,cap,minimum,seed)
                        sid = secrets.token_urlsafe(24)
                        cohorts[sid] = session
                        return self.reply({'session':sid,'cohort':session.view()})
                    if path == '/api/cohort/answer':
                        sid = body.get('session')
                        if sid not in cohorts:
                            raise ValueError('Cohort expired; start a new cohort.')
                        session = cohorts[sid]
                        event = session.answer_next(body.get('expected_turn'))
                        return self.reply({'session':sid,'cohort':session.view(),'event':event})
                    if path == '/api/start':
                        reference = None
                        initial_value = 1
                        if 'example' in body:
                            choice = body['example']
                            if type(choice) is not int or not 0 <= choice < 5:
                                raise ValueError('Choose one of the five example patients.')
                            reference = get_examples()[choice]
                            initial = reference['initial']
                            initial_value = reference['initial_value']
                        else:
                            initial = body.get('initial')
                        if initial not in engine.initials:
                            raise ValueError('Choose an initial finding.')
                        if len(singles)>=100:
                            raise ValueError('Restart the server to clear old sessions.')
                        # The reference sheet is stored for the UI only. Only the
                        # initial finding enters the inference history at start.
                        session = Consultation(engine,initial,initial_value,reference)
                        session.advance()
                        sid = secrets.token_urlsafe(24)
                        singles[sid] = session
                    elif path in ['/api/answer','/api/stop']:
                        sid = body.get('session')
                        if sid not in singles:
                            raise ValueError('Session expired; start again.')
                        session = singles[sid]
                        if path == '/api/answer':
                            session.answer(body.get('evidence'),body.get('value'),body.get('unknown') is True)
                        else:
                            session.reason, session.pending = 'user_stopped',None
                    else:
                        return self.reply({'error':'Not found'},404)
                    self.reply({'session':sid,'consultation':session.view()})
            except (ValueError,KeyError,TypeError) as exc:
                self.reply({'error':str(exc)},400)
            except Exception as exc:
                import traceback
                traceback.print_exc()
                self.reply({'error':str(exc)},500)
    return Handler


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model',required=True,help='Current clean model folder')
    parser.add_argument('--metadata',required=True,help='Original release_evidences.json')
    parser.add_argument('--data',default=None,help='Converted data folder with test_data.pkl for cohort replay')
    parser.add_argument('--max-turn',type=int,default=None)
    parser.add_argument('--port',type=int,default=8000)
    args = parser.parse_args()
    from clinic_model import Engine
    engine = Engine(args)
    bank = PatientBank(args.data,engine.features)
    server = ThreadingHTTPServer(('127.0.0.1',args.port),make_handler(engine,bank))
    print('Single patient: http://127.0.0.1:%d/' % args.port,flush=True)
    print('Shared clinic: http://127.0.0.1:%d/cohort' % args.port,flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__=='__main__':
    main()
