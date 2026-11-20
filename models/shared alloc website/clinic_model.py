"""Adapter for the current clean XGBoost + 416/128/208 MLP checkpoints."""
import json
import os
import numpy as np


class Engine:
    def __init__(self, args):
        import torch
        from torch import nn
        from xgboost import XGBClassifier
        self.torch = torch
        with open(os.path.join(args.model, 'config.json'), encoding='utf-8') as f:
            config = json.load(f)
        with open(args.metadata, encoding='utf-8-sig') as f:
            definitions = json.load(f)
        self.features = config['features']
        self.diseases = config['diseases']
        self.index = {name: i for i, name in enumerate(self.features)}
        self.questions = {}
        for name in self.features:
            definition = definitions[name]
            self.questions[name] = definition.get('question_en', name) if isinstance(definition, dict) else str(definition)
        self.max_turn = config['budget'] if args.max_turn is None else args.max_turn
        if not 1 <= self.max_turn <= min(config['budget'], len(self.features)-1):
            raise ValueError('Question cap must stay within the saved training budget.')
        self.initials = self.features
        self.device = 'cpu'
        self.predictor = XGBClassifier()
        self.predictor.load_model(os.path.join(args.model, 'clean_xgb.json'))
        self.predictor.set_params(device='cpu', n_jobs=4)
        n = len(self.features)
        self.selector = nn.Sequential(nn.Linear(2*n,128),nn.ReLU(),nn.Linear(128,n))
        self.selector.load_state_dict(torch.load(os.path.join(args.model,'clean_selector.pt'), map_location='cpu', weights_only=True))
        self.selector.eval()

    def question(self, eid):
        return self.questions[eid]

    def describe(self, eid):
        return {'id':eid,'text':self.question(eid),'kind':'binary','choices':[],'numbers':[]}

    def encode(self, eid, value):
        if eid not in self.index or type(value) is not bool:
            raise ValueError('Choose Yes or No for a known binary question.')
        return int(value)

    def display_answer(self, eid, value):
        return 'Yes' if value else 'No'

    def infer(self, answers, asked=None):
        n = len(self.features)
        values = np.zeros((1,n), dtype=np.float32)
        observed = np.zeros((1,n), dtype=np.float32)
        for name,value in answers.items():
            column = self.index[name]
            values[0,column] = value
            observed[0,column] = 1
        inputs = np.concatenate([values,observed],axis=1)
        p = self.predictor.predict_proba(inputs)[0]
        with self.torch.no_grad():
            scores = self.selector(self.torch.from_numpy(inputs)).numpy()[0]
        blocked = set(answers) if asked is None else set(asked)
        for name in blocked:
            scores[self.index[name]] = -np.inf
        q = int(scores.argmax()) if np.isfinite(scores).any() else None
        top = np.argsort(p)[::-1][:5]
        ranking = [{'disease':self.diseases[int(self.predictor.classes_[i])], 'probability':float(p[i])} for i in top]
        return {'top_5':ranking, 'entropy':float(-(p*np.log(np.clip(p,1e-8,1))).sum()),
                'question':self.features[q] if q is not None else None,
                'gain':float(scores[q]) if q is not None else None}

    def predict(self, history):
        return self.infer(dict(history))['top_5']

    def next_question(self, history, asked):
        q = self.infer(dict(history),asked)['question']
        return q, None if q else 'no_remaining_questions'
