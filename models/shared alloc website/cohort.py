"""Sequential shared-budget simulation. Reference labels never select actions."""
import pickle
import random
import os
import numpy as np

POLICIES = {'equal','gain','uncertainty','minimum_uncertainty'}


class PatientBank:
    def __init__(self, folder, features):
        self.folder = folder
        self.features = set(features)
        self.records = None
        self.eligible = None

    def sample(self, count, seed):
        if not self.folder:
            raise ValueError('Start the server with --data pointing to your converted DDXPlus folder.')
        if self.records is None:
            with open(os.path.join(self.folder,'test_data.pkl'),'rb') as f:
                self.records = pickle.load(f)['test']
            self.eligible = []
            for i,p in enumerate(self.records):
                explicit = p['goal']['explicit_inform_slots']
                if len(explicit)==1 and list(explicit)[0] in self.features:
                    self.eligible.append(i)
        if count > len(self.eligible):
            raise ValueError('Not enough eligible patients in test_data.pkl.')
        ids = random.Random(seed).sample(self.eligible,count)
        cases = []
        for i in ids:
            p = self.records[i]
            explicit = p['goal']['explicit_inform_slots']
            findings = dict(explicit)
            findings.update(p['goal']['implicit_inform_slots'])
            answers = {name:int(slot['option']) for name,slot in findings.items() if name in self.features}
            cases.append({'initial':list(explicit)[0], 'answers':answers,
                          'reference':p['disease_tag'],'record_id':i})
        return cases


class Cohort:
    def __init__(self, engine, cases, policy, average, cap, minimum, seed):
        if policy not in POLICIES:
            raise ValueError('Choose a supported allocation policy.')
        if not 1 <= average <= cap <= engine.max_turn:
            raise ValueError('Use 1 <= average <= cap <= trained model cap.')
        if not 0 <= minimum <= average:
            raise ValueError('Minimum must be between zero and the average budget.')
        if not 2 <= len(cases) <= 12:
            raise ValueError('Use 2 to 12 patients for this visualization.')
        self.engine, self.cases = engine,cases
        self.policy, self.average, self.cap, self.minimum = policy,average,cap,minimum
        self.seed = seed
        self.total = len(cases)*average
        self.used = 0
        self.counts = [0]*len(cases)
        self.answers = []
        self.states = []
        self.events = []
        self.initial_states = []
        self.ties = np.random.default_rng(seed).random(len(cases)).tolist()
        for case in cases:
            given = {case['initial']:case['answers'].get(case['initial'],1)}
            self.answers.append(given)
            state = engine.infer(given)
            self.states.append(state)
            self.initial_states.append(state)

    def phase(self):
        if self.used >= self.total:
            return 'complete'
        if self.policy == 'minimum_uncertainty' and min(self.counts) < self.minimum:
            return 'minimum'
        return self.policy

    def choose(self):
        if self.used >= self.total:
            return None,None
        eligible = [i for i in range(len(self.cases)) if self.counts[i]<self.cap and self.states[i]['question'] is not None]
        if not eligible:
            return None,None
        if self.policy == 'equal':
            i = min(eligible,key=lambda i:(self.counts[i],i))
            reason = 'Equal allocation: next patient in the questioning round.'
        elif self.policy == 'minimum_uncertainty' and min(self.counts)<self.minimum:
            i = min(eligible,key=lambda i:(self.counts[i],i))
            reason = 'Guaranteed follow-up %d of %d. Everyone completes this stage before uncertainty allocation.' % (self.counts[i]+1,self.minimum)
        else:
            metric = 'gain' if self.policy == 'gain' else 'entropy'
            i = max(eligible,key=lambda i:(self.states[i][metric],-self.ties[i]))
            reason = 'Largest predicted next-question gain.' if metric=='gain' else 'Highest current diagnostic entropy.'
        return i,reason

    def answer_next(self, expected_turn):
        if type(expected_turn) is not int or expected_turn != self.used:
            raise ValueError('The consultation changed. Refresh this cohort before continuing.')
        i,reason = self.choose()
        if i is None:
            raise ValueError('This cohort is complete.')
        before = self.states[i]
        phase = self.phase()
        q = before['question']
        # Simulator looks up the response; inference only sees revealed answers.
        answer = self.cases[i]['answers'].get(q,0)
        self.answers[i][q] = answer
        self.counts[i] += 1
        self.used += 1
        self.states[i] = self.engine.infer(self.answers[i])
        event = {'turn':self.used,'patient':i,'patient_name':'Patient %02d'%(i+1),
                 'evidence_id':q,'question':self.engine.question(q),'answer':bool(answer),
                 'reason':reason,'phase':phase,'follow_up':self.counts[i],
                 'entropy_before':before['entropy'],
                 'gain_before':before['gain'],'top_5':self.states[i]['top_5']}
        self.events.append(event)
        return event

    def view(self):
        i,reason = self.choose()
        patients = []
        for j,case in enumerate(self.cases):
            state = self.states[j]
            guaranteed = [{'number':e['follow_up'],'question':e['question'],
                           'evidence_id':e['evidence_id'],'answer':e['answer']}
                          for e in self.events if e['patient']==j and e['phase']=='minimum']
            patients.append({'id':j,'name':'Patient %02d'%(j+1),'questions':self.counts[j],
                             'initial':case['initial'],'initial_question':self.engine.question(case['initial']),
                             'top_5':state['top_5'],'entropy':state['entropy'],'gain':state['gain'],
                             'at_cap':self.counts[j]>=self.cap,'reference':case['reference'],
                             'guaranteed_questions':guaranteed})
        pending = None
        if i is not None:
            pending = {'patient':i,'patient_name':patients[i]['name'],
                       'evidence_id':self.states[i]['question'],
                       'question':self.engine.question(self.states[i]['question']),
                       'reason':reason,'entropy':self.states[i]['entropy'],'gain':self.states[i]['gain'],
                       'phase':self.phase(),'follow_up':self.counts[i]+1}
        return {'policy':self.policy,'average':self.average,'cap':self.cap,'minimum':self.minimum,
                'seed':self.seed,'total':self.total,'used':self.used,'remaining':self.total-self.used,
                'phase':self.phase(),
                'minimum_used':sum(min(n,self.minimum) for n in self.counts) if self.policy=='minimum_uncertainty' else 0,
                'minimum_total':len(self.cases)*self.minimum if self.policy=='minimum_uncertainty' else 0,
                'patients':patients,'pending':pending,'events':self.events,'complete':pending is None,
                'correct':sum(s['top_5'][0]['disease']==case['reference'] for s,case in zip(self.states,self.cases))}
