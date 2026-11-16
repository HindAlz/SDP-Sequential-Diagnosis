import os
import json
import numpy as np
import torch
from torch import nn
from xgboost import XGBClassifier


class DiagnosisModel:
    def __init__(self, folder):
        with open(os.path.join(folder, "schema.json"), encoding="utf-8") as f:
            schema = json.load(f)
        questions_path = os.path.join(folder, "questions.json")
        if os.path.isfile(questions_path):
            with open(questions_path, encoding="utf-8") as f:
                self.questions = json.load(f)
        else:
            with open(os.path.join(folder, "release_evidences.json"), encoding="utf-8-sig") as f:
                definitions = json.load(f)
            self.questions = {}
            for name in schema["features"]:
                self.questions[name] = definitions[name]["question_en"]
        self.features = schema["features"]
        self.diseases = schema["diseases"]
        self.budget = schema["budget"]
        self.index = {}
        for i, name in enumerate(self.features):
            self.index[name] = i

        self.predictor = XGBClassifier()
        self.predictor.load_model(os.path.join(folder, "xgboost.json"))
        self.predictor.set_params(device="cpu")
        count = len(self.features)
        self.selector = nn.Sequential(
            nn.Linear(count * 2, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(), nn.Linear(128, count))
        saved = torch.load(os.path.join(folder, "selector.pt"),
                           map_location="cpu", weights_only=True)
        self.selector.load_state_dict(saved["state_dict"])
        self.selector.eval()

    @torch.no_grad()
    def respond(self, answers, initial_id):
        # answers contains only revealed answers, including the initial finding.
        if initial_id not in answers:
            raise ValueError("Include the initial finding and its answer.")
        count = len(self.features)
        values = np.zeros((1, count), dtype=np.float32)
        mask = np.zeros((1, count), dtype=bool)
        for name, answer in answers.items():
            if name not in self.index or answer not in (0, 1):
                raise ValueError("Use known binary evidence IDs and 0/1 answers.")
            column = self.index[name]
            values[0, column] = answer
            mask[0, column] = True
        inputs = np.concatenate([values, mask.astype(np.float32)], axis=1)
        probabilities = self.predictor.predict_proba(inputs)[0]
        positions = probabilities.argsort()[::-1][:5]
        top_5 = []
        for position in positions:
            label = int(self.predictor.classes_[position])
            top_5.append({"disease": self.diseases[label],
                          "probability": float(probabilities[position])})
        asked = len(answers) - 1
        stopped = asked >= self.budget or bool(mask.all())
        question = None
        if not stopped:
            scores = self.selector(torch.from_numpy(inputs))
            scores = scores.masked_fill(torch.from_numpy(mask), float("-inf"))
            column = scores.argmax(dim=1).item()
            name = self.features[column]
            question = {"evidence_id": name, "text": self.questions[name]}
        return {"follow_up_questions_answered": asked,
                "remaining_budget": max(0, self.budget - asked),
                "stopped": stopped, "next_question": question,
                "predicted_disease": top_5[0]["disease"], "top_5": top_5}
