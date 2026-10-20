"""New namespace via supported MLflow APIs; never alter existing runs/configuration."""
from __future__ import annotations
import json
import math
import os
import time
from pathlib import Path
from urllib.request import urlopen
from mlflow.tracking import MlflowClient
from mlflow.entities import Metric, Param


class Tracker:
    def __init__(self, root, config, name):
        root = Path(root).resolve()
        experiment_name = config.get("experiment", "Universal Cognitive Lab")
        if experiment_name != "Universal Cognitive Lab":
            raise ValueError("This laboratory must use its own MLflow experiment namespace")
        uri = config.get("uri", "http://127.0.0.1:5000")
        self.fallback = False
        try:
            with urlopen(uri.rstrip("/") + "/health", timeout=2) as response:
                if response.status != 200:
                    raise RuntimeError("MLflow health check failed")
        except Exception:
            self.fallback = True
            os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"
            uri = (root / "runs/mlruns-isolated").as_uri()
        self.uri = uri
        self.client = MlflowClient(tracking_uri=uri)
        experiment = self.client.get_experiment_by_name(experiment_name)
        self.experiment_id = experiment.experiment_id if experiment else self.client.create_experiment(
            experiment_name, artifact_location=(root / "runs/mlartifacts").as_uri())
        self.run_id = self.client.create_run(self.experiment_id, tags={"mlflow.runName": name,
            "laboratory": "Universal Cognitive Lab", "claim": "Infrastructure smoke; no superiority claim"}).info.run_id

    def params(self, values):
        parameters = [Param(key, json.dumps(value, sort_keys=True) if isinstance(value,(dict,list)) else str(value))
                      for key, value in values.items()]
        self.client.log_batch(self.run_id, params=parameters)

    def metrics(self, values, step=0, prefix=""):
        timestamp = int(time.time() * 1000)
        metrics = [Metric(prefix + key, float(value), timestamp, int(step))
                   for key, value in values.items()
                   if isinstance(value,(int,float)) and not isinstance(value,bool) and math.isfinite(value)]
        if metrics:
            self.client.log_batch(self.run_id, metrics=metrics)

    def finish(self, directory, status="FINISHED"):
        try:
            self.client.log_artifacts(self.run_id, str(directory))
        except Exception:
            self.client.set_terminated(self.run_id, status="FAILED")
            raise
        self.client.set_terminated(self.run_id, status=status)

    def reference(self):
        return {"uri":self.uri,"experiment_id":self.experiment_id,"run_id":self.run_id,
                "isolated_fallback":self.fallback,
                "url":f"{self.uri}/#/experiments/{self.experiment_id}/runs/{self.run_id}" if not self.fallback else None}
