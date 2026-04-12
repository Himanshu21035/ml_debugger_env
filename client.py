
import requests
from typing import Optional
import time
class MLDebuggerClient:
    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip("/")


    def reset(self, task_id: str, retries: int = 3) -> dict:
        for attempt in range(retries):
            try:
                r = requests.post(f"{self.base_url}/reset",
                                json={"task_id": task_id}, timeout=60)
                r.raise_for_status()
                return r.json()
            except requests.exceptions.RequestException as e:
                if attempt < retries - 1:
                    print(f"[CLIENT] Reset failed (attempt {attempt+1}/{retries}), retrying in 5s...")
                    time.sleep(5)
                else:
                    raise

    def step(self, action_type: str,
             parameter: Optional[str] = None,
             reasoning: Optional[str] = None) -> dict:
        r = requests.post(f"{self.base_url}/step", json={
            "action_type": action_type,
            "parameter":   parameter,
            "reasoning":   reasoning,
        }, timeout=60)
        r.raise_for_status()
        return r.json()

    def state(self) -> dict:
        r = requests.get(f"{self.base_url}/state", timeout=30)
        r.raise_for_status()
        return r.json()

    def health(self) -> dict:
        r = requests.get(f"{self.base_url}/health", timeout=10)
        r.raise_for_status()
        return r.json()