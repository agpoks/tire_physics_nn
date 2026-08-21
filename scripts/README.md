# scripts/

Dataset acquisition and analytical fitting helpers.

**Nothing large downloads automatically.** Each `download_<name>.py` either fetches a
small subset or prints the exact manual steps and exits non-zero. This is deliberate:
several of the target datasets sit behind a repository landing page, a licence
acceptance or an author request, and a script that silently half-downloads one is worse
than a script that tells you what to do.

| script | dataset | status |
|---|---|---|
| `download_kit.py` | KIT inner-drum tire force transmission | verified URL, licence acceptance required → manual |
| `download_vetyt.py` | VeTyT bicycle tyre (Politecnico di Milano) | request from the authors → manual |
| `download_tum_cargo_bike.py` | TUM cargo bicycle tire | **source unverified** |
| `download_deep_dynamics.py` | Deep Dynamics (BayesRace + IAC) | public git repository → manual clone |
| `download_roboracer.py` | RoboRacer model-structured NN | **source unverified** |
| `download_qmotion.py` | Q-Motion pressure variation | **source unverified** |
| `download_f1_stints.py` | Formula 1 stint data (lap time, compound, tyre age, weather) | **works out of the box** — `pip install fastf1`, no API key |
| `fit_magic_formula.py` | — | fits the analytical baseline with `scipy` |

Every script prints its dataset's **type label** (real measurement / simulated / game
telemetry / synthetic), because that label has to travel with the data into any results
table.

After downloading, load through the adapters:

```python
from tire_nn.data import adapters
df = adapters.load("kit", root="data/raw")
```
