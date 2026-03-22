# AINM – konkurranserepo (NM i AI 2026)

Dette repoet samler arbeid på tvers av flere konkurranseoppgaver i NM i AI 2026 (app.ainm.no), inkludert:

- **AI Accounting Agent** (Tripletex)
- **Astar Island**
- **Object Detection**

Basert på offentlig innhold fra plattformen er «AI Accounting Agent» en innsending av `/solve`-endepunkt som evalueres på tvers av oppgaver, med leaderboard/rangering og oppgavebasert scoring.

## Repo-struktur

- `accounting-agent/` – agent og API-integrasjon for Tripletex-oppgaver
- `astar-island/` – kode for Astar Island-oppgaven (analyse, trening, innsending)
- `object-detection/` – objektgjenkjenning, trening, eksport, evaluering

## Object Detection – treningsartefakter

For at dommere skal kunne verifisere modellgrunnlaget, er artefakter hentet fra GCP-treningsmiljø og lagt inn i repoet:

- `object-detection/artifacts/weights/train_v6f_best.pt`
- `object-detection/artifacts/weights/train_v6f_best.onnx`
- `object-detection/artifacts/weights/train_v6f_results.csv`
- `object-detection/artifacts/data/NM_NGD_coco_dataset.zip`

Store filer er versjonert med **Git LFS**.

## Krav for å bruke repoet

- Python 3.11+ (avhengig av delprosjekt)
- Git LFS installert for å hente/store modellfiler
- Eventuelle API-nøkler settes via miljøvariabler (ikke hardkodet)

## Hurtigstart

```bash
git clone https://github.com/kevinha98/ainm.git
cd ainm
git lfs install
git lfs pull
```

Deretter kan du gå inn i ønsket delprosjekt (`accounting-agent`, `astar-island` eller `object-detection`) og kjøre prosjektspesifikke skript.

## Konkurransekontekst

Plattform: [app.ainm.no](https://app.ainm.no)  
Arrangør: Astar Technologies AS

Nyttige lenker (fra offentlig plattforminnhold):

- Dashboard: [app.ainm.no/dashboard](https://app.ainm.no/dashboard)
- Leaderboard: [app.ainm.no/leaderboard](https://app.ainm.no/leaderboard)
- Oppgaver: [app.ainm.no/tasks](https://app.ainm.no/tasks)
- Regler: [app.ainm.no/rules](https://app.ainm.no/rules)
- Tripletex docs: [app.ainm.no/docs/tripletex/overview](https://app.ainm.no/docs/tripletex/overview)
