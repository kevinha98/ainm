# AINM — NM i AI 2026

> Vårt bidrag til NM i AI 2026, arrangert av [Astar Technologies](https://app.ainm.no).
> Repoet dekker tre oppgaver: regnskapsagent, objektgjenkjenning og kartprediksjon.

---

## Innhold

- [Oppgaver](#oppgaver)
  - [Accounting Agent](#accounting-agent)
  - [Object Detection](#object-detection)
  - [Astar Island](#astar-island)
- [Kom i gang](#kom-i-gang)
- [Teknologi](#teknologi)
- [Treningsartefakter](#treningsartefakter)
- [Lenker](#lenker)
- [Lisens](#lisens)

---

## Oppgaver

### Accounting Agent

AI-agent som løser regnskapsoppgaver i Tripletex automatisk.

| Egenskap | Detalj |
| --- | --- |
| **Hva** | LLM-drevet agent som leser en oppgave, kaller Tripletex API og fører regnskap |
| **Modell** | Gemini 2.5 Flash |
| **Kjøremiljø** | Google Cloud Run (FastAPI + Uvicorn) |
| **Kode** | [`accounting-agent/`](accounting-agent/) |

Agenten får en oppgavebeskrivelse og Tripletex-tilgang, og bruker opp til 15 iterasjoner for å løse oppgaven. Den håndterer faktura, leverandør, bilag, bankavstemming, prosjekt, ansatt og mer.

### Object Detection

Gjenkjenning av dagligvarer på butikkhyller for NorgesGruppen.

| Egenskap | Detalj |
| --- | --- |
| **Hva** | Finner og klassifiserer produkter i butikkbilder |
| **Modell** | YOLOv8m + YOLOv8l + YOLO11m ensemble |
| **Teknikk** | Multi-skala TTA, WBF, soft-NMS, kategori-voting |
| **Kode** | [`object-detection/`](object-detection/) |

Tre modeller kjøres på flere oppløsninger med test-time augmentation. Resultatene kombineres med Weighted Boxes Fusion og soft-NMS for best mulig presisjon.

### Astar Island

Prediksjon av sivilisasjonsplassering på et ukjent kart.

| Egenskap | Detalj |
| --- | --- |
| **Hva** | Observerer kartområder og predikerer sannsynligheter per rute |
| **Teknikk** | Spatial kernel smoothing, terrengmodell, bayesiansk kalibrering |
| **Kode** | [`astar-island/`](astar-island/) |

Oppgaven gir et begrenset antall observasjoner per runde. Vi bruker romlig utjevning og terrengdata til å bygge sannsynlighetskart for fem seed-varianter.

---

## Kom i gang

```bash
git clone https://github.com/kevinha98/ainm.git
cd ainm
git lfs install
git lfs pull
```

Hvert delprosjekt har egne avhengigheter:

```bash
# Accounting Agent
cd accounting-agent
pip install -r requirements.txt

# Object Detection
cd object-detection
pip install ultralytics onnxruntime ensemble-boxes

# Astar Island
cd astar-island
pip install -r requirements.txt
```

> **Merk:** API-nøkler settes som miljøvariabler, ikke i kode.

---

## Teknologi

| Verktøy | Brukt til |
| --- | --- |
| Python 3.11+ | Alle tre oppgavene |
| Gemini 2.5 Flash | LLM for regnskapsagenten |
| YOLOv8 / YOLO11 | Objektgjenkjenning |
| FastAPI | Webserver for agenten |
| Google Cloud Run | Hosting av agenten |
| GCP VM (T4 GPU) | Trening av deteksjonsmodeller |
| Git LFS | Versjonering av store modellfiler |

---

## Treningsartefakter

Modellvekter og treningsdata fra GCP ligger i repoet slik at alt kan verifiseres:

| Fil | Størrelse | Beskrivelse |
| --- | --- | --- |
| `object-detection/artifacts/weights/train_v6f_best.pt` | 122 MB | PyTorch-vekter |
| `object-detection/artifacts/weights/train_v6f_best.onnx` | 82 MB | ONNX-eksport |
| `object-detection/artifacts/weights/train_v6f_results.csv` | 8 KB | Treningsmetrikker |
| `object-detection/artifacts/data/NM_NGD_coco_dataset.zip` | 896 MB | Treningsdata (COCO-format) |

Disse filene spores med **Git LFS**. Kjør `git lfs pull` etter kloning for å hente dem.

---

## Lenker

| Ressurs | Lenke |
| --- | --- |
| Plattform | [app.ainm.no](https://app.ainm.no) |
| Leaderboard | [app.ainm.no/leaderboard](https://app.ainm.no/leaderboard) |
| Regler | [app.ainm.no/rules](https://app.ainm.no/rules) |
| Tripletex-docs | [app.ainm.no/docs/tripletex/overview](https://app.ainm.no/docs/tripletex/overview) |

---

## Lisens

Laget for NM i AI 2026. Ikke ment for videre distribusjon.
