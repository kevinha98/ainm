# NM i AI 2026 — Konkurranserepo

**Lag:** Proptonomy.ai
**Konkurranse:** [NM i AI 2026](https://app.ainm.no) (19.–22. mars 2026), arrangert av Astar Technologies  

---

## Innhold

- [Oppgave 1 — Objektgjenkjenning](#oppgave-1--objektgjenkjenning-norgesgruppen)
- [Oppgave 2 — Regnskapsagent](#oppgave-2--regnskapsagent-tripletex)
- [Oppgave 3 — Kartprediksjon](#oppgave-3--kartprediksjon-astar-island)
- [Kom i gang](#kom-i-gang)
- [Teknologi](#teknologi)
- [Treningsartefakter](#treningsartefakter)
- [Lenker](#lenker)

---

## Oppgave 1 — Objektgjenkjenning (NorgesGruppen)

**Oppgave:** Finn og klassifiser alle produkter på bilder av butikkhyller.  
**Metrikk:** 0.7 × detection\_mAP\@0.5 + 0.3 × classification\_mAP\@0.5  
**Beste score:** 0.9256 (3. plass)

### Hvordan det fungerer

Tenk deg at du tar et bilde av en butikkhylle. Systemet skal finne *hvert eneste produkt* i bildet og si hva det er — melk, brød, Grandiosa, osv.

Vi bruker tre ulike bildemodeller (YOLO-familien) som hver ser på bildene i forskjellige oppløsninger. Noen ser "stort bilde" for å fange hele hyller, andre ser "nærbilde" for å fange små produkter.

Resultatene fra alle modellene kombineres: der flere modeller er enige om at et produkt finnes, øker vi tryggheten. Der bare én modell ser noe, senker vi den. Denne teknikken kalles *Weighted Boxes Fusion* (WBF) og gir bedre presisjon enn noen enkeltmodell alene.

### Tilnærming

- **Tre-modells ensemble:** YOLOv8m (100 MB), YOLOv8l (168 MB) og YOLO11m (78 MB)
- **Multi-skala inferens:** Hvert bilde kjøres på 1280, 1408 og 1536 piksler — totalt 12 gjennomkjøringer per bilde
- **Test-time augmentation (TTA):** Bilder flippes horisontalt for ekstra prediksjoner
- **Soft-NMS:** Fjerner duplikater med myk nedvekting i stedet for hard fjerning
- **Kategori-voting:** Når modellene er uenige om produkttype, vinner klassen med høyest samlet score

### Trening

Modellene ble trent på GCP VM med T4 GPU. Hovedmodellen (YOLOv8m v3) trente i 300 epoker med aggressiv augmentering — copy-paste (30%), sterk skalering, label smoothing og dropout. Tidligstopp ved epoke 188.

**Kode:** [`object-detection/`](object-detection/)

---

## Oppgave 2 — Regnskapsagent (Tripletex)

**Oppgave:** Motta regnskapsoppgaver på 6 språk og utfør korrekte API-kall mot Tripletex.  
**Metrikk:** Korrekthet × effektivitetsbonus (færre kall = høyere score, feil = trekk)  
**Beste score:** ~72.5% gjennomsnitt, flere perfekte 13/13

### Idé

Systemet fungerer som en AI-regnskapsfører. Det mottar en oppgave i naturlig språk — for eksempel *"Opprett en faktura til kunde X på 5000 kr med 25% mva"* — og utfører alle stegene i Tripletex automatisk, akkurat som en regnskapsfører ville gjort.

En stor språkmodell (Gemini 2.5 Flash) leser oppgaven, bestemmer hva som må gjøres, og kaller Tripletex sitt API steg for steg: opprette kunde, legge inn produkt, sende faktura, osv. Etter hvert API-kall leser modellen svaret og bestemmer neste steg.

Systemet håndterer alt fra enkel fakturering til komplekse oppgaver som årsoppgjør og bankavstemming.

### Arkitektur

- **Ren LLM-drevet agent:** Ingen hardkodet logikk — all strategi ligger i et stort systemprompt (~77 000 tegn) som fungerer som en komplett Tripletex-oppslagsverk
- **Iterativ løkke:** Opp til 20 iterasjoner per oppgave. Agenten bestemmer selv når den er ferdig
- **10+ interceptorer:** Blokkerer kjente feilruter (f.eks. inngående faktura → 403) og ruter automatisk til alternativer
- **Pre-caching:** Henter ~500 kontoer, mva-typer og betalingstyper *før* oppgaven starter, så agenten slipper å lete
- **Loop-deteksjon:** Oppdager om agenten gjentar de samme kallene og tvinger den ut av løkken
- **Flersspråklig:** Oppgaver kom på norsk, engelsk, tysk, fransk, spansk og portugisisk

### Erfaringer

Den viktigste lærdommen var at **oppgavespesifikke instruksjoner i systemprompt** slo alle andre tilnærminger. Vi prøvde å bygge abstraksjonslag og verktøy-moduler, men LLM-en resonnerer bedre med eksplisitte spesifikasjoner enn med hjelpefunksjoner.

**Kode:** [`accounting-agent/`](accounting-agent/)

---

## Oppgave 3 — Kartprediksjon (Astar Island)

**Oppgave:** Prediker sannsynligheten for hva som finnes i hver rute på et 40×40 kart (hav, skog, fjell, bosetting, havn, ruin) — med bare 50 observasjoner.  
**Metrikk:** score = 100 × exp(−3 × vektet\_KL), høyentropiceller teller mest  
**Beste score:** 80.07 (runde 5, rank 20/144)

### Konsept

Tenk deg et kart med 1600 ruter, og du får bare kikke på 50 av dem. Basert på det du ser, skal du gjette hva som finnes på de 1550 rutene du *ikke* har sett.

Systemet bruker seks ulike prediksjonsmodeller som angriper problemet fra forskjellige vinkler. Noen ser på naboruter ("skog ligger ofte ved fjell"), andre simulerer hvordan en sivilisasjon ville spredd seg over kartet, og en maskinlæringsmodell (HistGradientBoosting) lærer mønstre fra tidligere runder.

Alle seks modellers prediksjoner blandes sammen til ett felles sannsynlighetskart, som deretter kalibreres og skjerpes.

### Seks-modells ensemble

Seks modeller i ensemble:

1. **Markov-modell:** Ser på overgangssannsynligheter mellom terrengtyper — "hva er naboen til skog?"
2. **Monte Carlo-simulering:** Simulerer 2000 mulige kart og tar gjennomsnittet
3. **HGB (HistGradientBoosting):** Maskinlæringsmodell trent på 24+ romlige features (nabolag, avstand til bosetting, kantplassering)
4. **MRF (Markov Random Field):** Optimaliserer prediksjoner slik at naboruter henger sammen
5. **CA-modell (Cellular Automaton):** Simulerer bosettingsspredning over 50 tidssteg
6. **Direkte observasjon:** Bruker observerte ruter som fasit og interpolerer

Modellene vektes basert på historisk ytelse, og resultatet kalibreres mot kjente fordelinger.

### Observasjonsstrategi

Med bare 50 spørringer fordelt på 5 kartseeds bruker vi strategisk viewport-plassering — 9 synsfelt per seed gir ~full dekning, med 5 resterende spørringer rettet mot usikre områder.

**Kode:** [`astar-island/`](astar-island/)

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
# Objektgjenkjenning
cd object-detection
pip install ultralytics onnxruntime ensemble-boxes

# Regnskapsagent
cd accounting-agent
pip install -r requirements.txt

# Kartprediksjon
cd astar-island
pip install -r requirements.txt
```

> **Merk:** API-nøkler settes som miljøvariabler, ikke i kode.

---

## Teknologi

| Verktøy | Brukt til |
| --- | --- |
| Python 3.11+ | Alle tre oppgavene |
| Gemini 2.5 Flash | Språkmodell for regnskapsagenten |
| YOLOv8 / YOLO11 | Objektgjenkjenning |
| scikit-learn, scipy | Prediksjonsmodeller for kartoppgaven |
| FastAPI + Uvicorn | Webserver for agenten |
| Google Cloud Run | Hosting av agenten |
| GCP VM (T4 GPU) | Trening av bildemodeller |
| Git LFS | Store modellfiler i git |

---

## Treningsartefakter

Modellvekter og treningsdata fra GCP, slik at alt kan verifiseres:

| Fil | Størrelse | Beskrivelse |
| --- | --- | --- |
| `object-detection/artifacts/weights/train_v6f_best.pt` | 122 MB | PyTorch-vekter |
| `object-detection/artifacts/weights/train_v6f_best.onnx` | 82 MB | ONNX-eksport |
| `object-detection/artifacts/weights/train_v6f_results.csv` | 8 KB | Treningsmetrikker |
| `object-detection/artifacts/data/NM_NGD_coco_dataset.zip` | 896 MB | Treningsdata (COCO-format) |

Filene spores med **Git LFS**. Kjør `git lfs pull` etter kloning.

---

## Lenker

| Ressurs | Lenke |
| --- | --- |
| Plattform | [app.ainm.no](https://app.ainm.no) |
| Leaderboard | [app.ainm.no/leaderboard](https://app.ainm.no/leaderboard) |
| Regler | [app.ainm.no/rules](https://app.ainm.no/rules) |
| Tripletex-docs | [app.ainm.no/docs/tripletex/overview](https://app.ainm.no/docs/tripletex/overview) |
