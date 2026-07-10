# GraphNAMLSS

**Interpretierbare distributionelle Regression für graphstrukturierte Beobachtungen**

GraphNAMLSS verbindet Neural Additive Models for Location, Scale and Shape (NAMLSS) mit graphbasierten Repräsentationen. Das Ziel ist nicht nur eine bessere Punktschätzung, sondern eine kalibrierte prädiktive Verteilung, deren lokale Kovariateneffekte und graphabhängige Beiträge getrennt untersucht werden können.

## Motivation und Forschungsfrage

Distributionelle Regressionsmodelle beschreiben mehrere Parameter einer bedingten Verteilung und können damit Heteroskedastizität, Prognoseintervalle und Tail-Verhalten modellieren. Additive Modelle machen die Beiträge einzelner Kovariaten sichtbar, behandeln Beobachtungen jedoch typischerweise als bedingt unabhängig. Diese Annahme ist bei Verkehrsnetzen, räumlichen Gesundheitsdaten oder anderen relationalen Systemen unplausibel.

Graph Neural Networks nutzen Nachbarschaftsstruktur, liefern aber häufig nur Punktschätzungen oder schwer zerlegbare distributionelle Köpfe. GraphNAMLSS adressiert diese Lücke mit der Leitfrage:

> Verbessert Graphinformation die Qualität und Kalibrierung der vorhergesagten Verteilung über ein starkes lokales NAMLSS hinaus – und wirkt sie unterschiedlich auf Lage und Streuung?

Die methodische Hypothese lautet, dass lokale additive Effekte und strukturelle Graphkontexte komplementäre Information tragen. Besonders relevant ist, ob Nachbarschaftszustände nicht nur den Erwartungswert, sondern auch die konditionale Unsicherheit erklären.

## Modellierung

Für eine Beobachtung am Knoten $i$ und Zeitpunkt $t$ sei

$$
y_{it}\mid \mathbf{x}_{it},G \sim
\mathcal{N}(\mu_{it},\sigma_{it}^{2}).
$$

Jeder Verteilungsparameter $k\in\{\mu,\sigma\}$ besitzt einen eigenen additiven Prädiktor:

$$
\eta_{k,it}
= \beta_{k0}
+ \sum_{j=1}^{p}\tilde f_{kj}(x_{j,it})
+ \alpha_k\,\tilde g_{k}(i,t;G).
$$

Dabei bezeichnet $\tilde f_{kj}$ eine zentrierte lokale Shape Function, $\tilde g_k$ einen zentrierten graphbasierten Beitrag und $\alpha_k$ ein parameterspezifisches Graph Contribution Gate. Die Zentrierung verhindert, dass lokale oder graphbasierte Terme den Interzept absorbieren. Für das gaußsche Location-Scale-Modell gelten

$$
\mu_{it}=\eta_{\mu,it},
\qquad
\sigma_{it}=\operatorname{softplus}(\eta_{\sigma,it})+\varepsilon.
$$

Die Zerlegung ist die zentrale wissenschaftliche Eigenschaft des Modells: lokale Shape-Plots beschreiben kovariatenspezifische Effekte, Graphkarten zeigen räumlich-strukturelle Beiträge, und die Gates dienen als Diagnostik der parameterspezifischen Graphnutzung. Gate-Größen sind dabei weder kausale Attributionen noch alleiniger Nachweis eines empirischen Vorteils.

GraphNAMLSS bezeichnet eine Modellfamilie. Die vollständige Untersuchung variiert drei Faktoren in einem $2\times2\times2$-Design:

| Faktor | Ausprägungen |
|---|---|
| Graphrepräsentation | geteilter Encoder / parameterspezifische Encoder |
| Graphinjektion | additiver Graphbeitrag / graph-aware Shape Functions |
| Self-Feature-Zugriff | neighbor-focused / self-inclusive |

Der erste Tracer-Bullet beschränkt sich auf einen geteilten, additiven und nachbarschaftsorientierten Graphstrom. Die konzeptionelle Zielarchitektur verwendet GraphSAGE-artiges Message Passing. Der gegenwärtige Concept Demonstrator nutzt dafür bewusst eine leichte deterministische Approximation: gewichtete Nachbarschaftsaggregation, lokale lineare Shape Functions, Residuen-Fit für den Location-Graphkopf und einen Gauß-NLL-basierten Fit für den Scale-Graphkopf. Die Gates entstehen durch eine Unit-L2-Faktorisierung der fitted Graphkoeffizienten und sind noch nicht end-to-end optimizer-trainiert.

## Experimentelle Methodik

### Benchmark und Beobachtungseinheit

Der erste empirische Prüfstein ist **METR-LA**. Eine Beobachtung entspricht einem Sensor zu einem Vorhersagezeitpunkt mit lokalen Verkehrslags, Nachbarschaftslags und einem Zielwert 15 Minuten in der Zukunft. Das physische, gerichtete Verkehrsnetz und seine bereitgestellten Kantengewichte bleiben über die Zeit fest.

Die lokalen Kovariaten umfassen den aktuellen Messwert, Lags von 5, 10, 15, 30 und 60 Minuten sowie Tageszeit-, Wochentags- und Wochenendmerkmale. Der Graphstrom erhält zunächst Verkehrsinformationen aus der Nachbarschaft; Kalendermerkmale verbleiben im lokalen Strom, damit der strukturelle Beitrag inhaltlich abgrenzbar bleibt.

Die Daten werden chronologisch in 70 % Training, 10 % Validierung und 20 % Test aufgeteilt. Skalierungsparameter werden ausschließlich auf dem Trainingsblock geschätzt. Fehlende Zielwerte werden aus Loss und Metriken maskiert; Kovariaten dürfen nicht mit zukünftiger Information imputiert werden. Das Testset bleibt bis zur abschließenden Evaluation unangetastet.

### Vergleichsmodelle

Die Kernvergleiche erfüllen unterschiedliche methodische Rollen:

- **Deterministisches NAMLSS:** prüft den Mehrwert der Graphinformation gegenüber lokaler additiver distributioneller Regression.
- **Distributional GNN Baseline:** prüft, ob die additive Zerlegung gegenüber einem nicht-additiven Graphmodell mit direkten Verteilungsköpfen wissenschaftlichen Mehrwert bietet.
- **Point GNN Reference:** dient ausschließlich als Guardrail für MAE und RMSE, nicht als distributioneller Peer.
- **GAMLSS/BAMLSS und Graph-Feature-Varianten:** sind geplante statistische Baselines. Einfache Nachbarschaftssummaries sollen prüfen, ob eine gelernte Graphrepräsentation über handkonstruierte Graphmerkmale hinausgeht.

Neben einer einheitlichen Trainingsrezeptur ist ein grob parameterzahl-gematchter Vergleich vorgesehen, damit zusätzliche Modellkapazität nicht mit einem strukturellen Vorteil verwechselt wird.

### Zielgrößen und Entscheidungsregel

Primäre Zielgrößen sind Negative Log-Likelihood und CRPS sowie Intervallabdeckung und PIT-Diagnostik. MAE und RMSE sind Guardrails für die Punktprognose. Ein niedrigerer RMSE bei verschlechterter Kalibrierung gilt nicht als Erfolg der Methode.

Der **METR-LA Merit Probe** ist das erste Real-Data-Evidenzgate. Verglichen werden ein deterministisches NAMLSS und ein optimizer-trainiertes GraphNAMLSS; die übrigen Modelle liefern Kontext und Guardrails. Nur ein belastbares Signal in NLL oder CRPS bei akzeptabler Coverage, PIT, MAE und RMSE rechtfertigt die nächste Investitionsstufe. Bei einem unklaren Resultat folgt zunächst Diagnose von Daten, Baseline, Fit, Graphstrom, Scale-Prädiktor und Horizont – nicht automatisch eine Ausweitung des Projekts.

PEMS-BAY dient erst nach einem vielversprechenden METR-LA-Ergebnis als Replikationsgate. Bayesianische Graphgewichte, explizite strukturelle Unsicherheit und räumliche Gesundheitsanwendungen sind nachgelagerte Forschungsachsen.

## Wissenschaftlicher Beitrag

Der erwartete Beitrag liegt in drei Punkten:

1. eine distributionelle Graphregression, die Lage und Streuung separat mit Graphinformation anreichert;
2. eine explizite Zerlegung lokaler, graphbasierter und parameterspezifisch gegateter Effekte;
3. ein kontrolliertes Ablationsdesign, das Graphrepräsentation, Injektionsstrategie und Self-Feature-Zugriff voneinander trennt.

Damit ist die relevante Frage nicht nur, *ob* ein Graphmodell besser prognostiziert, sondern *welcher Teil der vorhergesagten Verteilung* von Graphstruktur profitiert und wie transparent dieser Effekt bleibt.

## Aktueller Evidenzstatus

Das Repository implementiert einen vollständigen synthetischen Concept-Demo-Pfad mit deterministischem NAMLSS, Tracer-Bullet GraphNAMLSS, Distributional GNN Baseline, Point GNN Reference, Artefakten und Bericht. Er validiert Datenfluss, Modellzerlegung, Likelihood, Metriken und Interpretierbarkeitsausgaben. Er ist keine empirische Bestätigung der Forschungshypothese.

Die kanonische Evidenzgrenze lautet: **Synthetic demo metrics are sanity checks; they are not evidence that GraphNAMLSS outperforms baselines.** Der Concept Demonstrator verwendet einen **lightweight deterministic Tracer-Bullet Fit** und ist **not the METR-LA Merit Probe**; dieser Fit **does not satisfy the Merit Probe Training Standard**.

The Current Concept Demonstrator does not implement Bayesian graph streams, structural uncertainty experiments, or medical application results. GAMLSS and BAMLSS remain future Statistical Baseline integrations through External Code Adapters; they are not completed Concept Demo Bundle requirements.

## Guided Demo Path

Nach Installation erzeugt ein Befehl das vollständige Demo-Bundle:

```bash
graphnamlss demo --output outputs/concept-demo
```

Der primäre Bericht liegt anschließend unter `outputs/concept-demo/report/tracer_bullet_evaluation_report.md`. Empfohlene Lesereihenfolge:

1. **Model Decomposition** – Zusammenspiel von lokalem und graphbasiertem Prädiktor.
2. **Local Shape Contributions** – `local_shape_mu.svg` und `local_shape_sigma.svg`.
3. **Graph Contributions** – `graph_contribution_mu.svg`, `graph_contribution_sigma.svg` und `graph_contributions.json`.
4. **Graph Gate Diagnostics** – `graph_gates.json` und die Grenzen ihrer Interpretation.
5. **Metrics** – splitweise Werte in den jeweiligen `metrics.json`-Dateien.
6. **Concept Demo Evidence Boundary** – zulässige und unzulässige Schlussfolgerungen.

Das Tutorial `notebooks/01_graphnamlss_concept_walkthrough.ipynb` führt dieselbe Argumentation schrittweise aus und verwendet gespeicherte Paketartefakte, statt Modelllogik zu duplizieren.

## Reproduzierbarkeit

### Plain Pyproject Workflow

Vorausgesetzt wird Python 3.12:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
python -m pytest
```

PyTorch Geometric ist für die explizite PyG-Umgebung optional installierbar:

```bash
python -m pip install -e ".[dev,pyg]"
```

Plattformspezifische CPU- oder CUDA-Wheels für PyTorch sollten vor dem Editable Install eingerichtet werden. Für wissenschaftliche Läufe sind Python-, PyTorch-, PyTorch-Geometric- und Accelerator-Versionen gemeinsam mit der aufgelösten Konfiguration zu archivieren.

### Synthetic Traffic Fixture

`graphnamlss.testing.load_synthetic_traffic_fixture()` stellt deterministische **test data** für schnelle Pipeline- und Modelltests bereit. Das Fixture enthält mehrere Sensoren, zeitliche Lags, Nachbarschaftsinformationen und Missingness-Fälle, ist aber **not a scientific benchmark**.

### PEMS-BAY Replication Dataset

Die Replikationskonfiguration liegt in `configs/pems_bay.yaml`. Erwartet werden die lokalen Artefakte `data/raw/pems-bay/pems-bay.h5` und `data/raw/pems-bay/adj_mx_bay.pkl`. Die Replikation verwendet denselben Split-, Skalierungs-, Missingness- und Graphvertrag wie METR-LA und wird erst nach dem Merit Probe aktiviert.

## Weiterführende Projektdokumente

- `RESEARCH_PLAN.md` – ausführlicher Forschungs- und Entwicklungsplan
- `ARCHITECTURE.md` – technische Komponenten und Experimentverträge
- `CONTEXT.md` – kontrolliertes Vokabular und Begriffsabgrenzungen
- `configs/metr_la_merit_probe.yaml` – maschinenlesbarer Evidenzvertrag
- `configs/primary_experiment_matrix.yaml` – vollständiges $2\times2\times2$-Design
- `docs/adr/` – dokumentierte Architektur- und Sequenzierungsentscheidungen
