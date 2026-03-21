# nintendo-metacritic-review-prediction

Computerlinguistisches Experiment, das im Rahmen des Seminars **„Künstliche Intelligenz und die (digitale) Gesellschaft“** im SM1 an der Universität zu Köln durchgeführt wurde.

Thema: **Automatische Vorhersage von Metacritic-Nutzerbewertungen anhand von Rezensionstexten**

Im Zentrum der Untersuchung steht die automatische Vorhersage von Nutzerbewertungen anhand von Rezensionstexten. Hierbei wurde zur besseren Vergleichbarkeit der Datensatz auf Videospiele eines einzelnen Publishers, nämlich Nintendo, beschränkt.

Es wurden dabei zwei Ansätze verfolgt und verglichen. Zum einen wurde ein vortrainiertes [Transformer-Modell](https://huggingface.co/FacebookAI/roberta-base) für die spezifische Aufgabe der Bewertungsprädiktion finegetuned. Zum anderen wurde ein [Large Language Model](https://developers.openai.com/api/docs/models/gpt-oss-120b) genutzt dessen Vorhersagen durch  Prompting erzeugt wurden. Dabei wurde das Modell Openai GPT OSS 120B verwendet, welches von [ki-connect-nrw](https://chat.kiconnect.nrw/app) bereitgestellt wurde.

Die Vorhersage der Bewertungen erfolgte auf einer Skala von 0 bis 4, wobei die einzelnen Klassen folgende Bedeutungen annehmen:

| Rating | Bedeutung         |
| --|------------------------|
| 0 | sehr schlecht          |
| 1 | schlecht               |
| 2 | durchschnittlich       |
| 3 | gut                    |   
| 4 | sehr gut               |

Als Datengrundlage dienten insgesamt 1200 Nutzerrezensionen von Metacritic von zehn verschiedenen Nintendo Spielen.

Die Modelle sind in diesem Projekt nicht enthalten. Diese kann man über den untenstehenden Link herunterladen.


## Setup
Das Projekt wurde mit Python 3.12 erstellt.

1. Herunterladen der Modelle von **Sciebo**
   
**Download-Link Modell ohne Metadaten**: https://uni-koeln.sciebo.de/s/ceNm2CGQg8wCsxK

**Download-Link Modell mit Metadaten**: https://uni-koeln.sciebo.de/s/FTLFfHRL6jcwcZJ

2. Ordnerstruktur herstellen -> Entpacken der Files

Die heruntergeladenen Ordner müssen in den Root Folder des Projektverzeichnisses eingefügt werden. Die Struktur sollte wie folgt aussehen:
<img width="1502" height="837" alt="Screenshot 2026-03-16 172318" src="https://github.com/user-attachments/assets/dab78a93-b96a-4af6-8498-f9dcf4b30e9c" />

### Preprocessing (nur wenn neue Daten erzeugt werden sollen)

1. Starten des Skripts **scraper.js** zum Scrapen der Daten von Metacritic. Vorher muss die Variable **GameID** gesetzt werden und ausgewählt werden, ob Metadaten gescraped werden sollen oder nicht.

<img width="201" height="18" alt="image" src="https://github.com/user-attachments/assets/feb213e0-bca5-4932-8ed9-d4f8ea134299" />


| GameID | Games                                |
| --|-------------------------------------------|
| 0 | Mario Kart World                          |
| 1 | Animal Crossing: New Horizons             |
| 2 | The Legend of Zelda: Breath of the Wild   |
| 3 | Pokemon Legends: Z-A                      |
| 4 | Nintendo Switch Sports                    |
| 5 | The Legend of Zelda: Tears of the Kingdom |        
| 6 | Pokemon Scarlet                           |
| 7 | Paper Mario: Sticker Star                 |
| 8 | Super Mario Party                         |
| 9 | Super Smash Bros. Ultimate                |

2. Starten des Python Skripts **attachJSON.py**, um die JSON-Files der einzelnen Spiele aneinanderzuketten.
   
### Training (nur wenn neue Modelle erzeugt werden sollen)
Starten der Python Skripte **train.py** und/oder **train_metadata.py**, um die Modelle für Daten mit/ohne Metadaten zu trainieren.

### Prediction (obligatorisch)
#### Transformer
Starten der Python Skripte **predict.py** und/oder **predict_metadata.py** im Ordner **transformer-prediction**, um die Vorhersagen für Daten mit/ohne Metadaten durchzuführen.

#### LLM
1. **.env.template** zu **.env** kopieren und eigenen API-Key einsetzen.
2. Starten des Python Skripts **predict.py** im Ordner **llm-prediction**, um die Vorhersagen für Daten mit/ohne Metadaten durchzuführen.

Bei Fragen oder Feedback:
📧 nhalft@smail.uni-koeln.de
