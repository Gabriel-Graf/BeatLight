# KI gesteuerte Lichttechnik mit Genre Klassifikation - BeatLight

## Einleitung

BeatLight wurde mit dem Ziel entwickelt, Lichttechnikern die Arbeit zu erleichtern, insbesondere in dynamischen
Veranstaltungssituationen mit mehreren DJs und Tanzflächen. Während sich DJs bei einer Veranstaltung regelmäßig
abwechseln, bleibt die Steuerung der Lichttechnik meist durchgehend in der Verantwortung eines einzelnen Operators.
Dies wird besonders dann zum Problem, wenn mehrere Tanzflächen gleichzeitig betreut werden müssen, die technische
Ausstattung ist zwar häufig vorhanden, jedoch fehlt es an ausreichendem Personal zur kontinuierlichen Steuerung. Gerade
bei elektronischen Musikveranstaltungen werden mehrere teils ähnliche oder ganz unterschiedliche Musikgenre pro
Tanzfläche gespielt oder werden nur grob auf mehrere Tanzflächen ver-teilt. Genau hier setzt BeatLight an: Als
intelligentes Tool erkennt es automatisch das aktuell gespielte Genre und ermöglicht eine automatische Anpassung der
Lichtstimmung, passend zur Musikrichtung. Hierfür müssen lediglich für alle Genre die erkannt werden sollen, passende
Lichtstimmungen erstellt werden. BeatLight verarbeitet dazu ein gewöhnliches Audiosignal (z. B. über AUX-Eingang),
klassifiziert die Musik mittels eines vortrainierten KI-Modells und übergibt die Genre-Informationen an die vorhandene
Lichtsteuerungssoftware. Lichttechniker haben dabei jederzeit die Möglichkeit, diese Klassifizierung zu überprüfen oder
manuell zu überschreiben. So muss der Lichttechniker (VJ) nur zu den Hauptspielzeiten aktiv eine Lichtshow steuern,
während die Lichtsteuerung mit BeatLight den Rest der Zeit automatisiert ablaufen kann. Hierfür möchten wir
Resolume, eins in der Lichttechnik weit verbreitete Software zur Steuerung großer Lichtsysteme als Einstiegspunkt
nutzten. Wichtig ist: BeatLight ist keine eigene Lichtsteuerungssoftware, sondern ein ergänzendes Tool zur
automatisierten Genre-Erkennung, das sich in bestehende Systeme integrieren lässt. Langfristig gedacht wäre eine
dedizierte KI basierte Lichtsteuerung denkbar.

## Vision

Das langfristige Ziel von BeatLight ist es, die kreative Arbeit von Lichttechnikern zu unterstützen, zu
inspirieren und breiteren Nutzergruppen zugänglich zu machen. Durch eine intelligente Verknüpfung von Musik und Licht
sollen einfache, aber wirkungsvolle Lichteffekte automatisch zur passenden Zeit ausgelöst werden. Kurzfristig
fokussieren wir uns auf genre basierte Preset-Steuerung, mittelfristig auf eine intelligente Automatisierung der
gesamten
Lichtatmosphäre.

## Projektstatus

Das Projekt befindet sich derzeit in der Konzeptionsphase. Der erste Prototyp ist schon funktionsfähig auch mit einem
kleinen User-Interface. Damit soll demonstriert werden, was BeatLight theoretic alles erreichen kann. Um dieses Projekt
zu benutzten müssen lediglich einige wenige Installationsschritte befolgt werden. Van da läuft alles automatisch.

## Python Installation

Installieren aller Abhängigkeiten:

   ```bash
   pip install -r requirements.txt
   ```

## Resolume Arena Installation

Resolume Arena wird in der Veranstaltungsszene häufig benutzt, da es Preiswert und sehr mächtig ist. Es bietet für
unseren Use-Case sogar eine WebSocket API an mit der man sogut wie jeden maus Klick in der Software fernsteuern kann.
Ein Überblick über die gesamte API findet sich hier: https://resolume.com/docs/restapi/

Resolume Arena & Wire muss installiert sein, um BeatLight zu nutzen. Die Software ist kostenpflichtig, aber kann auch
ohne probleme
kostenlos genutzt werden. Unter folgendem Link kann die Software heruntergeladen werden:
https://resolume.com/download/.

Um Resolume richtig einzustellen, muss der WebSocket Server aktiviert werden. Dies
kann unter Arena → Einstellungen → WebSocket Server aktiviert werden. Der Standardport ist auf der
IP http://10.2.0.2:8080 erreichbar. Danach muss das ``Resolume_Demo_Projekt.avc`` geladen werden, diese Datei liegt dem
Projekt bei.

<img src="/assets/Resolume_Webserver.png" alt="Resolume Webserver Einstellungen" width="600">

### GPU Unterstützung einrichten

Zur Nutzung von BeatLight ist keine GPU Unterstützung notwendig, da das vortrainierte Modell bereits mitgeliefert ist
siehe den Ordner ``saved_model/``.
Falls das Trainieren eines eigenen Models z.B. mit dem GTZAN Dataset gewünscht ist, wird empfohlen PyTorch mit CUDA
Unterstützung zu installieren. Dazu kann die offizielle Anleitung von PyTorch genutzt werden:
https://pytorch.org/get-started/locally/, https://developer.nvidia.com/cuda-12-9-0-download-archive

## Ordnerstruktur

- In Businessplan_Finanzplan befinden sich die Excel-Dateien für den Businessplan und Finanzplan. Also die Abgaben
  für diese Hausarbeit.
- Unter ``Datensatz`` befindet sich der handselektierte Datensatz, der für das Training des Modells genutzt wurde. Zur
  demonstrations zwecken haben wir uns für 7 Genre aus dem Bereich Rock und Metal entschieden:
    - 'Rock 'n' Roll'
    - 'Hard Rock'
    - 'Heavy Metal'
    - 'Trash Metal'
    - 'Metalcore'
    - 'Death Metal'
    - 'Grindcore'
- Im src-Ordner befinden sich die Python-Skripte, die für die Funktionalität von BeatLight verantwortlich sind.
    - In `src/Main_without_UI.py` befindet sich ein einfacher Use-Case nur um zu demonstrieren wie diese Technologie mit
      einander kombiniert werden kann.
    - In ``src/Main_with_UI.py`` befindet sich das User-Interface, welches die Steuerung von BeatLight
      ermöglicht. Hier kann der Nutzer grafisch die visuellen Effekte einstellen und wie BeatLight diese ansteuert.
    - Im ``src/Training.p``y befindet sich der Code zum Trainieren des Modells. Hierbei wurde eine eigen CNN architektur
      entworfen, die auf einen ebenfalls selbst erstellten Datensatz trainiert wurde.
    - Die restlichen Dateien im src-Ordner sind Hilfsklassen, die zum Training und dem aufzeichen vom audiosignal
      benötigt werden.

# BeatLight UI 

<img src="/assets/BeatLight UI Erklärung.png" alt="BeatLight UI Erklärung" width="600">

Zur Demonstration der Funktionalität von BeatLight wurde ein einfaches User-Interface entwickelt. Dieses ermöglicht es
dem Nutzer, verschiedene visuelle Effekte zu konfigurieren und sie mit dem passenden Genre zu verknüpfen. BeatLight
läuft dabei im Hintergrund und aktualisiert die Lichtstimmung automatisch in einem in der UI konfigurierten Intervall.

# Demo Video

Ein Video sagt mehr als tausend Worte. Hier ist eine kurze Demo von BeatLight in Aktion:

<video height="500" controls>
  <source src="assets/BeatLight%20Demo.mp4" type="video/mp4">
</video>


