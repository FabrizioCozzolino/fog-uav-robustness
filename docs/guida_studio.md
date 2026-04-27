# Guida allo studio del progetto

**Per**: ripasso personale ed esame orale
**Argomento**: semantic segmentation su VDD con U-Net, con focus su robustezza sotto nebbia GAN-generata
**Stato attuale**: Fase 0 e Fase 1 completate (setup + training pipeline funzionante)

---

## 0. Il quadro generale (da raccontare all'esame in 60 secondi)

> *"Il progetto consiste nel valutare la robustezza di un modello di semantic segmentation
> U-Net applicato a immagini aeree da drone (dataset VDD), quando queste immagini
> vengono degradate da nebbia sintetica generata da un GAN. La pipeline è divisa in
> cinque fasi: (1) addestramento della U-Net su VDD clean, (2) addestramento di un
> GAN che impara a trasformare immagini clean in immagini nebbiose a partire dal
> dataset Foggy Cityscapes, (3) applicazione del GAN a VDD per ottenere VDD_foggy,
> (4) test della U-Net originale su VDD_foggy per misurare il calo di performance,
> (5) re-training della U-Net su dati nebbiosi per verificare se la robustezza
> migliora."*

Questo è il "paragrafo di apertura" che ti serve saper ripetere a memoria.

---

## 1. Cos'è la semantic segmentation

### Concetto

La **semantic segmentation** assegna a **ogni pixel** di un'immagine un'etichetta di classe.
Non dice solo "c'è una macchina" (image classification) né "c'è una macchina nel rettangolo
X" (object detection), ma disegna esattamente il contorno pixel-per-pixel delle macchine,
delle strade, degli edifici, ecc.

Input:  immagine RGB di shape `(H, W, 3)`, valori 0-255
Output: mappa di etichette `(H, W)` dove ogni pixel è un intero `0..N-1` (N = numero classi)

Nel nostro caso N=7: `0=other, 1=wall, 2=road, 3=vegetation, 4=vehicle, 5=roof, 6=water`.

### Perché è difficile

- I bordi degli oggetti sono ambigui (dove finisce il tetto e comincia il muro?)
- Le dimensioni degli oggetti variano enormemente (un veicolo da 50 pixel vs una strada da 500 000 pixel)
- Serve sia contesto globale ("siamo in una scena urbana") sia dettaglio locale (bordi precisi)

### Cosa vuol dire "aerea da drone"

Le immagini non sono prese dal livello stradale, ma dall'alto (30°, 60°, 90° dal nadir).
Questo cambia completamente l'aspetto degli oggetti: un'auto vista dall'alto è un rettangolino,
non un profilo laterale. Per questo i modelli pretrained su ImageNet (dataset street-view)
non sono ideali — ma funzionano comunque come punto di partenza.

---

## 2. I dataset: che cosa sono e perché li abbiamo

### VDD (Varied Drone Dataset)

- 400 immagini aeree reali, 4000×3000 pixel, 7 classi
- Split già fatto dagli autori: 280 train / 80 val / 40 test (70/20/10)
- Formato:
  - immagini: `.JPG` RGB
  - maschere: `.png` grayscale dove il valore del pixel **è** il class ID (0..6)

### Foggy Cityscapes (il tuo dataset 500×3)

- 500 immagini **stradali** (non aeree, viste dall'auto) ripetute in 3 versioni:
  - `No_Fog`: immagine originale
  - `Medium_Fog`: nebbia media (visibilità ~300m)
  - `Dense_Fog`: nebbia densa (visibilità ~150m)
- È **paired**: lo stesso file `001.png` esiste in tutte e tre le cartelle ed è la stessa scena
- La nebbia è **sintetica**, prodotta applicando il modello fisico di scattering atmosferico
  alle immagini di Cityscapes (non è nebbia reale fotografata)

### Perché due dataset?

VDD non ha versioni nebbiose delle sue immagini. Allora la strategia è: *"imparo a mettere
nebbia in modo realistico da un altro dataset che però ha coppie clean/foggy, e poi applico
quella nebbia a VDD"*. Questo è **domain transfer** e il GAN è il motore che lo realizza.

### Cosa ti chiederanno all'esame

- *"Perché usate un dataset street-level per imparare la nebbia e poi la applicate ad
  immagini aeree?"* → Risposta: perché non esistono dataset aerei nebbiosi pubblici con
  coppie clean/foggy. È un limite metodologico dichiarato, e una delle domande aperte del
  progetto è proprio se la nebbia impari su scene stradali generalizzi bene a scene aeree.

- *"Il dataset nebbioso è 'realistico'?"* → È sintetico, generato dal modello di Koschmieder
  (scattering atmosferico) applicato ai depth map di Cityscapes. È convincente ad occhio ma
  non è identico alla nebbia reale. Il GAN aggiunge un ulteriore livello di astrazione.

---

## 3. PyTorch e i Dataset

### Cos'è PyTorch

Una libreria Python per deep learning. I suoi oggetti fondamentali sono:

- **Tensor**: come un numpy array ma può stare in GPU e sa calcolare i gradienti automaticamente
- **Module**: un pezzo di rete neurale (livello, modello intero)
- **Dataset**: classe che rappresenta una collezione di campioni
- **DataLoader**: carica campioni a batch dal Dataset, in modo parallelizzato

### Il protocollo Dataset in PyTorch

Per creare un dataset personalizzato serve solo una classe con **tre metodi**:

```python
class MioDataset(Dataset):
    def __init__(self, ...):
        # setup: leggo i percorsi dei file, NON li apro ancora
        self.img_paths = [...]
    def __len__(self):
        # quante immagini ho
        return len(self.img_paths)
    def __getitem__(self, idx):
        # PyTorch mi chiede l'elemento numero idx, io glielo restituisco
        return image, mask
```

PyTorch chiamerà `__getitem__` ogni volta che un DataLoader ha bisogno di un campione.
**Importante**: il Dataset non carica tutto in memoria (sarebbe impossibile con VDD =
4000×3000×400 immagini = molti GB). Apre il file solo quando viene richiesto.

### Cosa fa la nostra `VDDDataset` (file `src/datasets/vdd.py`)

1. In `__init__` costruisce la lista dei path delle immagini `.JPG` della cartella `src/`
   e verifica che per ognuna esista la maschera corrispondente in `gt/` (col nome base
   uguale ma estensione `.png`).
2. In `__getitem__(idx)`:
   - Legge l'immagine con `cv2.imread` (OpenCV restituisce BGR, noi la convertiamo in RGB)
   - Legge la maschera (è già grayscale con i class ID)
   - Se è stato passato un `transform` (Albumentations), lo applica
   - Altrimenti converte direttamente in tensore PyTorch

La maschera resta un tensore **intero** (`long`), non float, perché è una classe discreta
non un valore continuo.

### Dettaglio importante: perché BGR → RGB

OpenCV per motivi storici carica le immagini in ordine BGR (Blue-Green-Red) invece di RGB.
Se non convertiamo, le immagini sembrano "sbagliate" (rossi e blu invertiti) e il modello
pretrained (che si aspetta RGB) darà predizioni pessime. È un errore classico, ricordatelo.

---

## 4. Le augmentations (file `src/utils/transforms.py`)

### Cosa sono

Sono trasformazioni applicate al volo alle immagini di training per aumentare artificialmente
la varietà dei dati. Se ho 280 immagini, vedo virtualmente 280 × (variazioni possibili) = moltissime.

Aiutano contro l'**overfitting**: il modello non memorizza più le specifiche 280 immagini
ma impara i pattern generali.

### Le nostre scelte per il train

```python
A.Resize(512, 512)                    # ridimensiona a 512x512
A.HorizontalFlip(p=0.5)               # flip orizzontale con probabilità 50%
A.VerticalFlip(p=0.5)                 # flip verticale
A.RandomRotate90(p=0.5)               # ruota di 0/90/180/270 gradi
A.RandomBrightnessContrast(...)       # varia luminosità e contrasto
A.HueSaturationValue(...)             # varia tonalità e saturazione
A.Normalize(mean=IMAGENET, std=IMAGENET)  # normalizzazione
ToTensorV2()                          # converti in tensore PyTorch
```

### Perché queste e non altre

- **Flip orizzontale/verticale e RandomRotate90**: un'immagine aerea è naturalmente isotropa.
  Un drone può essere orientato in qualsiasi direzione, quindi ruotare l'immagine di 90°
  produce una vista perfettamente valida. Su immagini stradali non lo faresti (il cielo
  non può stare in basso).

- **Jitter di colori**: l'illuminazione varia molto nel corso della giornata. Insegnare al
  modello a essere invariante alla luminosità globale aiuta.

- **Niente scale/zoom o crop aggressivo**: avremmo potuto aggiungerli ma il resize standardizzato
  è già sufficiente per il primo baseline. Si possono introdurre dopo per spremere performance.

### Perché Normalize con statistiche di ImageNet

Il backbone ResNet-34 che useremo è stato addestrato su ImageNet, dove le immagini hanno una
certa distribuzione statistica (media e varianza per canale). Perché il modello pretrained
funzioni bene, le nostre immagini devono assomigliare statisticamente a quelle di ImageNet.
Quindi sottraiamo la media e dividiamo per lo std di ImageNet:

```
mean = (0.485, 0.456, 0.406)  # R, G, B
std  = (0.229, 0.224, 0.225)
```

Questo spiega perché nel DataLoader test abbiamo visto valori `[-2.12, 2.64]` dopo la
normalizzazione — non sono più [0,1] ma una distribuzione centrata sullo 0.

### Dettaglio: separazione train/val transforms

- **Train**: include augmentation casuali (HorizontalFlip, rotazioni, jitter)
- **Val / Test**: solo Resize + Normalize + ToTensor — **zero casualità**

Perché? La validazione deve essere **deterministica**: se valuti lo stesso modello sullo
stesso val set, devi avere la stessa metrica. Se applicassi augmentation casuali avresti
mIoU diverse ogni volta. Inoltre la valutazione deve misurare la capacità del modello sui
dati come sono, non su versioni augmentate.

---

## 5. L'architettura U-Net

### Idea generale

U-Net è una rete neurale a forma di "U" proposta nel 2015 per la segmentazione medica.
Ha due metà:

- **Encoder** (lato sinistro, discendente): riduce progressivamente la risoluzione spaziale
  (es. 512 → 256 → 128 → 64 → 32) e aumenta il numero di canali/feature (3 → 64 → 128 → 256 → 512).
  Cattura il **contesto globale**.

- **Decoder** (lato destro, ascendente): riporta la risoluzione all'originale (32 → 64 → 128 → 256 → 512)
  tramite upsampling, ricostruendo i dettagli spaziali.

- **Skip connections**: collegamenti diretti fra ogni livello dell'encoder e il corrispondente
  livello del decoder. Servono perché l'encoder perde informazione spaziale durante il
  downsampling: lo skip permette al decoder di recuperare quei dettagli.

Output finale: tensor `(B, N_classi, H, W)` — un "logit" per classe per pixel. Facendo
`argmax` lungo la dimensione delle classi ottieni la mappa di classi `(B, H, W)`.

### Encoder pretrained: ResNet-34

Invece di scrivere un encoder da zero usiamo **ResNet-34**, una rete convoluzionale classica
(2015, 34 layer) i cui pesi sono stati pre-addestrati su ImageNet (1M di immagini, 1000 classi).

Perché conviene:
- Parte da una "inizializzazione intelligente" invece che da pesi casuali
- Converge molto più velocemente (in poche epoch)
- Generalizza meglio, soprattutto con dataset piccoli (280 immagini di train sono poche!)

La libreria `segmentation_models_pytorch` ci dà U-Net con backbone a scelta in una riga:

```python
model = smp.Unet(encoder_name="resnet34", encoder_weights="imagenet", classes=7)
```

### Cosa ti chiederanno all'esame

- *"Perché U-Net e non DeepLabV3+ o SegFormer?"* → Il README del progetto la specifica come
  baseline. Inoltre è leggera, ben documentata, facile da training su dati piccoli.
- *"Perché pretrained su ImageNet se le scene sono aeree?"* → Anche se il dominio cambia,
  le feature low-level (edge, texture) che ResNet impara restano utili. È il principio del
  transfer learning.

---

## 6. Il training loop

Questa è la parte più importante da capire perché è dove succede il "vero" deep learning.

### Lo schema generale

Per ogni epoch:

```
FASE DI TRAIN:
    model.train()
    for ogni batch (images, masks) nel train_loader:
        optimizer.zero_grad()            # azzera i gradienti precedenti
        logits = model(images)            # forward pass: calcola le previsioni
        loss = criterion(logits, masks)   # confronta previsioni con verità
        loss.backward()                   # backward pass: calcola i gradienti
        optimizer.step()                  # aggiorna i pesi del modello

FASE DI VAL:
    model.eval()
    with torch.no_grad():                  # non serve calcolare gradienti
        for ogni batch nel val_loader:
            logits = model(images)
            metriche.update(logits, masks)
    results = metriche.compute()

    if val_mIoU > best_mIoU:
        torch.save(model.state_dict(), "best.pth")
```

### Cos'è la loss

Una **funzione di loss** misura quanto le predizioni del modello sono sbagliate rispetto
alla verità. Il training consiste nel minimizzare questa loss.

Per la segmentation usiamo **CrossEntropyLoss**, che è la scelta standard per problemi
multi-classe:

```
loss(logits, target) = -log(softmax(logits)[target])
```

In italiano: "quanto bassa è la probabilità assegnata dal modello alla classe corretta?".
Se il modello assegna probabilità 1.0 alla classe giusta, loss = 0. Se assegna 0, loss = +∞.

Nota: PyTorch vuole i **logits** (output raw del modello, non normalizzati), non le
probabilità già passate attraverso softmax. Lo fa internamente per stabilità numerica.

### Cos'è l'optimizer

Un algoritmo che aggiorna i pesi del modello in modo da ridurre la loss. Il più semplice è
SGD (Stochastic Gradient Descent):

```
nuovo_peso = vecchio_peso - learning_rate × gradiente
```

Noi usiamo **AdamW**, una variante più raffinata di SGD che:
- Adatta il learning rate ad ogni parametro in base alla storia dei gradienti
- Include weight decay (una forma di regolarizzazione)

Learning rate (`lr=1e-4`): quanto "grande" è il passo di aggiornamento. Troppo alto → la
loss oscilla o esplode. Troppo basso → training lentissimo.

### Il Learning Rate Scheduler

Man mano che il training procede, ridurre il lr aiuta a "rifinire" la convergenza. Noi
usiamo **Cosine Annealing**: il lr parte da 1e-4 e scende lungo una curva coseno fino a
~0 all'ultima epoch. È una scelta popolare e robusta.

### Gradient clipping

`clip_grad_norm_(params, 1.0)` limita la norma del gradiente a 1.0. Serve a prevenire
"explosioni" di gradiente che possono destabilizzare il training. In pratica è una
salvaguardia, usata di default.

### `model.train()` vs `model.eval()`

Alcuni layer (Dropout, BatchNorm) si comportano diversamente in training vs evaluation:
- Dropout: attivo in training (spegne neuroni a caso), inattivo in eval
- BatchNorm: in training usa statistiche del batch corrente; in eval usa statistiche medie
  accumulate durante il training

Se ti dimentichi `model.eval()` in validazione, le metriche sono scorrette. Errore molto
comune, ricordati.

### `torch.no_grad()`

In validazione non serve calcolare i gradienti (non faremo backward). Disabilitarli con
`with torch.no_grad():` risparmia molta memoria e velocizza il forward pass.

### Checkpoint: cosa salviamo

```python
torch.save({
    "epoch": epoch,
    "model_state_dict": model.state_dict(),       # pesi del modello
    "optimizer_state_dict": optimizer.state_dict(), # stato dell'optimizer (momenti, ecc.)
    "mIoU": best_miou,
    "args": vars(args),                           # parametri di training
}, "best.pth")
```

Salviamo sia `best.pth` (il miglior modello visto sul val set) che `last.pth` (l'ultimo).
Serve sempre tenere il "best" perché le ultime epoch non sono necessariamente le migliori
(il modello può peggiorare in overfitting).

---

## 7. Le metriche

### mIoU (mean Intersection over Union)

La metrica principale per la segmentation. Per una classe singola:

```
IoU = | predizione ∩ verità | / | predizione ∪ verità |
```

Esempio: il modello predice 100 pixel come "road", la verità dice che 120 sono "road", di
cui 80 coincidono col modello. Allora IoU = 80 / (100 + 120 - 80) = 80/140 ≈ 0.57.

**mIoU** = media degli IoU di tutte le classi. È una metrica bilanciata: una classe rara
pesa quanto una frequente.

### F1 score

Media armonica di precision e recall, per classe. Simile a mIoU ma formula diversa.

```
precision = TP / (TP + FP)   # dei pixel che ho detto "road", quanti sono davvero road?
recall    = TP / (TP + FN)   # dei pixel che sono davvero road, quanti ho identificato?
F1        = 2 × precision × recall / (precision + recall)
```

### Accuracy pixel-wise

La percentuale di pixel correttamente classificati. È una metrica ingenua: se il 90% dell'
immagine è "vegetation" e il modello predice sempre "vegetation", l'accuracy è 90% ma il
modello è inutile. **Per questo si guarda principalmente mIoU, non accuracy**, nei problemi
di segmentation.

### IoU per classe

Calcoliamo anche l'IoU di ogni singola classe. Ci aspettiamo che classi frequenti
(vegetation, road) abbiano IoU alto e classi rare (water, vehicle) IoU basso. Se una classe
è costantemente 0, significa che il modello non l'ha mai imparata → problema serio.

---

## 8. TensorBoard: il monitor del training

Durante il training salviamo ad ogni epoch nel log di TensorBoard:
- `train/loss_epoch`: la loss media di training
- `val/loss`, `val/mIoU`, `val/F1`, `val/accuracy`: metriche di validazione
- `val/iou_<classe>`: IoU per ogni classe
- `train/lr`: il learning rate corrente (per verificare che lo scheduler funzioni)

Aprendo TensorBoard nel browser vedi questi grafici. Cosa cercare:

- **loss di train che scende**: il modello sta imparando
- **loss di val che scende e poi risale**: overfitting, serve early stopping o più regolarizzazione
- **mIoU che sale e poi satura**: convergenza raggiunta
- **IoU per classe**: se una classe ha IoU=0 persistente, controllare se è presente nel train set

---

## 9. Strategia hardware: perché non lo abbiamo fatto tutto in locale

Il tuo Zenbook 14 OLED ha Intel Arc integrata, non una GPU NVIDIA. PyTorch non ha CUDA, deve
girare su CPU. Un epoch completo di VDD (70 batch) su CPU richiederebbe ~5-10 minuti → 30
epoch sarebbero ~3-5 ore. Moltiplicando per Pix2Pix e per re-training = intere giornate.

Colab Free offre una **T4 (NVIDIA)** gratis per 12 ore/sessione, con cui un epoch dura ~30
secondi → 30 epoch in 15 minuti. Quindi:

- **In locale**: sviluppo del codice, debugging, smoke test con dataset piccoli
- **Su Colab**: training reali

Questo è un pattern standard nella pratica del deep learning. Sapere quando usare l'uno o
l'altro è parte del mestiere.

---

## 10. Cosa diciamo all'esame sullo smoke test

Il nostro smoke test (2 epoch su 20 immagini, su CPU, in 1 minuto) non è un "risultato": è
una **verifica di correttezza** della pipeline. Cosa ci ha dimostrato:

- Le dimensioni dei tensori sono corrette in tutti i punti (input 3×512×512, output 7×512×512)
- La loss scende (1.86 → 1.63): il gradiente fluisce, i pesi si aggiornano
- Le metriche salgono (mIoU 0.13 → 0.21): il modello sta imparando davvero, non solo
  "fittando" rumore
- Il salvataggio del checkpoint, il logging, il subset mode funzionano

Questo tipo di verifica end-to-end si chiama **smoke test** (test del fumo): come provare
l'elettricità di un dispositivo nuovo — se non prende fuoco, sei a buon punto.

---

## 11. Elenco delle domande tipiche d'esame e le risposte brevi

**Q: Perché 512×512?**
A: Compromesso fra accuratezza e velocità. Sufficiente per preservare i dettagli degli
oggetti (veicoli, tetti) ma permette batch più grandi in memoria. Standard per U-Net.

**Q: Perché CrossEntropy e non Dice loss?**
A: CrossEntropy è il baseline standard. Per classi molto sbilanciate (es. il 2% dei pixel
sono "vehicle" nel nostro caso) la Dice loss spesso fa meglio. Una possibile estensione è
usare una combinazione `α × CE + β × Dice`. Per il baseline iniziale CE è la scelta giusta.

**Q: Come evitate l'overfitting?**
A: Tre meccanismi:
1. Data augmentation (flip, rotate, jitter)
2. Weight decay (L2 regolarizzazione nei pesi)
3. Selezione del best checkpoint su val set (non su train set)

**Q: Perché Adam e non SGD?**
A: AdamW adatta il learning rate ad ogni parametro ed è più tollerante a lr non perfettamente
tarati. Per SGD servirebbe un tuning accurato di lr e momentum. Per un progetto di corso Adam
dà risultati buoni con meno prove.

**Q: Come scegliete il numero di epoch?**
A: Iniziamo con un budget ragionevole (30-50) e guardiamo TensorBoard. Se la val loss
scende ancora, si può aumentare. Se satura o risale (overfitting), si può tagliare prima.

**Q: Split di VDD: come è fatto?**
A: Lo split train/val/test è già fornito dagli autori del dataset (70/20/10). Noi lo
usiamo così com'è. Il val set serve per scegliere il best model e tarare gli iperparametri,
il test set sarà usato **solo** per la valutazione finale della robustezza.

**Q: La nebbia GAN è "giusta"?**
A: Ci aspettiamo due limiti:
1. Il GAN è addestrato su Cityscapes (street-level), le immagini di VDD sono aeree
   → domain gap, ma è proprio quello che il progetto indaga
2. La "nebbia ground truth" di Foggy Cityscapes è già sintetica (modello fisico), quindi
   il GAN apprende un'approssimazione di un'approssimazione. È un limite noto.

---

## 12. Roadmap: cosa ancora ci manca

Fasi non ancora affrontate (da affrontare nelle prossime sessioni):

- **Fase 1 - Step 5**: training reale della U-Net su Colab (baseline su VDD clean)
- **Fase 2**: Pix2Pix — architettura GAN, loss adversariale, training
- **Fase 3**: generazione di VDD_foggy applicando il Pix2Pix addestrato
- **Fase 4**: test di robustezza (U-Net clean applicata a VDD_foggy)
- **Fase 5**: re-training della U-Net su dati nebbiosi, valutazione finale
- **Fase 6**: stesura del report con grafici, tabelle e discussione

---

## 13. Glossario veloce

| Termine | Definizione breve |
|---------|-------------------|
| Backbone / Encoder | La parte della rete che estrae feature dalle immagini |
| Logits | Output grezzo del modello, prima del softmax |
| Softmax | Funzione che converte logits in probabilità [0,1] che sommano a 1 |
| Batch | Gruppo di campioni processati insieme (per efficienza) |
| Epoch | Un passaggio completo su tutto il dataset di training |
| Step / Iteration | Un aggiornamento dei pesi = un batch processato |
| Gradient | Derivata della loss rispetto a ciascun peso |
| Backward pass | Calcolo automatico di tutti i gradienti (chain rule) |
| Checkpoint | File `.pth` che contiene lo stato del modello |
| State dict | Dizionario Python che mappa nomi dei layer → tensori dei pesi |
| Fine-tuning | Partire da pesi pretrained e continuare il training sul tuo dataset |
| Transfer learning | Riusare un modello addestrato per un task A su un task B correlato |
| Domain gap | Differenza tra il dominio di addestramento e quello di test |
| Overfitting | Il modello memorizza il train set ma non generalizza al val set |
| Data augmentation | Trasformazioni casuali applicate al training per aumentare varietà |

---

## 14. Come esercitarsi prima dell'esame

Suggerisco di:

1. **Aprire ogni file del progetto e spiegarselo a voce**: se riesci a raccontarlo a una
   persona che non sa di ML, lo sai davvero.

2. **Ripetere 3 volte lo smoke test** cambiando un parametro alla volta: prova
   `--batch-size 4`, poi `--lr 1e-3`, poi `--no-pretrained`. Osserva come cambiano le
   metriche. Questo ti allena a "leggere" i numeri.

3. **Disegnare a mano la pipeline end-to-end**: dal dataset all'output. Un flowchart su
   carta ti fa interiorizzare la struttura.

4. **Leggere la documentazione di**:
   - Il paper U-Net originale (Ronneberger 2015) — 5 pagine, molto chiaro
   - Il README di `segmentation_models_pytorch`
   - Il paper VDD (2024) per conoscere il dataset

5. **Simulare una presentazione di 10 minuti**: obiettivo → pipeline → scelte → risultati
   (smoke test) → next steps (Colab, GAN). Registrati, riascoltati.

---

*Guida preparata il 22 aprile 2026. Aggiorna dopo ogni fase successiva.*
