# Spiegazione file-per-file del codice

Questo documento spiega **ogni file che abbiamo scritto**, con commenti a blocchi di codice
e spiegazioni pensate per chi non ha mai lavorato con PyTorch / deep learning.

L'ordine è quello in cui li abbiamo creati, perché riflette la costruzione logica del progetto:
prima lo scheletro, poi il dataset, poi il modello, poi il training.

---

## Indice dei file

| # | File | Cosa fa |
|---|------|---------|
| 1 | `requirements.txt` | Lista delle librerie Python necessarie |
| 2 | `requirements-colab.txt` | Versione ridotta per Colab |
| 3 | `.gitignore` | File che git deve ignorare |
| 4 | `README.md` | Documentazione di progetto |
| 5 | `scripts/check_env.py` | Verifica che l'ambiente sia a posto |
| 6 | `scripts/download_vdd.py` | Scarica il dataset VDD da HuggingFace |
| 7 | `src/datasets/vdd.py` | **Classe Dataset per VDD** |
| 8 | `scripts/visualize_vdd.py` | Script per vedere campioni VDD |
| 9 | `src/utils/transforms.py` | **Pipeline di augmentation** |
| 10 | `scripts/test_dataloader.py` | Test end-to-end del dataloader |
| 11 | `src/models/unet.py` | **Costruzione del modello U-Net** |
| 12 | `scripts/test_model.py` | Smoke test del modello |
| 13 | `src/evaluation/metrics.py` | **Calcolo mIoU, F1, accuracy** |
| 14 | `src/training/train_unet.py` | **Training loop completo** |

I file in **grassetto** sono i più importanti da studiare.

---

## 1. `requirements.txt`

### A cosa serve

Elenco di tutte le librerie Python che il progetto usa. Con `pip install -r requirements.txt`
pip le scarica e installa tutte in una volta.

### Riga per riga

```
torch>=2.2.0
torchvision>=0.17.0
```
Il cuore di tutto. `torch` è PyTorch, la libreria per deep learning. `torchvision` aggiunge
utility specifiche per immagini (dataset standard, modelli classici).

```
segmentation-models-pytorch>=0.3.3
```
Libreria che ci dà U-Net e altre architetture per segmentation "pronte all'uso". Risparmia
centinaia di righe di codice.

```
timm>=0.9.12
```
Timm (PyTorch Image Models) è un repository di backbone pretrained. `segmentation-models-pytorch`
la usa internamente.

```
albumentations>=1.4.0
opencv-python>=4.9.0
Pillow>=10.2.0
```
Libreria di augmentation (Albumentations), e le due librerie principali per manipolare
immagini in Python (OpenCV e Pillow).

```
torchmetrics>=1.3.0
scikit-learn>=1.4.0
```
Torchmetrics: metriche (mIoU, F1, accuracy, ecc.) integrate con PyTorch.
Scikit-learn: usata per utility varie (es. split, normalizzazione).

```
numpy>=1.26.0
pandas>=2.2.0
scipy>=1.12.0
matplotlib>=3.8.0
seaborn>=0.13.0
```
Numeri, tabelle, matematica scientifica, grafici. Il pacchetto base di Python scientifico.

```
tensorboard>=2.16.0
```
Il tool con cui monitoriamo il training in tempo reale.

```
tqdm>=4.66.0
```
Libreria che stampa le barre di progresso che hai visto durante il training (quelle con
le percentuali che avanzano).

```
pyyaml>=6.0.1
hydra-core>=1.3.2
```
Gestione di file di configurazione (non le usiamo ancora ma potremmo in futuro).

```
einops>=0.7.0
```
Libreria per manipolare dimensioni di tensori in modo leggibile (non strettamente necessaria
ma utile).

```
jupyter>=1.0.0
ipykernel>=6.29.0
```
Per notebook Jupyter (se decidiamo di usarli).

```
torch-fidelity>=0.3.0
lpips>=0.1.4
```
Metriche per valutare la qualità delle immagini generate dal GAN (le useremo nella Fase 2).

```
huggingface_hub>=0.20.0
kaggle>=1.6.0
```
Librerie per scaricare dataset da HuggingFace e Kaggle.

---

## 2. `requirements-colab.txt`

Stesse librerie ma più corto: **su Colab PyTorch è già preinstallato** (con CUDA!), quindi
non serve reinstallarlo. Questo file contiene solo le librerie aggiuntive.

---

## 3. `.gitignore`

### A cosa serve

Dice a git quali file **non** tracciare. Serve perché certi file non vanno mai nella
repository:
- sono troppo grossi (dataset, checkpoint)
- sono personali (file di configurazione dell'editor, credenziali)
- sono rigenerabili (cache Python)

### Parti principali

```
data/raw/**
data/processed/**
```
I dataset sono grossi (VDD è 2 GB). Non li pushiamo su GitHub. Gli `**` significano
"ricorsivamente, tutto quello che c'è dentro".

```
!data/raw/.gitkeep
```
Il `!` è una negazione: "ma tieni `.gitkeep`". È un trucco per far sì che git tracci
comunque la struttura di cartelle anche se vuote. `.gitkeep` è un file vuoto convenzionale.

```
checkpoints/**
*.pth
*.pt
```
I pesi del modello possono essere centinaia di MB. Non si pushano.

```
__pycache__/
*.py[cod]
```
Python genera questi file automaticamente quando importi moduli (sono i moduli compilati
in bytecode). Sono ricostruiti quando servono, quindi non vanno versionati.

```
.venv/
```
L'ambiente virtuale Python. Contiene migliaia di file della libreria standard e delle
dipendenze. **Mai** pushare un venv: si ricostruisce con `pip install -r requirements.txt`.

```
kaggle.json
.env
```
Credenziali. Assolutamente mai pubblicarle.

---

## 4. `README.md`

Documentazione del progetto. È il primo file che qualcuno vede quando apre la repo su
GitHub. Contiene:
- pipeline di alto livello (il diagramma ASCII)
- struttura cartelle
- istruzioni di setup (locale e Colab)
- link ai dataset

GitHub lo renderizza automaticamente se si chiama `README.md` e sta nella radice.

---

## 5. `scripts/check_env.py`

### A cosa serve

Script di sanity check che verifica se Python + PyTorch + tutte le librerie funzionano
correttamente. Da lanciare all'inizio per essere sicuri di non avere problemi installativi.

### Blocco 1: import e intestazione

```python
import sys
import platform

def main():
    print("=" * 60)
    print("FOG-UAV-ROBUSTNESS :: Environment Check")
    print("=" * 60)
```
Stampa un'intestazione carina con 60 trattini. `sys` dà accesso a info di Python,
`platform` a info del sistema operativo.

### Blocco 2: info Python

```python
print(f"\n[Python]")
print(f"  Version : {sys.version.split()[0]}")
print(f"  Platform: {platform.system()} {platform.release()} ({platform.machine()})")
```
`sys.version` è una stringa tipo `"3.11.9 (main, ...)"`. Lo splittiamo sugli spazi e
prendiamo il primo pezzo: `"3.11.9"`.
`platform.system()` → "Windows", `platform.machine()` → "AMD64".

### Blocco 3: PyTorch e CUDA

```python
try:
    import torch
    print(f"  Version       : {torch.__version__}")
    print(f"  CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  CUDA device   : {torch.cuda.get_device_name(0)}")
        print(f"  CUDA VRAM     : {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
except ImportError:
    print("\n[PyTorch] NOT installed. ...")
    return
```

Il `try/except ImportError` serve a dare un errore pulito se PyTorch non è installato
(invece di un crash). `torch.cuda.is_available()` ritorna True solo se esiste una GPU
NVIDIA con i driver CUDA. Sulla tua macchina è False, su Colab sarà True.

### Blocco 4: librerie core

```python
for name in ["torchvision", "segmentation_models_pytorch", "albumentations",
             "torchmetrics", "cv2", "numpy", "matplotlib"]:
    try:
        mod = __import__(name)
        version = getattr(mod, "__version__", "?")
        print(f"  OK   {name:32s} {version}")
    except ImportError:
        print(f"  MISS {name}")
```

Questo è un ciclo che prova a importare ogni libreria della lista. `__import__(name)` è
un modo "dinamico" di importare dato il nome come stringa. Se l'import funziona stampiamo
"OK", altrimenti "MISS".

`getattr(mod, "__version__", "?")` legge l'attributo `__version__` se esiste, altrimenti
ritorna `"?"`. Alcuni moduli non hanno un attributo `__version__` (es. OpenCV, che usa
`cv2.__version__`).

### Blocco 5: raccomandazione device

```python
if torch.cuda.is_available():
    print("  --> CUDA GPU detected. Use device='cuda'.")
elif hasattr(torch, "xpu") and torch.xpu.is_available():
    print("  --> Intel XPU detected. Use device='xpu' (via IPEX).")
else:
    print("  --> Only CPU available. Use device='cpu'.")
```

Seleziona automaticamente il miglior dispositivo disponibile. Su Colab consiglierà CUDA,
sulla tua macchina CPU.

---

## 6. `scripts/download_vdd.py`

Script brevissimo per scaricare VDD da HuggingFace:

```python
from huggingface_hub import snapshot_download

path = snapshot_download(
    repo_id='RussRobin/VDD',
    repo_type='dataset',
    local_dir='data/raw/VDD',
)
print(f'Download completato in: {path}')
```

`snapshot_download` scarica tutto il contenuto di una repo HuggingFace. `repo_type='dataset'`
perché VDD è catalogato come dataset (non come modello). `local_dir` specifica dove salvare
i file. La funzione usa le credenziali salvate con `hf auth login` per accedere a repo gated.

---

## 7. `src/datasets/vdd.py` ⭐ IMPORTANTE

Questo è il **cuore del data pipeline**. La classe `VDDDataset` è ciò che PyTorch chiamerà
ogni volta che ha bisogno di un'immagine durante il training.

### Blocco 1: docstring e import

```python
"""VDD (Varied Drone Dataset) PyTorch Dataset class.

VDD has 7 semantic classes (masks are grayscale uint8, class ID per pixel):
    0 = other
    1 = wall
    ...
"""
from pathlib import Path
from typing import Callable, Optional, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
```

`from pathlib import Path`: classe Python per manipolare percorsi di file in modo
cross-platform (funziona su Windows, Linux, Mac senza cambiare codice).

`from typing import Callable, Optional, Tuple`: type hints. Aiutano a documentare il codice
(es. "questa funzione ritorna una tupla"), non cambiano il comportamento a runtime.

`from torch.utils.data import Dataset`: la classe base di PyTorch che stiamo estendendo.

### Blocco 2: costanti a livello di modulo

```python
VDD_CLASSES = {
    0: "other",
    1: "wall",
    2: "road",
    3: "vegetation",
    4: "vehicle",
    5: "roof",
    6: "water",
}

VDD_COLOR_MAP = np.array(
    [
        [  0,   0,   0],  # 0 other       - black
        [128,   0,   0],  # 1 wall        - dark red
        [128,  64, 128],  # 2 road        - purple
        [  0, 128,   0],  # 3 vegetation  - green
        [ 64,   0, 128],  # 4 vehicle     - violet
        [128, 128,   0],  # 5 roof        - olive
        [  0,   0, 128],  # 6 water       - dark blue
    ],
    dtype=np.uint8,
)
```

`VDD_CLASSES`: dizionario da class ID a nome leggibile. Serve per stampare info nei log.

`VDD_COLOR_MAP`: mappa da class ID a colore RGB. Serve per **visualizzare** le maschere.
Le maschere raw sono immagini grayscale con valori 0-6: sono illeggibili a occhio nudo
(il pixel 2 è appena distinguibile dal pixel 3). Moltiplicando per questa mappa trasformiamo
ogni class ID in un colore visibile.

`dtype=np.uint8`: interi tra 0 e 255. È il formato standard per i colori.

### Blocco 3: classe VDDDataset - costruttore

```python
class VDDDataset(Dataset):
    NUM_CLASSES = 7
    IMG_EXT = ".JPG"
    MASK_EXT = ".png"

    def __init__(
        self,
        root: str,
        split: str = "train",
        transform: Optional[Callable] = None,
    ) -> None:
```

Attributi di classe (`NUM_CLASSES`, `IMG_EXT`, `MASK_EXT`): costanti che non cambiano per
nessun'istanza. Raccolte qui per facilità di manutenzione (se VDD cambiasse il formato
delle immagini basta cambiare una riga).

Parametri:
- `root`: percorso alla cartella VDD (quella che contiene train/val/test)
- `split`: quale split usare
- `transform`: funzione opzionale di augmentation. Se `None`, ritorniamo i dati grezzi.

### Blocco 4: validazione e costruzione liste

```python
if split not in ("train", "val", "test"):
    raise ValueError(f"split must be one of train/val/test, got '{split}'")

self.root = Path(root)
self.split = split
self.transform = transform

self.img_dir = self.root / split / "src"
self.mask_dir = self.root / split / "gt"

if not self.img_dir.is_dir():
    raise FileNotFoundError(f"Image directory not found: {self.img_dir}")
```

- Controllo che lo split sia valido (rifiutiamo tipo `split="tran"` con typo).
- Costruiamo i percorsi alle cartelle. `Path` permette di usare `/` come operatore di
  concatenazione, quindi `root / "train" / "src"` diventa `root/train/src` (con separatori
  corretti per l'OS).
- Verifichiamo che le cartelle esistano. Se no, errore chiaro.

```python
self.img_paths = sorted(self.img_dir.glob(f"*{self.IMG_EXT}"))
if len(self.img_paths) == 0:
    raise RuntimeError(...)

missing = [
    p.name for p in self.img_paths
    if not (self.mask_dir / (p.stem + self.MASK_EXT)).is_file()
]
if missing:
    raise RuntimeError(...)
```

`glob("*.JPG")`: trova tutti i file con estensione `.JPG` nella cartella. È un pattern
standard shell-like.

`sorted(...)`: ordine alfabetico garantito (altrimenti sarebbe dipendente dall'OS). Serve
per **riproducibilità**: se fisso un seed e chiedo il campione 5, voglio sempre lo stesso
file.

Il controllo `missing`: per ogni immagine verifichiamo che esista la maschera corrispondente.
`p.stem` è il nome del file senza estensione (`"DJI_0008.JPG"` → `"DJI_0008"`). Se mancano
maschere, errore.

### Blocco 5: metodo `__len__`

```python
def __len__(self) -> int:
    return len(self.img_paths)
```

PyTorch chiama questo quando deve sapere la dimensione del dataset. Restituiamo quante
immagini abbiamo. Permette a `len(dataset)` di funzionare.

### Blocco 6: metodo `__getitem__` (il più importante!)

```python
def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
    img_path = self.img_paths[idx]
    mask_path = self.mask_dir / (img_path.stem + self.MASK_EXT)

    image = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"Failed to read image: {img_path}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
```

Questo è chiamato **ogni volta** che il DataLoader vuole un campione. Carica l'immagine
al volo (non la teniamo in RAM).

`cv2.imread(..., cv2.IMREAD_COLOR)`: legge l'immagine a colori. Attenzione: **OpenCV
restituisce BGR**, non RGB! Ordine dei canali invertito rispetto allo standard.

`cv2.cvtColor(..., cv2.COLOR_BGR2RGB)`: converte BGR → RGB. Passo fondamentale, se lo
dimentichi il modello riceve dati sbagliati.

```python
mask = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
if mask is None:
    raise RuntimeError(...)
if mask.ndim != 2:
    raise RuntimeError(...)
```

`IMREAD_UNCHANGED`: legge l'immagine "come è". Per la maschera vogliamo grayscale singolo
canale, senza conversioni automatiche.

`mask.ndim != 2`: la maschera deve essere 2D (solo H×W, no canali). Se avesse 3 dimensioni
significherebbe che è RGB, e dovremmo cambiare strategia.

```python
if self.transform is not None:
    out = self.transform(image=image, mask=mask)
    image = out["image"]
    mask = out["mask"].long() if torch.is_tensor(out["mask"]) else torch.from_numpy(out["mask"]).long()
else:
    image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
    mask = torch.from_numpy(mask).long()

return image, mask
```

Due rami:

**Con transform (caso normale)**: chiamiamo `transform(image=..., mask=...)`. Albumentations
accetta questo formato specifico e ritorna un dict `{"image": ..., "mask": ...}`. Poi
estraiamo i tensori. La conversione `.long()` serve a trasformare il dtype della maschera
a intero a 64 bit (PyTorch lo vuole per la CrossEntropyLoss).

**Senza transform**: costruiamo i tensori a mano:
- `torch.from_numpy(image)`: converte numpy → PyTorch
- `.permute(2, 0, 1)`: riordina le dimensioni da `(H, W, 3)` a `(3, H, W)`. **PyTorch vuole
  i canali per primi** (convenzione "channels-first"), numpy/OpenCV per ultimi (channels-last).
- `.float() / 255.0`: converte a float e normalizza da [0,255] a [0,1]

### Blocco 7: metodo `decode_segmap` (utility)

```python
@staticmethod
def decode_segmap(mask: np.ndarray) -> np.ndarray:
    if mask.ndim != 2:
        raise ValueError(f"Expected 2D mask, got {mask.shape}")
    return VDD_COLOR_MAP[mask]
```

Trasforma una maschera grayscale (class ID) in un'immagine RGB colorata per visualizzazione.

`VDD_COLOR_MAP[mask]`: fancy indexing di numpy. Se `mask` ha shape `(H, W)` con valori da
0 a 6, e `VDD_COLOR_MAP` ha shape `(7, 3)`, allora `VDD_COLOR_MAP[mask]` ritorna un array
di shape `(H, W, 3)` dove ogni pixel è il colore della sua classe. Magia di numpy.

`@staticmethod`: significa che questo metodo non dipende dall'istanza (non usa `self`).
Si può chiamare come `VDDDataset.decode_segmap(mask)` senza istanziare il dataset.

---

## 8. `scripts/visualize_vdd.py`

Script che carica VDD e salva una figura matplotlib con alcuni campioni. Serve come
sanity check visivo.

### Parti principali

```python
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
```

Questo è un trucco necessario perché lo script sta in `scripts/` ma deve importare da
`src/`. Aggiungiamo la cartella del progetto (due livelli sopra il file) al `sys.path`,
così Python trova il modulo `src`.

```python
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--root", default="data/raw/VDD/VDD")
    p.add_argument("--split", default="train", choices=["train", "val", "test"])
    p.add_argument("--n", type=int, default=4)
    p.add_argument("--out", default="outputs/figures/vdd_samples.png")
    return p.parse_args()
```

`argparse`: libreria standard per gestire argomenti da linea di comando. Permette di
lanciare lo script con `--split val --n 6` invece di hard-codare.

```python
ds = VDDDataset(root=args.root, split=args.split, transform=None)
```

Istanziamo il dataset **senza transform** per ottenere le immagini originali non
normalizzate (useremo quelle per il plot).

```python
indices = np.linspace(0, len(ds) - 1, n, dtype=int)
```

`np.linspace(0, N-1, n)`: genera `n` numeri equamente distanziati da 0 a N-1. Esempio:
`linspace(0, 279, 4)` → `[0, 93, 186, 279]`. Serve per prendere campioni sparsi nel
dataset invece che tutti i primi.

```python
fig, axes = plt.subplots(n, 3, figsize=(14, 4 * n))
```

Crea una figura matplotlib con `n` righe e 3 colonne. `figsize` in pollici.

```python
for row, idx in enumerate(indices):
    image, mask = ds[idx]
    img_np = image.permute(1, 2, 0).numpy()
    mask_np = mask.numpy()
    mask_rgb = VDDDataset.decode_segmap(mask_np)
    overlay = (0.55 * img_np + 0.45 * (mask_rgb / 255.0)).clip(0, 1)
```

Per ogni campione:
- Estraiamo immagine e maschera come tensori
- Le convertiamo in numpy
- `permute(1, 2, 0)` di nuovo: tensori PyTorch sono CHW, matplotlib vuole HWC
- Coloriamo la maschera
- Facciamo un **overlay** (sovrapposizione): 55% immagine + 45% maschera colorata. Utile
  per vedere se la segmentazione si allinea correttamente alla foto.

```python
plt.savefig(out_path, dpi=80, bbox_inches="tight")
```

Salva la figura. `dpi=80` (risoluzione bassa perché le immagini sono grandi e non vogliamo
un file enorme). `bbox_inches="tight"` ritaglia i margini bianchi.

---

## 9. `src/utils/transforms.py` ⭐ IMPORTANTE

Pipeline di augmentation con Albumentations.

### Blocco 1: costanti ImageNet

```python
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
```

Questi sono i valori **standard universali** usati da quasi tutti i modelli pretrained su
ImageNet. Sono la media e la std per canale (R, G, B) calcolate sul training set di ImageNet.

### Blocco 2: `get_train_transform`

```python
def get_train_transform(image_size: int = 512) -> A.Compose:
    return A.Compose(
        [
            A.Resize(height=image_size, width=image_size, interpolation=1),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.RandomBrightnessContrast(
                brightness_limit=0.2,
                contrast_limit=0.2,
                p=0.5,
            ),
            A.HueSaturationValue(
                hue_shift_limit=10,
                sat_shift_limit=15,
                val_shift_limit=10,
                p=0.3,
            ),
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2(),
        ]
    )
```

`A.Compose`: concatena più trasformazioni. Le applica in sequenza nell'ordine dato.
Ogni trasformazione ha un parametro `p` (probabilità di applicarla).

Ordine delle trasformazioni (importante!):
1. **Resize**: prima, per standardizzare le dimensioni
2. **Trasformazioni geometriche** (flip, rotate): devono essere applicate **sia all'immagine
   che alla maschera** contemporaneamente. Albumentations lo fa automaticamente se passi
   `mask=...`.
3. **Trasformazioni di colore** (brightness, hue): solo sull'immagine, non sulla maschera
   (non vuoi cambiare i class ID!). Anche questo Albumentations lo gestisce da solo.
4. **Normalize**: sottrae mean e divide per std. Applicato solo all'immagine.
5. **ToTensorV2**: converte numpy → tensore PyTorch e riordina in CHW.

`interpolation=1` nel Resize: usa `INTER_LINEAR` (bilineare) per le immagini. Per le
maschere Albumentations usa automaticamente `INTER_NEAREST` (nearest neighbor), altrimenti
i class ID diventerebbero decimali dopo l'interpolazione.

### Blocco 3: `get_eval_transform`

```python
def get_eval_transform(image_size: int = 512) -> A.Compose:
    return A.Compose(
        [
            A.Resize(height=image_size, width=image_size, interpolation=1),
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2(),
        ]
    )
```

Solo resize + normalize + tensor. **Nessuna augmentation random**. La validazione e il
test devono essere deterministici.

### Blocco 4: `denormalize`

```python
def denormalize(tensor, mean=IMAGENET_MEAN, std=IMAGENET_STD):
    import torch
    if tensor.ndim == 3:
        tensor = tensor.unsqueeze(0)
    mean_t = torch.tensor(mean).view(1, 3, 1, 1).to(tensor.device)
    std_t = torch.tensor(std).view(1, 3, 1, 1).to(tensor.device)
    out = tensor * std_t + mean_t
    out = out.clamp(0.0, 1.0)
    return out.squeeze(0) if out.shape[0] == 1 else out
```

Inverte la normalizzazione. Utile per **visualizzare** un batch dopo che l'abbiamo
normalizzato: i valori normalizzati sarebbero nel range [-2, +2] e matplotlib mostrerebbe
immagini sbagliate.

Formula: `x_denormalized = x_normalized * std + mean`.

`.view(1, 3, 1, 1)`: cambia lo shape dei tensori mean e std a `(1, 3, 1, 1)` per permettere
broadcasting con un batch `(B, 3, H, W)`.

`.clamp(0.0, 1.0)`: forza i valori nel range [0, 1] eliminando piccole imprecisioni
numeriche.

---

## 10. `scripts/test_dataloader.py`

Test end-to-end del dataloader. Verifica che caricamento, augmentation, batching e
normalizzazione funzionino insieme.

### Punti salienti

```python
loader = DataLoader(
    ds,
    batch_size=args.batch_size,
    shuffle=(args.split == "train"),
    num_workers=args.num_workers,
    pin_memory=False,
    drop_last=False,
)
```

Creazione del `DataLoader`:
- `batch_size`: quanti campioni in un batch
- `shuffle`: se rimescolare l'ordine a ogni epoch. Sì per train, no per val/test (deve
  essere deterministico).
- `num_workers`: quanti processi paralleli caricano i dati. 0 = tutto nel thread principale
  (più lento ma nessun problema di serializzazione). Su Windows, `num_workers > 0` può dare
  problemi con `multiprocessing`, per questo iniziamo con 0.
- `pin_memory`: ottimizzazione per transfer CPU → GPU. Inutile senza GPU.
- `drop_last`: se l'ultima batch è incompleta, scartala? No, la teniamo.

```python
for i, (images, masks) in enumerate(loader):
    if i == 0:
        assert images.ndim == 4 and images.shape[1] == 3, ...
        assert masks.ndim == 3, ...
        assert images.dtype == torch.float32, ...
        assert masks.dtype == torch.long, ...
```

Gli `assert` sono sanity check runtime: se le condizioni non sono vere, crash con un
errore descrittivo. Tipica pratica "defensive programming". Verifichiamo che:
- Images abbia 4 dimensioni (B, C, H, W) con C=3 (RGB)
- Masks abbia 3 dimensioni (B, H, W)
- Dtypes corretti (float per immagini, long per maschere)

```python
total_images += images.shape[0]
elapsed = time.perf_counter() - t0
print(f"[TIMING] {total_images} images in {elapsed:.2f}s ...")
```

Misuriamo il throughput. Se i dati si caricano troppo lentamente, il training sarà
bloccato dall'I/O anche se la GPU è potente.

---

## 11. `src/models/unet.py` ⭐ IMPORTANTE

### Blocco 1: la funzione `build_unet`

```python
def build_unet(
    num_classes: int = 7,
    encoder_name: str = "resnet34",
    encoder_weights: Optional[str] = "imagenet",
    in_channels: int = 3,
) -> nn.Module:
    model = smp.Unet(
        encoder_name=encoder_name,
        encoder_weights=encoder_weights,
        in_channels=in_channels,
        classes=num_classes,
    )
    return model
```

Una funzione factory molto sottile che delega tutto a `segmentation_models_pytorch`.
`smp.Unet` costruisce:
- L'encoder (ResNet-34 in questo caso) con pesi pretrained caricati automaticamente
- Il decoder U-Net standard
- Le skip connections tra encoder e decoder
- Un layer finale che mappa ai `num_classes` canali di output

Il modello si comporta come una normale `nn.Module` PyTorch: ha `model(x)`, `.train()`,
`.eval()`, `model.state_dict()`, ecc.

### Blocco 2: utility `count_parameters`

```python
def count_parameters(model: nn.Module) -> tuple:
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total
```

Conta i parametri del modello. `p.numel()` è il numero di elementi del tensore (es. un
layer 512×1024 ha 524288 parametri). `p.requires_grad` indica se il parametro verrà
aggiornato durante il training (True di default; si può congelare parti del modello
impostandolo a False).

### Blocco 3: `human_readable`

```python
def human_readable(n: int) -> str:
    if n >= 1_000_000_000: return f"{n / 1e9:.2f}B"
    if n >= 1_000_000: return f"{n / 1e6:.2f}M"
    if n >= 1_000: return f"{n / 1e3:.1f}K"
    return str(n)
```

Stampa "24.44M" invece di "24436871". Utility pura di presentazione.

---

## 12. `scripts/test_model.py`

Smoke test del modello: costruzione + forward + backward. Non lo approfondisco riga per
riga perché è simile a `test_dataloader.py` come struttura. Le cose da sapere:

```python
model.eval()
with torch.no_grad():
    logits = model(x)
```

Forward in modalità evaluation. `torch.no_grad()` disabilita il calcolo dei gradienti
(non servono in inference, risparmiamo memoria).

```python
model.train()
logits = model(x)
loss = criterion(logits, targets)
loss.backward()
optimizer.step()
```

Forward + backward + step in modalità training. `loss.backward()` è la "magia" di PyTorch:
calcola automaticamente i gradienti di loss rispetto a ogni parametro usando la chain rule
(algoritmo di backpropagation).

```python
grads_present = sum(
    1 for p in model.parameters() if p.grad is not None and p.grad.abs().sum() > 0
)
```

Verifichiamo che i gradienti siano effettivamente fluiti in tutti i parametri. Se un
parametro ha `grad == None` o tutti zeri, c'è un bug nel grafo computazionale (es. un
layer disconnesso).

---

## 13. `src/evaluation/metrics.py` ⭐ IMPORTANTE

Wrapper intorno a `torchmetrics` per calcolare mIoU, F1 e accuracy.

### Blocco 1: il costruttore

```python
class SegmentationMetrics:
    def __init__(
        self,
        num_classes: int,
        device: torch.device,
        class_names: Optional[List[str]] = None,
    ) -> None:
        self.num_classes = num_classes
        self.device = device
        self.class_names = class_names or [f"class_{i}" for i in range(num_classes)]

        self.miou = JaccardIndex(
            task="multiclass", num_classes=num_classes, average="macro",
        ).to(device)
```

`JaccardIndex` è il nome formale di "IoU". `torchmetrics` lo calcola per noi.
- `task="multiclass"`: più di due classi
- `average="macro"`: media semplice degli IoU per classe (= mIoU classico)
- `.to(device)`: le metriche di torchmetrics hanno stati interni (matrice di confusione),
  che vanno spostati sul dispositivo giusto.

```python
self.per_class_iou = JaccardIndex(..., average="none").to(device)
```

Stessa cosa ma `average="none"` ritorna un IoU per ogni classe separatamente (non la media).

```python
self.f1 = F1Score(..., average="macro").to(device)
self.acc = Accuracy(..., average="micro").to(device)
```

F1 macro = media dei F1 per classe. Accuracy micro = accuracy globale (totale pixel corretti
/ totale pixel). Abbiamo scelto diverse aggregazioni per diverse metriche, seguendo la
letteratura.

### Blocco 2: `update`

```python
def update(self, logits: torch.Tensor, targets: torch.Tensor) -> None:
    preds = logits.argmax(dim=1)
    self.miou.update(preds, targets)
    self.per_class_iou.update(preds, targets)
    self.f1.update(preds, targets)
    self.acc.update(preds, targets)
```

`logits.argmax(dim=1)`: per ogni pixel, trova la classe con lo score più alto. Da shape
`(B, 7, H, W)` a shape `(B, H, W)`. **Questa è la predizione finale del modello.**

Poi "aggiorniamo" ogni metrica con le predizioni del batch corrente. Internamente
`torchmetrics` accumula una matrice di confusione, così alla fine può calcolare tutte le
metriche con una singola passata.

### Blocco 3: `compute`

```python
def compute(self) -> Dict:
    per_class = self.per_class_iou.compute().detach().cpu().numpy()
    return {
        "mIoU": float(self.miou.compute().item()),
        "F1": float(self.f1.compute().item()),
        "accuracy": float(self.acc.compute().item()),
        "per_class_iou": {
            name: float(per_class[i]) for i, name in enumerate(self.class_names)
        },
    }
```

Alla fine dell'epoch chiamiamo `compute()` per ottenere i valori finali. Restituiamo un
dict Python con scalari float (facile da loggare o salvare su JSON).

`.detach()`: stacca il tensore dal grafo computazionale (non ci servono più gradienti).
`.cpu()`: se era su GPU, porta su CPU.
`.numpy()`: converte a numpy.

### Blocco 4: `reset`

```python
def reset(self) -> None:
    self.miou.reset()
    ...
```

**Importante**: dopo `compute()` serve chiamare `reset()` all'inizio dell'epoch successivo,
altrimenti le metriche continuerebbero ad accumulare sui vecchi dati.

---

## 14. `src/training/train_unet.py` ⭐⭐ IL PIÙ IMPORTANTE

Il training loop completo. Lo spezzo in blocchi funzionali.

### Blocco A: argomenti CLI

```python
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", default="data/raw/VDD/VDD")
    p.add_argument("--image-size", type=int, default=512)
    p.add_argument("--subset", type=int, default=0,
                   help="If >0, use only first N samples ...")
    ...
```

Tutti gli iperparametri sono configurabili da linea di comando. Questo è **cruciale** per:
- Riproducibilità (ogni run salva la sua `config.json`)
- Esperimenti rapidi (cambi `--lr` e rilanci senza toccare il codice)
- Smoke test (`--subset 20` prende solo 20 campioni)

### Blocco B: setup

```python
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(args.seed)
```

Fissiamo il seed per riproducibilità. Stesso seed → stessi risultati. Nota: riproducibilità
completa di PyTorch è complessa (richiede anche `torch.backends.cudnn.deterministic = True`)
ma per il nostro scopo è sufficiente.

```python
run_dir = Path(args.output_dir) / args.run_name
run_dir.mkdir(parents=True, exist_ok=True)
with open(run_dir / "config.json", "w") as f:
    json.dump(vars(args), f, indent=2)
```

Creiamo una cartella unica per questo run (nome basato su timestamp) e salviamo la
configurazione. `vars(args)` converte il `Namespace` di argparse in un dizionario.

### Blocco C: dataset e loader

```python
train_ds = VDDDataset(args.data_root, "train",
                      transform=get_train_transform(args.image_size))
val_ds = VDDDataset(args.data_root, "val",
                    transform=get_eval_transform(args.image_size))

if args.subset > 0:
    k_train = min(args.subset, len(train_ds))
    k_val = min(max(args.subset // 2, 4), len(val_ds))
    train_ds = Subset(train_ds, list(range(k_train)))
    val_ds = Subset(val_ds, list(range(k_val)))
```

`Subset`: wrapper che espone solo certi indici del dataset originale. Con `--subset 20`
prendiamo le prime 20 immagini di train e ~10 di val. Utile per smoke test veloci.

### Blocco D: modello, loss, optimizer, scheduler

```python
model = build_unet(
    num_classes=args.num_classes,
    encoder_name=args.encoder,
    encoder_weights=None if args.no_pretrained else "imagenet",
).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
```

- `.to(device)` sposta il modello su GPU (se disponibile)
- `CrossEntropyLoss` per classificazione multiclasse
- `AdamW` è una variante di Adam con weight decay corretto
- `CosineAnnealingLR`: il learning rate parte da `args.lr` e scende coseno-idalmente a ~0
  nelle `args.epochs` epoche. Aiuta la convergenza finale.

```python
metrics = SegmentationMetrics(args.num_classes, device, class_names)
writer = SummaryWriter(log_dir=str(run_dir / "tb"))
```

Metriche e TensorBoard writer. `SummaryWriter` scrive eventi in `run_dir/tb/` che
TensorBoard leggerà.

### Blocco E: il ciclo principale

```python
for epoch in range(1, args.epochs + 1):
    t0 = time.perf_counter()
    train_loss = train_one_epoch(...)
    val_loss, val_results = validate(...)
    scheduler.step()
    dt = time.perf_counter() - t0

    print(f"[epoch {epoch}] train_loss={train_loss:.4f} ...")

    history.append({...})

    if miou > best_miou:
        best_miou = miou
        torch.save({...}, run_dir / "best.pth")

    with open(run_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)
```

Il loop principale. Per ogni epoch:
1. Train per un'epoch completa
2. Valida
3. Avanza lo scheduler del LR
4. Stampa una riga riassuntiva
5. Appende alla history
6. Se ho migliorato il best mIoU, salvo il checkpoint
7. Riscrivo history.json (se faccio Ctrl+C non perdo i dati già ottenuti)

### Blocco F: funzione `train_one_epoch`

```python
def train_one_epoch(model, loader, optimizer, criterion, device, writer,
                    epoch: int, args):
    model.train()
    total_loss = 0.0
    n_batches = 0
    pbar = tqdm(loader, desc=f"[train] epoch {epoch}", leave=False, ncols=100)
    for i, (images, masks) in enumerate(pbar):
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, masks)
        loss.backward()
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1
        pbar.set_postfix(loss=f"{loss.item():.4f}")

    avg_loss = total_loss / max(n_batches, 1)
    writer.add_scalar("train/loss_epoch", avg_loss, epoch)
    writer.add_scalar("train/lr", optimizer.param_groups[0]["lr"], epoch)
    return avg_loss
```

Il cuore del training. Analisi dettagliata:

- `model.train()`: mette il modello in training mode (BatchNorm usa statistiche del batch,
  Dropout attivo).
- `tqdm`: crea la barra di progresso che hai visto. `leave=False` = la cancella alla fine.
- `.to(device, non_blocking=True)`: trasferisce i tensori su GPU. `non_blocking=True` è
  un'ottimizzazione: il trasferimento è asincrono rispetto al codice CPU.
- `optimizer.zero_grad(set_to_none=True)`: **cruciale**. I gradienti in PyTorch **si
  accumulano** per default. Se non li azzeri, le epoch successive userebbero la somma di
  tutti i gradienti precedenti → disastro. `set_to_none=True` è leggermente più efficiente
  che scrivere zeri.
- `logits = model(images)`: forward pass.
- `loss = criterion(logits, masks)`: calcola la loss.
- `loss.backward()`: **backpropagation**. Calcola il gradiente di loss rispetto a ogni
  parametro del modello.
- `clip_grad_norm_(params, 1.0)`: limita la norma del gradiente a 1.0. Previene esplosioni
  di gradiente.
- `optimizer.step()`: applica l'update dei pesi.
- `loss.item()`: estrae il valore scalare dal tensore (altrimenti `loss` è un tensore
  con grafo computazionale annesso, che consumerebbe memoria).
- `pbar.set_postfix(loss=...)`: aggiorna la loss mostrata nella barra di progresso.
- `writer.add_scalar(...)`: logga su TensorBoard.

### Blocco G: funzione `validate`

```python
@torch.no_grad()
def validate(model, loader, criterion, metrics, device, writer, epoch):
    model.eval()
    metrics.reset()
    total_loss = 0.0
    n_batches = 0
    for images, masks in tqdm(loader, ...):
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        logits = model(images)
        loss = criterion(logits, masks)
        total_loss += loss.item()
        n_batches += 1
        metrics.update(logits, masks)

    avg_loss = total_loss / max(n_batches, 1)
    results = metrics.compute()

    writer.add_scalar("val/loss", avg_loss, epoch)
    writer.add_scalar("val/mIoU", results["mIoU"], epoch)
    writer.add_scalar("val/F1", results["F1"], epoch)
    writer.add_scalar("val/accuracy", results["accuracy"], epoch)
    for cls, iou in results["per_class_iou"].items():
        writer.add_scalar(f"val/iou_{cls}", iou, epoch)

    return avg_loss, results
```

Differenze dal train loop:
- `@torch.no_grad()` come decoratore: disabilita gradient per tutta la funzione.
- `model.eval()`: modalità evaluation.
- `metrics.reset()`: azzera le metriche accumulate all'epoch precedente.
- `metrics.update()` ad ogni batch, poi `metrics.compute()` alla fine.
- **Nessun** `loss.backward()` o `optimizer.step()`: solo calcolo passivo.
- Logging di loss, mIoU, F1, accuracy E IoU per ogni singola classe.

---

## Schema completo del flusso

Quando lanci `python src/training/train_unet.py --epochs 30 --batch-size 8`:

```
1. parse_args()  →  args
2. set seed, crea run_dir, salva config.json
3. Costruisce train_ds, val_ds (VDDDataset)
4. Costruisce train_loader, val_loader (DataLoader)
5. Costruisce model (build_unet + sposta su device)
6. Costruisce criterion, optimizer, scheduler
7. Costruisce metrics, tensorboard writer
8. FOR epoch in 1..30:
     8a. train_one_epoch():
         - model.train()
         - for batch:
              zero_grad → forward → loss → backward → grad clip → step
         - Logga train/loss, train/lr su TB
     8b. validate():
         - model.eval() + no_grad
         - for batch:
              forward → loss → metrics.update
         - metrics.compute() → mIoU, F1, acc, per-class IoU
         - Logga tutto su TB
     8c. scheduler.step()
     8d. if miou > best:  salva best.pth
     8e. Aggiorna history.json
9. Salva last.pth
10. writer.close()
```

Ogni riga del codice ha una ragione di essere. Se qualcosa non ti è chiara, torna qui e
leggi il blocco corrispondente.

---

## Come usare questo documento

1. **Apri in parallelo il file di codice e questa spiegazione**. Leggi una riga di codice
   e poi la spiegazione corrispondente.
2. **Modifica qualcosa e osserva**. Cambia il learning rate, l'encoder, le augmentations.
   Vedi come cambia il training. Niente ti fa capire meglio che vedere le conseguenze.
3. **Scrivi a penna un diagramma della pipeline**. Dal file del dataset all'output di
   TensorBoard. Se riesci a disegnarlo, l'hai capito.
4. **Prova a rispondere a queste domande ad alta voce**:
   - A cosa serve `optimizer.zero_grad()`?
   - Perché converto BGR → RGB in `__getitem__`?
   - Qual è la differenza tra `model.train()` e `model.eval()`?
   - Cos'è il `num_workers` di DataLoader e perché lo teniamo a 0 su Windows?
   - Perché la loss parte circa da `log(num_classes) ≈ 1.95`?

---

*Documento preparato il 22 aprile 2026. Se modifichiamo il codice, aggiorneremo anche qui.*
