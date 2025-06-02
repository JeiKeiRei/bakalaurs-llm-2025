# Valodas Modeļu Pielietojumi Depresijas Atpazīšanā un Ārstēšanā

Bakalaura darbs par lielo valodas modeļu (LLM) izmantošanu depresijas un pašnāvniecisko domu klasifikācijā sociālo mediju tekstos.

**Autors:** Artūrs Ābele  
**Darba vadītājs:** Maksims Ivanovs, Ph.D
**Universitāte:** Latvijas Universitāte, Eksakto zinātņu un tehnoloģiju fakultāte Datorikas nodaļa
**Gads:** 2025

## Pētījuma apraksts

Šis pētījums salīdzina lokāli darbināmu un mākoņa valodas modeļu efektivitāti depresijas un pašnāvniecisko domu atšķiršanā, izmantojot SDCNL (Suicide vs Depression Classification) datu kopu no Reddit sociālās platformas.

### Testētie modeļi:
- **Mākoņa modelis:** ChatGPT-4o mini
- **Lokālie modeļi:**
  - DeepSeek-v2 (16B)
  - MedLLaMA2 (7B) 
  - Qwen3 (1.7B)

## Galvenie rezultāti:
- Labākais rezultāts: Qwen3 neapmācīts (F1=0.836).
- Lokālie modeļi brīžiem var pārspēt mākoņa risinājumus.
- Mākoņa modeļi uzrāda stabilāku rezultātu nekā lokālie modeļi, kas padara tos vairāk uzticamus.

## Instalācija un iestatīšana

### Sistēmas prasības

**Minimālās prasības:**
- Python 3.8+
- 16GB RAM (ieteicami 32GB)
- GPU ar 8GB+ VRAM (lokālajiem modeļiem)
- 50GB+ brīvās diska vietas

### 1. Repozitorija klonēšana

```bash
git clone https://github.com/JeiKeiRei/bakalaurs-llm-2025.git
cd bakalaurs-llm-2025
```

## 2. Python bibliotēku instalācija

```bash
pip install pandas numpy scikit-learn matplotlib seaborn
```

## 3. OLLAMA platformas instalācija (lokālajiem modeļiem)

**Linus/macOS**
```bash
curl -fsSL https://ollama.ai/install.sh | sh
```

**Arch Linux (pacman):**
```bash
sudo pacman -S ollama
```

## 4. Lokālo modeļu lejupielāde
```bash
# Palaist OLLAMA servisu
sudo systemctl start ollama

# Lejupielādēt modeļus
ollama pull deepseek-v2:16b    # ~24GB
ollama pull medllama2:7b       # ~12GB  
ollama pull qwen3:1.7b         # ~3GB
```

## Izmantošana

### 1. Datu sagatavošana

```bash
python scripts/prepare_ai_train_test_files.py
```
Šis skripts:
- Izveido 30 apmācības piemērus
- Sagatavo 50 testa paraugus
- Ģenerē failus trained/untrained testēšanai

### 2. Lokālo modeļu testēšana

```bash
python scripts/train_test_local_ai.py
```
Šis skripts automātiski testē visus lokālos modeļus gan ar, gan bez apmācības piemēriem.

### 3. Mākoņa modeļa testēšana (ChatGPT)

**Manuāla testēšana caur ChatGPT web interface:**
- Atveriet jaunu ChatGPT sesiju
- Bez apmācības: iekopējiet comparison_test_5431_test_samples_no_training.txt
- Ar apmācību: vispirms comparison_test_5431_training_examples.txt, tad comparison_test_5431_test_samples_with_training.txt
- Manuāli ierakstiet rezultātus atbilstošajos CSV failos


### 4. Rezultātu analīze
```bash
python scripts/comprehensive_analysis.py
```
Šis skripts ģenerē:
- Veiktspējas metrikas visiem modeļiem
- Salīdzinošas vizualizācijas
- Detalizētu analīzes atskaiti

## Galvenie rezultāti

| Modelis | Apstākļi | F1 rezultāts | Precizitāte | Jūtība |
|---------|----------|:------------:|:-----------:|:------:|
| **Qwen3 (1.7B)** | Neapmācīts | **0.836** | 0.767 | 0.920 |
| MedLLaMA2 (7B) | Apmācīts | 0.677 | 0.550 | 0.880 |
| DeepSeek-v2 (16B) | Apmācīts | 0.667 | 0.696 | 0.640 |
| ChatGPT 4o mini | Abi | 0.667 | 0.750 | 0.600 |

**Labākais rezultāts:** Qwen3 neapmācīts ar F1=0.836  
**Stabilākais:** ChatGPT 4o mini ar konsekventiem rezultātiem  
**Neuzticamākais:** Qwen3 (0.000 → 0.836 atkarībā no apmācības)

## Vizualizācijas

Repozitorijā pieejamas šādas analīzes:
1. **Modeļu veiktspējas salīdzinājums** - Četru metriku salīdzinājums
2. **Apmācības ietekmes analīze** - Karstuma karte ar uzlabojumiem
3. **☁Lokālie vs Mākoņa modeļi** - Izvietošanas veidu salīdzinājums
4. **Modeļa izmēra analīze** - Parametru skaita ietekme uz veiktspēju

## Svarīgi ierobežojumi

- **Tehniskās prasības:** Lieli modeļi prasa daudz GPU atmiņas
- **Datu apjoms:** Eksperimenti veikti ar ierobežotu paraugu skaitu (50)
- **Valoda:** Visi eksperimenti angļu valodā
- **Ētikas apsvērumi:** Izmanto publisko, anonimizēto datu kopu

## Datu kopa

Izmantota **SDCNL** (Suicide vs Depression Classification) datu kopa:

- **Avots:** Haque et al. (2021) - [SDCNL GitHub](https://github.com/ayaanzhaque/SDCNL)
- **Saturs:** 1,895 Reddit ieraksti no r/SuicideWatch un r/Depression
- **Licenza:** Akadēmiskai izmantošanai

## Saites

- **Oriģinālais SDCNL pētījums:** [arxiv.org/abs/2102.09427](https://arxiv.org/abs/2102.09427)
- **SDCNL datu kopa:** [github.com/ayaanzhaque/SDCNL](https://github.com/ayaanzhaque/SDCNL)
- **OLLAMA platforma:** [ollama.ai](https://ollama.ai)

## Datu kopas atsauce

```bibtex
@inproceedings{haque2021deep,
 title={Deep Learning for Suicide and Depression Identification with Unsupervised Label
Correction},
 author={Ayaan Haque and Viraaj Reddi and Tyler Giallanza},
 booktitle={ICANN 2021},
 year={2021},
 url={https://github.com/ayaanzhaque/SDCNL}
}
```


