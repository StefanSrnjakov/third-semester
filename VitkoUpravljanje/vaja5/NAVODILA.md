# 🚀 Hitri vodič za uporabo notebook-a

## 📝 Kaj je bilo ustvarjeno?

✅ **Vaja5_Multipla_Regresija.ipynb** - Glavni notebook (50 celic)
✅ **README.md** - Podrobna dokumentacija
✅ **expanded_data.csv** - Vhodni podatki (110 vrstic)

## 🎯 Kako začeti?

### 1. Namestite potrebne knjižnice

```bash
pip install pandas numpy matplotlib seaborn scipy scikit-learn statsmodels openpyxl
```

**Ali z conda:**
```bash
conda install pandas numpy matplotlib seaborn scipy scikit-learn statsmodels openpyxl
```

### 2. Odprite notebook

```bash
cd /Users/stefansrjakovferi/predmeti/third-semester/VitkoUpravljanje/vaja5
jupyter notebook Vaja5_Multipla_Regresija.ipynb
```

**Ali z Jupyter Lab:**
```bash
jupyter lab Vaja5_Multipla_Regresija.ipynb
```

**Ali z VS Code:**
- Odprite mapo v VS Code
- Kliknite na `Vaja5_Multipla_Regresija.ipynb`
- Izberite Python kernel

### 3. Zaženite notebook

- **Opcija 1:** Kliknite "Run All" za izvajanje vseh celic
- **Opcija 2:** Izvajajte celice postopoma s Shift+Enter

## 📊 Struktura notebook-a

```
0️⃣ Uvoz knjižnic (celica 2-3)
   └─ Naloži vse potrebne module

1️⃣ Priprava podatkov (celice 4-13)
   ├─ Nalaganje expanded_data.csv
   ├─ Kreiranje novih spremenljivk
   └─ Vizualizacija (korelacije, distribucije)

2️⃣ Gradnja modelov (celice 14-15)

3️⃣ MODEL 1: Linearna regresija - Trajanje (celice 16-21)
   ├─ Specifikacija
   ├─ Gradnja modela
   ├─ Preverjanje predpostavk
   └─ Tabela rezultatov

4️⃣ MODEL 2: Linearna regresija - Skupni čas (celice 22-27)
   ├─ Specifikacija
   ├─ Gradnja modela
   ├─ Preverjanje predpostavk
   └─ Tabela rezultatov

5️⃣ MODEL 3: Logistična regresija - Napake (celice 28-33)
   ├─ Specifikacija
   ├─ Gradnja modela
   ├─ ROC krivulja & Confusion matrix
   └─ Tabela rezultatov (z Odds Ratios)

6️⃣ MODEL 4: Ordinalna regresija - Kategorija (celice 34-39)
   ├─ Specifikacija
   ├─ Gradnja modela
   ├─ Confusion matrix
   └─ Tabela rezultatov (s thresholds)

7️⃣ Primerjava modelov (celici 40-41)
   └─ Tabela primerjave vseh 4 modelov

8️⃣ Interpretacija in zaključki (celica 42)
   └─ Ključne ugotovitve in priporočila

9️⃣ Shranjevanje rezultatov (celica 43-44)
   └─ Export v Excel: rezultati_regresija.xlsx
```

## ✨ Kaj vsak model počne?

### Model 1: Napoved trajanja koraka
- **Y:** Trajanje (h)
- **X:** Čakanje, Je_VA, Napake, Iteracija
- **Rezultat:** Razumevanje, kaj vpliva na trajanje

### Model 2: Napoved skupnega časa
- **Y:** Skupni_cas (h)
- **X:** Je_VA, Je_NVA, Napake, Iteracija, Cakanje_ord
- **Rezultat:** Celoten čas procesa (trajanje + čakanje)

### Model 3: Napoved napak
- **Y:** Ima_napako (0/1)
- **X:** Trajanje, Čakanje, Je_NVA, Iteracija
- **Rezultat:** Verjetnost, da pride do napake

### Model 4: Napoved kategorije trajanja
- **Y:** Trajanje_kategorija (Kratko/Srednje/Dolgo)
- **X:** Čakanje, Je_VA, Napake, Cakanje_ord
- **Rezultat:** Klasifikacija trajanja v kategorije

## 🔍 Kaj preveriti?

Za vsak model je prikazano:

✅ **Koeficienti (β)** - Učinek spremenljivke
✅ **p-vrednosti** - Statistična značilnost
✅ **R² / Pseudo R²** - Delež pojasnjene variance
✅ **AIC/BIC** - Kakovost modela
✅ **Predpostavke:**
   - Normalnost residualov (Shapiro-Wilk)
   - Homoskedastičnost (Breusch-Pagan)
   - Multikolinearnost (VIF)

## 📈 Rezultati

Po zagonu boste dobili:

1. **4 regresijske modele** z vsemi statistikami
2. **Grafe:**
   - Korelacijska matrika
   - Q-Q ploti
   - Residuals ploti
   - ROC krivulja
   - Confusion matrices
3. **Tabele rezultatov** v slovenščini z interpretacijami
4. **Excel datoteko** `rezultati_regresija.xlsx` (5 zavihkov)

## 💡 Interpretacija rezultatov

### Linearna regresija
- **β > 0:** Povečanje X poveča Y
- **β < 0:** Povečanje X zmanjša Y
- **p < 0.05:** Statistično značilen učinek (*)
- **p < 0.01:** Zelo značilen učinek (**)
- **p < 0.001:** Izjemno značilen učinek (***)

### Logistična regresija
- **OR > 1:** Povečuje verjetnost napake
- **OR < 1:** Zmanjšuje verjetnost napake
- **OR = 1:** Ni učinka

### Ordinalna regresija
- **β > 0:** Poveča verjetnost višje kategorije
- **β < 0:** Zmanjša verjetnost višje kategorije

## 🎓 Zahteve vaje

Notebook pokriva vse zahteve iz naloge:

✅ **1. Priprava podatkov**
   - Izbrana ciljna spremenljivka (Y)
   - Ustrezni napovedniki (X)
   - Dodatne kombinirane spremenljivke
   - Opis spremenljivk

✅ **2. Gradnja več modelov**
   - 4 različni modeli
   - 2 različna tipa regresije (linearna, logistična, ordinalna)
   - Preverjanje predpostavk
   - Predstavitev veljavnosti predpostavk

✅ **3. Predstavitev modelov**
   - Tabele z vsemi bistvenimi kazalniki
   - β (nestandardiziran in standardiziran)
   - SE, t/z, p
   - Interpretacije za vsak model

## ⚠️ Možne težave

### Če se knjižnice ne naložijo:
```bash
# Poskusite upgrade pip
pip install --upgrade pip

# Nato ponovno namestite
pip install -r requirements.txt
```

### Če Excel ne dela:
- Ni kritično, rezultate lahko preberete v notebook-u
- Openpyxl ni potreben za analizo, samo za export

### Če se grafi ne prikažejo:
```python
# Dodajte na začetek notebook-a
%matplotlib inline
```

## 📞 Pomoč

Če imate težave:
1. Preverite, da so vse knjižnice nameščene
2. Preverite, da je `expanded_data.csv` v isti mapi
3. Zaženite celice po vrsti (ne preskakujte)
4. Preglejte README.md za dodatne informacije

---

**Uspešno delo! 🎉**

Če imate vprašanja ali potrebujete dodatne prilagoditve, se obrnite na svojega asistenta.



