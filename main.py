import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score

plt.rcParams['figure.dpi'] = 80
plt.rcParams['figure.figsize'] = (3, 2)
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 7
plt.rcParams['axes.titlesize'] = 8
plt.rcParams['axes.labelsize'] = 7
plt.rcParams['xtick.labelsize'] = 6
plt.rcParams['ytick.labelsize'] = 6

df = pd.read_csv('CEAS_08.csv')

print("=" * 60)
print("OSNOVNE INFORMACIJE O DATASETU")
print("=" * 60)
print(df.head())
df.info()
print(df.describe())
print("Broj duplikata:", df.duplicated().sum())
print("Nedostajuće vrijednosti:\n", df.isnull().sum())

df = df.dropna()
df = df.drop_duplicates()
print(f"\nBroj redova nakon čišćenja: {len(df)}")
print(f"Legitimnih poruka: {(df['label'] == 0).sum()}")
print(f"Phishing poruka:   {(df['label'] == 1).sum()}")

def extract_features(text):
    if not isinstance(text, str):
        text = str(text)
    return {
        'duzina_teksta'      : len(text),
        'broj_rijeci'        : len(text.split()),
        'broj_url'           : text.lower().count('http'),
        'broj_email_adresa'  : text.count('@'),
        'broj_uzvicnika'     : text.count('!'),
        'broj_upitnika'      : text.count('?'),
        'broj_brojeva'       : sum(c.isdigit() for c in text),
        'udio_velikih_slova' : sum(c.isupper() for c in text) / max(len(text), 1),
        'kljucne_rijeci'     : sum(text.lower().count(w) for w in [
            'click', 'verify', 'account', 'urgent', 'password',
            'free', 'winner', 'limited', 'offer', 'congratulations',
            'bank', 'update', 'confirm', 'login', 'suspend'
        ])
    }

print("\nIzvlačim karakteristike...")
features_df = df['body'].apply(extract_features).apply(pd.Series)
features_df['urls_postojeci'] = df['urls'].values
df_full = pd.concat([df[['label']], features_df], axis=1)
df_full['is_phishing'] = df_full['label'].astype(int)
feature_cols = list(features_df.columns)
print("Gotovo.")

# Grafik 1 — Distribucija klasa
fig, ax = plt.subplots(figsize=(4, 3))
counts = df_full['is_phishing'].value_counts().sort_index()
bars = ax.bar(['Legitimni', 'Phishing'], counts.values,
              color=['#2ecc71', '#e74c3c'], edgecolor='black', linewidth=0.8)
for bar, val in zip(bars, counts.values):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 30,
            f'{val:,}', ha='center', va='bottom', fontsize=8, fontweight='bold')
ax.set_title('Distribucija klasa u datasetu', fontweight='bold')
ax.set_ylabel('Broj e-poruka')
ax.set_xlabel('Klasa')
plt.tight_layout()
plt.savefig('slika1.png', dpi=300, bbox_inches='tight')
plt.show()

# Grafik 2 — Prosječne vrijednosti karakteristika
fig, ax = plt.subplots(figsize=(6, 3))
mean_features = df_full.groupby('is_phishing')[feature_cols].mean()
mean_features.T.plot(kind='bar', ax=ax, color=['#2ecc71', '#e74c3c'],
                     edgecolor='black', linewidth=0.6)
ax.set_title('Prosječne vrijednosti karakteristika: Legitimni vs Phishing', fontweight='bold')
ax.set_xlabel('Karakteristika')
ax.set_ylabel('Prosječna vrijednost')
ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha='right', fontsize=7)
ax.legend(['Legitimni (0)', 'Phishing (1)'], fontsize=7)
plt.tight_layout()
plt.savefig('slika2.png', dpi=300, bbox_inches='tight')
plt.show()

# Grafik 3 — Korelacijska matrica
fig, ax = plt.subplots(figsize=(5, 4))
corr = df_full[feature_cols + ['is_phishing']].corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='RdYlGn',
            center=0, linewidths=0.5, ax=ax, annot_kws={'size': 6})
ax.set_title('Korelacijska matrica karakteristika', fontweight='bold')
plt.tight_layout()
plt.savefig('slika3.png', dpi=300, bbox_inches='tight')
plt.show()

# Grafik 4 — Distribucija dužine teksta
fig, ax = plt.subplots(figsize=(4, 3))
data_leg = df_full.loc[df_full['is_phishing']==0, 'duzina_teksta']
data_phi = df_full.loc[df_full['is_phishing']==1, 'duzina_teksta']
ax.hist(data_leg, bins=50, range=(0, 5000), alpha=0.6, color='#2ecc71', label='Legitimni', edgecolor='none')
ax.hist(data_phi, bins=50, range=(0, 5000), alpha=0.6, color='#e74c3c', label='Phishing', edgecolor='none')
ax.axvline(data_leg.median(), color='#27ae60', linestyle='--', linewidth=1.2, label=f'Medijan leg. ({int(data_leg.median())})')
ax.axvline(data_phi.median(), color='#c0392b', linestyle='--', linewidth=1.2, label=f'Medijan phi. ({int(data_phi.median())})')
ax.set_title('Distribucija dužine teksta', fontweight='bold')
ax.set_xlabel('Broj karaktera (0–5000)')
ax.set_ylabel('Broj e-poruka')
ax.legend(fontsize=6)
plt.tight_layout()
plt.savefig('slika4.png', dpi=300, bbox_inches='tight')
plt.show()

# =============================================================================
# 5. MODELI MAŠINSKOG UČENJA
# =============================================================================

X = df_full[feature_cols]
y = df_full['is_phishing']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print(f"\nTrening skup: {len(X_train)} | Test skup: {len(X_test)}")

modeli = {
    'Logistička regresija': LogisticRegression(max_iter=1000, random_state=42),
    'Random Forest'       : RandomForestClassifier(n_estimators=100, random_state=42)
}

rezultati = {}
for naziv, model in modeli.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, model.predict_proba(X_test)[:,1])
    rezultati[naziv] = {'accuracy': acc, 'auc': auc, 'model': model, 'y_pred': y_pred}
    print(f"\n--- {naziv} ---")
    print(f"Accuracy: {acc:.4f} | AUC: {auc:.4f}")
    print(classification_report(y_test, y_pred, target_names=['Legitimni', 'Phishing']))

# Grafik 5 — Confusion Matrix
fig, axes = plt.subplots(1, 2, figsize=(7, 3))
for ax, (naziv, res) in zip(axes, rezultati.items()):
    cm = confusion_matrix(y_test, res['y_pred'])
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=['Legitimni', 'Phishing'],
                yticklabels=['Legitimni', 'Phishing'],
                annot_kws={'size': 8})
    ax.set_title(f'Confusion Matrix\n{naziv}', fontweight='bold')
    ax.set_ylabel('Stvarna klasa')
    ax.set_xlabel('Predviđena klasa')
plt.tight_layout()
plt.savefig('slika5.png', dpi=300, bbox_inches='tight')
plt.show()

# Grafik 6 — Važnost karakteristika
rf_model = rezultati['Random Forest']['model']
importances = pd.Series(rf_model.feature_importances_, index=feature_cols).sort_values(ascending=True)
colors = ['#e74c3c' if i >= len(importances)-3 else '#3498db' for i in range(len(importances))]
fig, ax = plt.subplots(figsize=(5, 3))
importances.plot(kind='barh', ax=ax, color=colors, edgecolor='black', linewidth=0.5)
ax.set_title('Važnost karakteristika — Random Forest', fontweight='bold')
ax.set_xlabel('Važnost (Gini)')
plt.tight_layout()
plt.savefig('slika6.png', dpi=300, bbox_inches='tight')
plt.show()

print("\n" + "=" * 60)
print("REZULTATI")
print("=" * 60)
best = max(rezultati.items(), key=lambda x: x[1]['auc'])
print(f"Najbolji model : {best[0]}")
print(f"Accuracy       : {best[1]['accuracy']:.4f}")
print(f"AUC            : {best[1]['auc']:.4f}")
top3 = importances.sort_values(ascending=False).head(3).index.tolist()
print(f"Top 3 prediktori: {', '.join(top3)}")