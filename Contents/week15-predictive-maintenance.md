# 第 15 週｜論文整併實戰（一）：裝備預測性維護整合研究

> [!NOTE]
> **課程**：AI 大數據分析　**週次**：第 15 週／共 16 週　**時數**：3 小時
> **本週定位**：整合前 14 週方法，示範一份完整「預測性維護」研究從資料到論文的完整流程
> **使用工具**：Google Colab（Python：pandas、scikit-learn、shap）

---

## 0. 本週定位：從「學方法」到「做研究」

前 14 週，每一週聚焦一種方法（迴歸、分群、決策樹、類神經網路、時間序列……），並各自附帶「碩士論文延伸應用」的獨立示範。但一份真正的碩士論文，通常不會只用**一種**方法——而是把資料前處理、預測建模、模型解釋、決策應用**串接成一條完整的研究管線**。本週與下週（第 16 週）是「論文整併實戰」，不再介紹新方法，而是示範**如何把已經學過的方法組裝成一份完整的研究**，並直接對應到論文的章節架構。

本週主題「裝備預測性維護（Predictive Maintenance）」，整合以下模組：

| 研究模組 | 對應課程週次 |
|---|---|
| 資料前處理（標準化、缺失值） | 第 9 週 |
| 分類建模（是否即將需要維護） | 第 10、12 週 |
| 迴歸建模（剩餘使用壽命 RUL 預測） | 第 10、12、14 週 |
| 模型可解釋性（SHAP） | 第 12 週 |
| 決策支援（轉化為維修排程） | 第 6、8 週 |

> [!NOTE]
> 第 16 週會以「智慧庫存動態補給」為主題，示範另一種整合方式（結合第 1 週需求預測、第 5 週安全庫存、第 11 週分群、第 14 週時間序列），讓學員看到同一套方法庫，可以因應不同研究問題組合出不同的整合架構——這正是碩士論文選題與方法論設計最核心的能力。

---

## 1. 學習目標

完成本週課程後，學員應能夠：

1. 說明預測性維護研究的核心問題結構，以及與傳統定期保養（Preventive Maintenance）的差異。
2. 具體理解「依群組（如車輛、裝備）切分訓練/測試集」的方法論重要性，避免資料洩漏。
3. 建立涵蓋分類（故障預警）與迴歸（剩餘壽命預測）的雙軌預測模型。
4. 運用 SHAP 將預測結果轉化為個案層級的可解釋診斷。
5. 將預測結果與排程／容量限制結合，產出具體的維修排程決策建議。
6. 依本週的研究架構，撰寫一份完整碩士論文的章節大綱。

---

## Hour 1｜研究設計與整合架構（60 分鐘）

### 1.0 預測性維護研究問題界定

**傳統定期保養**：不論裝備實際狀態，依固定週期（如每 3 個月）強制保養——簡單但沒有效率：狀態良好的裝備被過度保養（浪費資源），狀態惡化的裝備可能來不及等到排定的保養日期就已故障。

**預測性維護**：運用感測數據**即時**評估裝備的健康狀態，動態決定「這一件裝備該不該優先維護、還能撐多久」。這正是本週研究問題的核心——同時回答兩個問題：

1. **分類問題**：這件裝備**是否**即將需要維護？（對應第 10、12 週的分類方法）
2. **迴歸問題**：這件裝備**還剩下多久**（剩餘使用壽命，Remaining Useful Life, RUL）可以繼續運作？（對應第 10、12、14 週的迴歸／時間序列方法）

兩個問題看似相似，實務用途不同：分類問題適合「快速篩選出高風險清單」；RUL 迴歸問題則能進一步支援「該在哪一天安排維修」的具體排程決策。

---

### 1.1 資料結構的關鍵特性：退化軌跡（Degradation Trajectory）

預測性維護的資料，與前幾週示範的「橫斷面快照」資料（每件裝備一筆觀測值）有本質上的差異——**同一件裝備會被連續追蹤多天，形成一條從服役到故障的完整軌跡**。這種資料結構在學術文獻中稱為「運行至故障資料（Run-to-Failure Data）」，本週模擬資料即採此結構。

> [!IMPORTANT]
> 這個資料結構帶來一個極其重要、卻經常被忽略的方法論陷阱：**同一件裝備、相鄰兩天的感測數據高度相關**（今天的震動值與昨天的震動值幾乎一樣）。若依照前幾週習慣的做法，直接對「列（每一筆觀測值）」隨機切分訓練／測試集，會導致**同一件裝備的資料同時出現在訓練集與測試集中**——模型實質上已經「看過」這件裝備的其他天數據，測試集的評估結果會被高估（資料洩漏）。**正確做法是依「裝備（群組）」切分，確保測試集中的裝備，訓練階段完全沒見過**，本週 Hour 2 會具體示範這個陷阱的實際影響幅度。

---

### 1.2 整合式研究架構

```
原始感測資料（多裝備、多天連續追蹤）
        │
        ▼
   資料前處理（標準化）── 對應第9週
        │
        ▼
  依裝備（而非依列）切分訓練／測試集
        │
        ├─────────────┬─────────────┐
        ▼             ▼             │
  分類模型          迴歸模型          │
（是否即將故障）    （剩餘壽命RUL）    │
 對應第10、12週    對應第10、12、14週  │
        │             │             │
        └──────┬──────┘             │
               ▼                    │
        SHAP可解釋性分析 ── 對應第12週  │
               │                    │
               ▼                    │
     轉化為維修排程優先順序清單 ◄──────┘
        （容量限制下的排程決策）
          對應第6、8週
```

這個架構呈現的正是一份典型碩士論文「研究方法」章節的骨架——**每一個方框，都是一份獨立可以引用文獻方法、卻又彼此串接的模組**。

---

### 1.3 論文章節架構對應

| 論文章節 | 本週對應內容 |
|---|---|
| 第一章　緒論（研究背景、動機、目的） | 1.0 節：定期保養 vs. 預測性維護的問題界定 |
| 第二章　文獻探討 | 各模組方法的相關文獻（分類/迴歸模型、SHAP 可解釋性、維修排程理論），可分別引用第 6、10、12、14 週介紹的方法之原始文獻 |
| 第三章　研究方法 | 1.2 節整合架構圖，逐一說明每個模組採用的方法與理由 |
| 第四章　實證分析與結果 | Hour 2 之完整 Python 實作結果（模型績效比較、SHAP 解釋、排程模擬結果） |
| 第五章　結論與建議 | 綜合各模組發現，提出對實務單位的具體建議與研究限制 |

> [!TIP]
> 許多學員撰寫論文時，容易把「方法」章節寫成「我用了 A 方法、B 方法、C 方法」的**清單式羅列**，卻沒有說明**為什麼**這些方法要串接在一起、彼此之間的資料如何流動。1.2 節的架構圖，正是用來解決這個問題——**畫出模組之間的資料流向，用一句話說明每個箭頭「為什麼」存在**，這是讓研究方法章節具備邏輯完整性、而非方法堆砌的關鍵。

---

## Hour 2｜Python 示範演練：完整預測性維護管線（60 分鐘）

### 2.0 環境設置與退化軌跡模擬資料生成

```python
# ============================================================
# 第15週 Hour 2：完整預測性維護研究管線
# 情境：模擬60輛車從服役到故障的完整退化軌跡
# ============================================================
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib

!apt-get -qq install fonts-noto-cjk > /dev/null 2>&1
matplotlib.rcParams['font.sans-serif'] = ['Noto Sans CJK JT', 'Noto Sans CJK TC']
matplotlib.rcParams['axes.unicode_minus'] = False

np.random.seed(42)
n_vehicles = 60
records = []

for vid in range(1, n_vehicles+1):
    lifetime = np.random.randint(80, 220)  # 此車輛從服役到故障的總天數(每輛車皆不同)
    for day in range(1, lifetime+1):
        frac = day / lifetime  # 生命週期進度(0=剛服役, 1=當天故障)
        # 震動與溫度隨生命週期進度呈非線性上升(愈接近故障、劣化速度愈快)
        vibration = 1.0 + 4.0*(frac**2) + np.random.normal(0, 0.3)
        engine_temp = 85 + 12*(frac**2.5) + np.random.normal(0, 1.5)
        pressure = 100 - 15*(frac**2) + np.random.normal(0, 2)
        mileage_km = 200*day + np.random.normal(0, 20)
        RUL = lifetime - day  # 剩餘使用壽命(天)：迴歸任務的目標變數
        records.append([vid, day, lifetime, mileage_km, engine_temp, vibration, pressure, RUL])

df = pd.DataFrame(records, columns=['vehicle_id','day','lifetime','mileage_km',
                                      'engine_temp_c','vibration_mm_s','pressure_kpa','RUL'])
# 分類任務目標：RUL在15天以內視為「即將需要維護」
df['needs_maintenance_soon'] = (df['RUL'] <= 15).astype(int)

print(f"資料集形狀: {df.shape}（{n_vehicles}輛車，共{df.shape[0]}筆逐日觀測記錄）")
print(f"\n即將需要維護比例: {df['needs_maintenance_soon'].mean()*100:.2f}%（呼應第10週不平衡分類情境）")

# 視覺化：挑選3輛車，觀察其震動值隨生命週期的退化軌跡
plt.figure(figsize=(9, 5))
for vid in [1, 20, 40]:
    sub = df[df['vehicle_id']==vid]
    plt.plot(sub['day'], sub['vibration_mm_s'], label=f'車輛{vid}(總壽命{sub["lifetime"].iloc[0]}天)')
plt.xlabel('服役天數'); plt.ylabel('震動值 (mm/s)')
plt.title('車輛震動值退化軌跡示例')
plt.legend(); plt.grid(alpha=0.3)
plt.show()
```

---

### 2.1 資料前處理與探索（對照第 9 週）

```python
# ============================================================
# 2.1 資料前處理與探索性分析（對照第9週）
# ============================================================

print(df.describe())
print(f"\n缺失值檢查:\n{df.isna().sum()}")

# 相關係數：確認各感測指標與RUL的關聯方向
features = ['mileage_km','engine_temp_c','vibration_mm_s','pressure_kpa']
print(f"\n各特徵與RUL的相關係數:")
print(df[features+['RUL']].corr()['RUL'])
```

---

### 2.2 關鍵方法論驗證：依裝備切分 vs. 依列切分

```python
# ============================================================
# 2.2 資料洩漏陷阱驗證（對照1.1節）
# ============================================================
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

# --- 錯誤做法：依「列」隨機切分 ---
Xtr_wrong, Xte_wrong, ytr_wrong, yte_wrong = train_test_split(
    df[features], df['needs_maintenance_soon'], test_size=0.25,
    random_state=42, stratify=df['needs_maintenance_soon']
)
rf_wrong = RandomForestClassifier(n_estimators=200, max_depth=8, random_state=42, class_weight='balanced')
rf_wrong.fit(Xtr_wrong, ytr_wrong)
auc_wrong = roc_auc_score(yte_wrong, rf_wrong.predict_proba(Xte_wrong)[:,1])
print(f"錯誤做法（依列隨機切分）: AUC={auc_wrong:.4f}")

# --- 正確做法：依「車輛」切分，確保測試集車輛訓練時完全沒見過 ---
vehicle_ids = df['vehicle_id'].unique()
train_ids, test_ids = train_test_split(vehicle_ids, test_size=0.25, random_state=42)
train_df = df[df['vehicle_id'].isin(train_ids)]
test_df = df[df['vehicle_id'].isin(test_ids)]
print(f"訓練集車輛數={len(train_ids)}, 測試集車輛數={len(test_ids)}")

rf_correct = RandomForestClassifier(n_estimators=200, max_depth=8, random_state=42, class_weight='balanced')
rf_correct.fit(train_df[features], train_df['needs_maintenance_soon'])
auc_correct = roc_auc_score(test_df['needs_maintenance_soon'],
                              rf_correct.predict_proba(test_df[features])[:,1])
print(f"正確做法（依車輛切分）: AUC={auc_correct:.4f}")
print(f"\nAUC虛高幅度: {(auc_wrong-auc_correct)*100:.2f}個百分點")
```

> [!CAUTION]
> 執行後會發現兩者的 AUC 差距在本組模擬資料中並不算巨大（約 0.3–0.5 個百分點），這是因為本模擬資料中每輛車的退化軌跡都遵循相近的數學規律，模型不需要「記住特定車輛」也能表現良好。**但在真實世界的資料中，每件裝備經常存在只有該裝備才有的個體差異（如特定的組裝瑕疵、特定的操作習慣），此時依列切分造成的資料洩漏可能遠比本示範更嚴重**。無論差距大小，**依群組切分都是預測性維護研究方法論上唯一站得住腳的做法**，論文中應清楚交代採用此切分方式的理由，這往往是審查者最先檢視的方法論細節之一。

---

### 2.3 分類模型：是否即將需要維護（對照第 10、12 週）

```python
# ============================================================
# 2.3 分類模型比較：邏輯斯迴歸 vs. 隨機森林
# ============================================================
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

Xtr, ytr = train_df[features], train_df['needs_maintenance_soon']
Xte, yte = test_df[features], test_df['needs_maintenance_soon']

scaler = StandardScaler()
Xtr_s = scaler.fit_transform(Xtr)
Xte_s = scaler.transform(Xte)

def eval_clf(model, Xtr_in, Xte_in, name):
    model.fit(Xtr_in, ytr)
    pred = model.predict(Xte_in)
    proba = model.predict_proba(Xte_in)[:, 1]
    print(f"{name}: 準確率={accuracy_score(yte,pred):.4f}, 精確率={precision_score(yte,pred):.4f}, "
          f"召回率={recall_score(yte,pred):.4f}, AUC={roc_auc_score(yte,proba):.4f}")
    return model

lr = eval_clf(LogisticRegression(max_iter=1000, class_weight='balanced'), Xtr_s, Xte_s, "邏輯斯迴歸（第10週）")
rf_clf = eval_clf(RandomForestClassifier(n_estimators=200, max_depth=8, random_state=42,
                                            class_weight='balanced'), Xtr, Xte, "隨機森林（第12週）")
```

**預期輸出**：兩種方法皆能達到極高的 AUC（>0.98），召回率亦相當高（>0.9）——這是因為本週的模擬退化軌跡具有相對規律的函數形式，容易被兩種方法學習到；**在軍事維護情境下，高召回率尤其重要**（呼應第 10 週 3.5 節的討論：漏判一件即將故障的裝備，代價遠高於一次不必要的預防性檢修）。

---

### 2.4 迴歸模型：剩餘使用壽命（RUL）預測（對照第 10、12、14 週）

```python
# ============================================================
# 2.4 迴歸模型：直接預測RUL
# ============================================================
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error

ytr_rul, yte_rul = train_df['RUL'], test_df['RUL']

rf_reg = RandomForestRegressor(n_estimators=200, max_depth=8, random_state=42)
rf_reg.fit(Xtr, ytr_rul)
pred_rul = rf_reg.predict(Xte)

print(f"隨機森林RUL預測: R2={r2_score(yte_rul,pred_rul):.4f}, "
      f"MAE={mean_absolute_error(yte_rul,pred_rul):.2f}天")

# 視覺化：預測RUL vs. 實際RUL散布圖
plt.figure(figsize=(6, 6))
plt.scatter(yte_rul, pred_rul, alpha=0.15, s=10)
plt.plot([0, yte_rul.max()], [0, yte_rul.max()], 'r--', label='完美預測線')
plt.xlabel('實際RUL(天)'); plt.ylabel('預測RUL(天)')
plt.title('RUL預測散布圖')
plt.legend(); plt.grid(alpha=0.3)
plt.show()
```

> [!NOTE]
> 觀察散布圖應可發現：**RUL 越接近 0（即將故障）時，預測誤差通常也越小**（點越貼近對角線），這符合直覺——裝備越接近故障，感測數據的異常訊號越明確，模型越容易準確判斷；反之，裝備剛服役不久時，RUL 的預測不確定性本來就較高（畢竟距離故障還有很長一段時間，中間可能發生的變化更多），這在論文討論章節中是值得specifically說明的重要發現。

---

### 2.5 SHAP 可解釋性分析（對照第 12 週）

```python
# ============================================================
# 2.5 SHAP可解釋性分析：全域重要性與艦隊即時健康報告
# ============================================================
!pip install shap --quiet
import shap

explainer = shap.TreeExplainer(rf_clf)

# 取每輛車最新一筆記錄，模擬「艦隊當前健康狀態即時報告」
latest_status = test_df.sort_values('day').groupby('vehicle_id').tail(1).copy()
shap_values = explainer.shap_values(latest_status[features])
sv = shap_values[:, :, 1] if np.array(shap_values).ndim == 3 else shap_values[1]

mean_abs_shap = np.abs(sv).mean(axis=0)
importance = pd.DataFrame({
    'feature': features, 'SHAP重要性': mean_abs_shap
}).sort_values('SHAP重要性', ascending=False)
print("=== 全域特徵重要性（SHAP）===")
print(importance)

# 個案解釋：找出風險最高的車輛，具體說明原因
latest_status['risk_proba'] = rf_clf.predict_proba(latest_status[features])[:, 1]
top_idx = latest_status['risk_proba'].idxmax()
row_pos = latest_status.index.get_loc(top_idx)
print(f"\n=== 風險最高車輛（車輛{latest_status.loc[top_idx,'vehicle_id']:.0f}）個案解釋 ===")
print(f"預測風險機率={latest_status.loc[top_idx,'risk_proba']:.4f}, 實際RUL={latest_status.loc[top_idx,'RUL']}天")
for feat, val in zip(features, sv[row_pos]):
    print(f"  {feat}: SHAP貢獻={val:+.4f}")
```

**預期輸出**：震動值通常是全域最重要的特徵（因為本週模擬設計中，震動值隨生命週期進度上升最為劇烈），個案解釋能具體呈現「這輛車風險最高，主要因為震動值異常」——**這正是第 12 週強調的「不能只給答案、要給理由」在預測性維護場景的直接體現**：維保人員拿到這份報告，不只知道「哪輛車最該優先處理」，還知道「該優先檢查哪個子系統」。

---

### 2.6 決策支援：轉化為容量限制下的維修排程（對照第 6、8 週）

```python
# ============================================================
# 2.6 決策支援：RUL預測 + 有限維修站容量 => 具體排程建議
# ============================================================

latest_status['predicted_RUL'] = rf_reg.predict(latest_status[features])

# 假設僅有3個維修站容量，每輛車進場保養需5天，依預測RUL由小到大排序(最緊急者優先，呼應第6週EDD概念)
n_bays = 3
service_days_per_vehicle = 5

schedule = latest_status.sort_values('predicted_RUL').reset_index(drop=True)
schedule['scheduled_start_day'] = (schedule.index // n_bays) * service_days_per_vehicle
schedule['will_fail_before_scheduled'] = schedule['scheduled_start_day'] > schedule['RUL']

print("=== 維修排程表（前12輛，依預測RUL排序，3站維修容量）===")
print(schedule[['vehicle_id', 'RUL', 'predicted_RUL', 'scheduled_start_day',
                  'will_fail_before_scheduled']].head(12).to_string(index=False))

n_at_risk = schedule['will_fail_before_scheduled'].sum()
print(f"\n在目前{n_bays}站維修容量下，預計有 {n_at_risk} 輛車會在排入維修前就已故障")
print(f"管理建議：增加維修站容量、或針對這些車輛安排緊急插隊維修（呼應第8週等候線容量規劃概念）")
```

**預期輸出**：這個表格把本週前面所有模組的成果，轉化為一份**指揮官能直接使用的具體行動清單**——不只是「哪些車有風險」，而是「在現有維修站容量下，哪些車來不及排上、需要立即處理」，這正是預測性維護研究從「預測」走向「決策支援」的關鍵一步，也是論文第四章「實證分析」最有說服力的具體產出。

---

## Hour 3（前 40 分鐘）｜延伸練習與論文寫作實戰

### 3.1 延伸練習：不同退化模式裝備之完整流程練習（附完整程式碼）

```python
# ============================================================
# 延伸練習：通信裝備（線性退化模式，對照本週非線性退化模式）
# ============================================================
np.random.seed(77)
n_units = 40
records2 = []
for uid in range(1, n_units+1):
    lifetime = np.random.randint(60, 150)
    for day in range(1, lifetime+1):
        frac = day / lifetime
        # 與Hour2非線性退化不同，此處模擬線性退化模式(訊號強度穩定線性衰退)
        signal_strength = 90 - 40*frac + np.random.normal(0, 3)
        error_rate = 0.5 + 8*frac + np.random.normal(0, 0.5)
        RUL2 = lifetime - day
        records2.append([uid, day, lifetime, signal_strength, error_rate, RUL2])

df2 = pd.DataFrame(records2, columns=['unit_id','day','lifetime','signal_strength','error_rate','RUL'])
df2['needs_maintenance_soon'] = (df2['RUL'] <= 10).astype(int)

features2 = ['signal_strength', 'error_rate']
uids = df2['unit_id'].unique()
train_uids, test_uids = train_test_split(uids, test_size=0.25, random_state=42)
train_df2 = df2[df2['unit_id'].isin(train_uids)]
test_df2 = df2[df2['unit_id'].isin(test_uids)]

rf2 = RandomForestClassifier(n_estimators=200, max_depth=6, random_state=42, class_weight='balanced')
rf2.fit(train_df2[features2], train_df2['needs_maintenance_soon'])
proba2 = rf2.predict_proba(test_df2[features2])[:, 1]
print(f"通信裝備分類: AUC={roc_auc_score(test_df2['needs_maintenance_soon'],proba2):.4f}")

rf_reg2 = RandomForestRegressor(n_estimators=200, max_depth=6, random_state=42)
rf_reg2.fit(train_df2[features2], train_df2['RUL'])
pred_rul2 = rf_reg2.predict(test_df2[features2])
print(f"通信裝備RUL迴歸: R2={r2_score(test_df2['RUL'],pred_rul2):.4f}, "
      f"MAE={mean_absolute_error(test_df2['RUL'],pred_rul2):.2f}天")
```

**詳解**：線性退化模式的資料通常更容易預測（$R^2$ 應高於本週 Hour 2 的非線性退化範例），這是因為線性關係對大多數模型（包含樹狀模型）而言都相對容易學習，這本身也是論文中值得討論的發現——**不同裝備類型的退化模式差異，可能需要不同的建模策略**。

---

### 3.2 論文寫作實戰：從研究管線到五章結構

> [!NOTE]
> 以下說明如何把本週建立的完整研究管線，轉化為一份標準碩士論文的五章結構，並具體對應應該放入哪些圖表。

**第一章　緒論**：以 1.0 節「定期保養 vs. 預測性維護」的問題界定為基礎，說明研究動機（如：現行定期保養制度導致資源浪費或維護不及時的具體案例）、研究目的（建立分類＋迴歸雙軌預測模型，並提出容量限制下的排程決策支援）。

**第二章　文獻探討**：分節回顧（1）預測性維護與 RUL 預測之相關文獻、（2）分類與迴歸模型之方法文獻（可引用第 10、12 週提及的原始方法論文）、（3）SHAP 等可解釋 AI 方法之文獻、（4）維修排程理論文獻（可引用第 6 週）。

**第三章　研究方法**：置入 1.2 節整合架構圖，逐一說明資料來源（若為實際論文，應說明真實資料的蒐集方式）、前處理步驟、採用的分類與迴歸模型、模型評估指標的選擇理由、SHAP 分析方法、排程決策規則的設計邏輯。

**第四章　實證分析與結果**：依序呈現：（1）資料探索性分析圖表（如 2.0 節退化軌跡圖）、（2）分類模型績效比較表（2.3 節）、（3）RUL 迴歸模型績效與散布圖（2.4 節）、（4）SHAP 全域重要性圖與個案解釋（2.5 節）、（5）排程模擬結果與管理意涵（2.6 節）。

**第五章　結論與建議**：總結各模組的主要發現、對實務單位的具體建議（如「建議優先監控震動值指標」「建議增加維修站容量」）、研究限制（如「本研究採模擬資料，未來應以實際感測資料驗證」）、後續研究方向。

> [!TIP]
> **依裝備切分訓練/測試集**（2.2 節）這個方法論細節，建議在第三章明確交代、並在第四章附上如 2.2 節的洩漏效應驗證結果——這是預測性維護類論文最容易被口試委員追問的方法論議題之一，主動說明並提供驗證數據，能大幅提升論文的方法論嚴謹度與說服力。

---

## Hour 3（後 20 分鐘）｜完整碩士論文大綱範例

> [!NOTE]
> 以下提供兩份完整的論文大綱範例，直接示範如何把本週的整合研究架構，包裝成具體、可執行的碩士論文題目。每份大綱皆包含具體的研究問題、章節配置與預期產出的圖表清單，可作為學員規劃自己論文架構時的參考範本。

### 3.3 論文大綱範例一：《以隨機森林為基礎之戰甲車預測性維護與維修排程整合研究》

**研究問題**：現行戰甲車定期保養制度，是否能以感測數據驅動的預測性維護取代或輔助，並在有限維修站容量下提供最適的維修排程建議？

| 章節 | 內容配置 |
|---|---|
| 第一章 緒論 | 1.1 研究背景（現行定期保養制度之限制）；1.2 研究動機與目的；1.3 研究範圍與限制；1.4 論文架構 |
| 第二章 文獻探討 | 2.1 預測性維護與RUL預測文獻回顧；2.2 隨機森林與集成學習方法；2.3 SHAP可解釋AI；2.4 維修排程理論 |
| 第三章 研究方法 | 3.1 研究架構圖（對照本週1.2節）；3.2 資料來源與變數說明；3.3 資料前處理與訓練/測試集切分策略（含依裝備切分之方法論說明）；3.4 分類與迴歸模型設定；3.5 SHAP分析方法；3.6 排程決策規則設計 |
| 第四章 實證分析 | 4.1 資料探索性分析（退化軌跡圖）；4.2 資料洩漏效應驗證（對照2.2節）；4.3 分類模型績效比較（邏輯斯迴歸vs隨機森林）；4.4 RUL迴歸模型績效；4.5 SHAP全域與個案解釋；4.6 容量限制下之排程模擬結果 |
| 第五章 結論與建議 | 5.1 研究發現總結；5.2 對戰甲車保修管理之實務建議；5.3 研究限制；5.4 後續研究建議（如建議延伸至第16週之整合式後勤補給架構） |

**預期產出圖表清單**：退化軌跡圖（3–5輛車示例）、分類模型ROC曲線比較圖、RUL預測散布圖、SHAP特徵重要性長條圖、SHAP個案解釋圖、排程甘特圖（呼應第7週CPM/PERT甘特圖概念）。

---

### 3.4 論文大綱範例二：《整合類神經網路與時間序列方法之通信裝備劣化預警系統研究》

**研究問題**：相較於樹狀模型，類神經網路與時間序列方法（LSTM）能否更準確地捕捉通信裝備的非線性劣化樣態，並提供更精確的預警時機？

| 章節 | 內容配置 |
|---|---|
| 第一章 緒論 | 1.1 通信裝備妥善率對任務遂行之重要性；1.2 研究動機（現行預警機制之不足）；1.3 研究目的與問題 |
| 第二章 文獻探討 | 2.1 通信裝備可靠度與劣化模型文獻；2.2 類神經網路與LSTM方法（對照第13、14週）；2.3 傳統統計方法與深度學習方法之比較文獻 |
| 第三章 研究方法 | 3.1 研究架構圖；3.2 資料生成/蒐集方式；3.3 特徵工程（对照第9週）；3.4 三方模型設計：邏輯斯迴歸（基準）、隨機森林、MLP／LSTM；3.5 模型比較與選擇準則；3.6 正則化策略設計（對照第13週） |
| 第四章 實證分析 | 4.1 資料探索與退化軌跡視覺化；4.2 三方模型分類績效比較；4.3 過度配適與正則化效果驗證（對照第13週2.4節）；4.4 時間序列方法（ARIMA/LSTM）於RUL趨勢預測之應用；4.5 模型選擇之成本效益討論（效能提升 vs. 可解釋性與維運成本，對照第12週3.4節精神） |
| 第五章 結論與建議 | 5.1 各方法適用情境總結；5.2 對通信裝備妥善率管理之建議；5.3 研究限制與後續研究方向 |

**預期產出圖表清單**：訓練/驗證損失收斂曲線圖（對照第13週2.2節）、三方模型ROC曲線比較圖、正則化前後訓練/驗證差距比較表、RUL時間序列預測圖、模型選擇決策矩陣（效能 vs. 可解釋性 vs. 維運成本三維比較表）。

> [!IMPORTANT]
> 這兩份大綱範例的共通點是：**每一節的內容，都能具體對應到本課程某一週的理論或實作**，這正是本課程設計的初衷——**16 週的內容不是互相獨立的知識點，而是一套完整的研究方法工具箱，論文寫作的過程，就是從這個工具箱中挑選適合自己研究問題的工具、並清楚說明挑選的理由**。範例一聚焦「單一最佳模型＋決策應用」的實務導向論文；範例二則聚焦「多方法系統性比較」的方法論導向論文，兩種論文定位皆為碩士論文常見且被接受的類型，學員可依自己的研究興趣與指導教授的期待，選擇適合的定位。

---

## 附錄A：理論常見問答（概念釐清 Q&A）

> [!NOTE]
> **Q1：如果我的實際研究資料，每件裝備只有一筆觀測值（沒有連續追蹤的退化軌跡），2.2 節「依裝備切分」的方法論還適用嗎？**
> A：若資料本身就是橫斷面（每個個體僅一筆觀測值，如第 9–12 週示範的車輛感測資料），則不存在「同一裝備資料同時出現在訓練與測試集」的問題，依列隨機切分即可。**依群組切分的必要性，專門針對「同一個體被重複追蹤多次」的追蹤資料（Panel Data／縱貫資料）**，這是本週資料結構與前幾週資料結構最關鍵的差異，論文中應明確說明自己的資料屬於哪一種結構、並採用對應正確的切分方式。

> [!NOTE]
> **Q2：分類模型（是否即將故障）與迴歸模型（RUL預測）一定要兩個都做嗎？可以只做一個嗎？**
> A：可以只做一個，端看研究問題的重點。若研究重點是「建立一套快速篩選預警機制」，分類模型已經足夠；若研究重點是「精確安排維修排程」，則需要迴歸模型提供具體的天數資訊。本週示範兩者兼具，是為了展示完整的方法論工具箱，實際論文可以依研究問題聚焦其中一種，不需要每次都做全套。

> [!NOTE]
> **Q3：本週的模擬資料是否可以直接當作論文的實證資料？**
> A：不建議。本週的模擬資料是為了**教學示範方法論而設計**，其退化函數形式是研究者事先設定的，不具備真實世界的實證意義。若要撰寫正式論文，應該蒐集真實的感測資料或歷史維修紀錄；若真實資料難以取得，可以參考本週的模擬邏輯，在論文中明確說明「本研究以模擬資料驗證方法可行性，未來應以實際資料進一步驗證」，並將此列為研究限制。

> [!NOTE]
> **Q4：整合這麼多方法，會不會讓論文的方法論部分顯得過於龐雜、缺乏聚焦？**
> A：這是實務上常見的疑慮，解決方式是**明確界定每個模組在整體研究問題中的角色，而非平鋪直敘地羅列方法**。如本週 1.2 節的架構圖所示，每個模組都有清楚的「輸入—處理—輸出」定位，且輸出會成為下一個模組的輸入——只要能說清楚這個資料流動的邏輯，即使用了四、五種方法，論文的方法論部分依然能保持清晰聚焦，而非顯得雜亂拼湊。

---

## 附錄B：本週方法快速索引表

| 任務 | 主要函數／類別 | 所屬套件 |
|---|---|---|
| 依群組切分訓練/測試集 | `train_test_split`（對唯一群組ID操作，而非對列操作） | `sklearn.model_selection` |
| 分類模型 | `LogisticRegression`、`RandomForestClassifier` | `sklearn.linear_model`、`sklearn.ensemble` |
| 迴歸模型 | `RandomForestRegressor` | `sklearn.ensemble` |
| SHAP可解釋性 | `TreeExplainer` | `shap` |
| 依群組取最新記錄 | `df.groupby(...).tail(1)` | `pandas` |

---

## 附錄C：常見易混淆概念澄清

| 容易混淆的概念 | 差異說明 |
|---|---|
| 「橫斷面資料」 vs 「追蹤資料（Panel Data）」 | 橫斷面資料每個個體僅一筆觀測值（第9-13週多數範例）；追蹤資料同一個體被重複追蹤多次（本週退化軌跡資料），兩者訓練/測試集切分策略不同 |
| 「分類任務」 vs 「迴歸任務」於預測性維護情境 | 分類任務回答「是否」即將故障（快速篩選）；迴歸任務回答「還剩多久」（具體排程依據），兩者互補而非互斥 |
| 「模型層級的可解釋性」 vs 「決策層級的可解釋性」 | SHAP提供的是「模型為何如此判斷」的可解釋性；2.6節的排程規則提供的是「為何如此決策」的可解釋性，兩者層次不同，完整的決策支援系統兩者都需要 |

---

## 附錄D：本週與前後課程週次的關聯

| 週次 | 關聯方式 |
|---|---|
| 第 6、8 週：排程與等候線理論 | 本週2.6節之維修排程決策，直接應用第6週優先權排程與第8週容量限制概念 |
| 第 9 週：資料前處理 | 本週資料前處理與依群組切分之方法論，建立於第9週訓練/測試集切分原則之上並進一步延伸 |
| 第 10、12、14 週：預測建模方法 | 本週分類與迴歸模型直接沿用第10、12週方法，並可延伸應用第14週時間序列方法於RUL趨勢預測 |
| 第 16 週：論文整併實戰（二） | 下週將以「智慧庫存動態補給」為主題，示範另一種整合架構，與本週形成兩種不同研究問題的整合範例對照 |

## 附錄E：本週模擬資料集與程式碼彙整

| 資料集 | 用途 | 摘要 |
|---|---|---|
| 車輛退化軌跡資料（60輛車，8890筆） | Hour2主要研究管線 | 分類AUC>0.98，迴歸R2約0.80 |
| 通信裝備線性退化資料（40件，延伸練習） | 延伸練習 | 線性退化模式，對照非線性退化模式之預測難易度差異 |

---

## 參考資料

- 預測性維護與剩餘使用壽命（RUL）預測之研究框架，可參考國際知名的 NASA C-MAPSS 渦輪引擎劣化模擬資料集（PHM 領域標準基準資料集之一），其資料結構（多引擎、多感測器、運行至故障軌跡）與本週模擬資料設計理念一致。
- SHAP 可解釋性方法：Lundberg, S. M., & Lee, S. I. (2017). *A Unified Approach to Interpreting Model Predictions*.

---

*下週課程：第 16 週｜論文整併實戰（二）－智慧庫存動態補給整合研究（最終週）。*
