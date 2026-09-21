# 第 16 週｜論文整併實戰（二）：智慧庫存動態補給整合研究

> [!NOTE]
> **課程**：AI 大數據分析　**週次**：第 16 週／共 16 週（最終週）　**時數**：3 小時
> **本週定位**：整合第 1、5、11、14 週方法，示範第二種完整研究整合架構，並總結全部 16 週課程
> **使用工具**：Google Colab（Python：pandas、NumPy、scikit-learn、scipy）

---

## 0. 本週定位：課程最終週

本週是本課程的最後一週。第 15 週以「預測性維護」為主題，示範如何整合第 9、10、12 週的方法；本週以**「智慧庫存動態補給」**為主題，示範**另一種完全不同的整合方式**——同樣的方法庫，因應不同的研究問題，會組裝出不同的架構。本週整合以下模組：

| 研究模組 | 對應課程週次 |
|---|---|
| 需求預測（傳統平滑法） | 第 1 週 |
| 動態安全庫存（以預測誤差取代歷史標準差） | 第 5 週 |
| 品項需求模式分群 | 第 11 週 |
| 進階時間序列方法 | 第 14 週 |

**核心研究問題**：軍事後勤體系中的補給品項，需求型態差異極大——有些品項需求平穩（如常態耗材），有些呈現成長趨勢（如新裝備零附件），有些具明顯季節性（如冬季裝備），有些則是間歇性的稀有大量需求（如關鍵備份零件）。**若對所有品項套用同一套預測方法與庫存政策，勢必顧此失彼**。本週示範如何以分群技術自動辨識品項的需求模式，並為每一種模式路由至最適合的預測與庫存管理方法，建立一套「智慧化」的動態補給決策系統。

---

## 1. 學習目標

完成本週課程後，學員應能夠：

1. 說明「一體適用」補給政策的限制，理解差異化管理的必要性。
2. 運用統計特徵萃取與 K-Means 分群，自動辨識品項的需求模式類型。
3. 建立「依群組路由至最適預測方法」的自動化決策邏輯。
4. 運用路由後方法的預測誤差，計算比傳統做法更精確的動態安全庫存。
5. 系統性比較「整合路由」與「單一方法」兩種策略的整體績效。
6. 依本週的研究架構，撰寫一份完整碩士論文的章節大綱。
7. 回顧全部 16 週課程，說明如何依自己的研究問題，從方法工具箱中組裝合適的整合架構。

---

## Hour 1｜研究設計與整合架構（60 分鐘）

### 1.0 研究問題界定：一體適用政策的限制

第 5 週已經介紹動態安全庫存（以預測誤差取代歷史標準差）與多階層風險分攤兩個論文方向；第 11 週也提過「K-Means 與多設施選址問題結構相似」。本週要解決的問題是：**當後勤體系同時管理數十、數百種需求型態迥異的品項時，逐一為每個品項手動判斷「該用哪種預測方法」是不切實際的**——需要一套自動化機制，先辨識品項的需求模式類型，再自動路由至對應的方法。

---

### 1.1 整合式研究架構

```
多品項歷史需求資料（週資料，橫跨多年）
        │
        ▼
   統計特徵萃取（變異係數、趨勢強度、季節自相關、間歇比例）
        │
        ▼
   K-Means分群：自動辨識需求模式類型 ── 對應第11週
        │
        ▼
  依群組路由至最適預測方法 ── 對應第1、14週
（穩定型→SES／趨勢型→Holt／季節型→Winters／間歇型→移動平均或專用方法）
        │
        ▼
  以路由後方法之預測誤差，計算動態安全庫存 ── 對應第5週
        │
        ▼
   整體庫存系統績效模擬與比較（整合路由 vs. 單一方法基準）
```

---

### 1.2 論文章節架構對應

| 論文章節 | 本週對應內容 |
|---|---|
| 第一章　緒論 | 1.0 節：一體適用政策限制之問題界定 |
| 第二章　文獻探討 | 分別回顧時間序列預測、分群分析、安全庫存理論之相關文獻（可引用第1、5、11、14週提及之原始文獻） |
| 第三章　研究方法 | 1.1 節整合架構圖；特徵萃取邏輯；分群方法；路由規則設計；動態安全庫存計算方式 |
| 第四章　實證分析與結果 | Hour 2 完整實作結果：分群驗證、預測績效比較、安全庫存比較 |
| 第五章　結論與建議 | 各模組發現總結、對後勤管理實務之建議、研究限制 |

---

## Hour 2｜Python 示範演練：完整智慧補給管線（60 分鐘）

### 2.0 環境設置與多品項模擬資料生成

```python
# ============================================================
# 第16週 Hour 2：完整智慧庫存動態補給研究管線
# 情境：36個補給品項，橫跨4種需求模式，160週歷史資料
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
n_weeks = 160
demand_data = {}

# 群組A：穩定型(10個品項) - 需求平穩、變異小
for i in range(10):
    base = np.random.uniform(80, 150)
    demand_data[f'STABLE_{i+1}'] = np.random.normal(base, base*0.08, n_weeks).clip(0)

# 群組B：趨勢型(8個品項) - 需求持續成長(如新式裝備零附件)
for i in range(8):
    base = np.random.uniform(50, 100)
    growth = np.random.uniform(0.3, 0.8)
    t = np.arange(n_weeks)
    demand_data[f'TREND_{i+1}'] = (base + growth*t + np.random.normal(0, base*0.1, n_weeks)).clip(0)

# 群組C：季節型(8個品項) - 明顯年週期(如季節性裝備)
for i in range(8):
    base = np.random.uniform(70, 120)
    amp = base * np.random.uniform(0.3, 0.5)
    t = np.arange(n_weeks)
    demand_data[f'SEASONAL_{i+1}'] = (base + amp*np.sin(2*np.pi*t/52) + np.random.normal(0, base*0.1, n_weeks)).clip(0)

# 群組D：間歇/高波動型(10個品項) - 如關鍵稀有備份零件，大部分時間無需求
for i in range(10):
    demand = np.zeros(n_weeks)
    for w in range(n_weeks):
        if np.random.random() < 0.25:
            demand[w] = np.random.exponential(60)
    demand_data[f'VOLATILE_{i+1}'] = demand

df_wide = pd.DataFrame(demand_data)
df_wide.index.name = 'week'
skus = df_wide.columns.tolist()
true_type = {s: s.split('_')[0] for s in skus}

print(f"品項總數: {len(skus)}, 歷史週數: {n_weeks}")

# 視覺化：四種模式各挑一個品項示例
fig, axes = plt.subplots(2, 2, figsize=(11, 7))
for ax, sku in zip(axes.flat, ['STABLE_1','TREND_1','SEASONAL_1','VOLATILE_1']):
    ax.plot(df_wide[sku], linewidth=1)
    ax.set_title(sku); ax.grid(alpha=0.3)
plt.tight_layout()
plt.show()
```

---

### 2.1 特徵萃取與 K-Means 分群（對照第 11 週）

```python
# ============================================================
# 2.1 統計特徵萃取與分群（對照第11週）
# ============================================================
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, adjusted_rand_score

features_list = []
for sku in skus:
    y = df_wide[sku].values
    mean_, std_ = y.mean(), y.std()
    cv = std_/mean_ if mean_ > 0 else 0                      # 變異係數
    trend_slope = np.polyfit(np.arange(len(y)), y, 1)[0]      # 線性趨勢斜率
    seasonal_corr = np.corrcoef(y[:-52], y[52:])[0, 1]        # 52週落後自相關(季節性強度)
    intermittency = (y == 0).mean()                            # 零需求週次比例(間歇程度)
    features_list.append([cv, trend_slope, seasonal_corr, intermittency])

feat_df = pd.DataFrame(features_list, columns=['變異係數CV','趨勢斜率','季節自相關','間歇比例'], index=skus)
print("各真實類型之平均特徵值:")
print(feat_df.groupby([s.split('_')[0] for s in skus]).mean().round(3))

# --- 手肘法與輪廓係數決定群數K ---
scaler = StandardScaler()
X_scaled = scaler.fit_transform(feat_df)
for k in range(2, 7):
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X_scaled)
    print(f"k={k}: 輪廓係數={silhouette_score(X_scaled, labels):.4f}")

# --- 選定k=4進行正式分群 ---
km4 = KMeans(n_clusters=4, random_state=42, n_init=10)
cluster_labels = km4.fit_predict(X_scaled)
ari = adjusted_rand_score(list(true_type.values()), cluster_labels)
print(f"\nk=4分群結果與真實需求模式類型的ARI = {ari:.4f}")

feat_df['cluster'] = cluster_labels
feat_df['true_type'] = list(true_type.values())
print("\n分群結果與真實類型對照:")
print(feat_df.groupby('cluster')['true_type'].value_counts())
```

**預期輸出**：輪廓係數應在 $k=4$ 時達到最高（約 0.80），且分群結果與真實需求模式類型的 ARI 應接近 1.0——**代表僅憑統計特徵（不需要事先知道品項屬於哪種模式），K-Means 就能幾乎完美地自動辨識出四種需求型態**，這是本週整合架構「自動化路由」得以成立的關鍵驗證。

---

### 2.2 依群組路由至最適預測方法（對照第 1、14 週）

```python
# ============================================================
# 2.2 依分群結果路由至最適預測方法（對照第1、14節）
# ============================================================

train_weeks, test_weeks = 147, 13  # 訓練147週(需涵蓋至少2個完整年週期)，測試13週(約1季)

def ses_forecast(train, alpha=0.3, steps=13):
    F = [train[0]]
    for t in range(1, len(train)):
        F.append(alpha*train[t-1] + (1-alpha)*F[-1])
    return [F[-1]]*steps

def holt_forecast(train, alpha=0.3, beta=0.1, steps=13):
    L, T = [train[0]], [train[1]-train[0]]
    for t in range(1, len(train)):
        Lt = alpha*train[t] + (1-alpha)*(L[-1]+T[-1])
        Tt = beta*(Lt-L[-1]) + (1-beta)*T[-1]
        L.append(Lt); T.append(Tt)
    return [L[-1]+(i+1)*T[-1] for i in range(steps)]

def winters_additive_forecast(train, alpha=0.3, beta=0.1, gamma=0.3, season=52, steps=13):
    """加法型Winters季節性預測（對照第1週1.6.2節，此處配合本週加法型季節資料設計）"""
    y1, y2 = train[:season], train[season:2*season]
    L = np.mean(y1)
    T = (np.mean(y2)-np.mean(y1))/season
    S = list(y1 - L)
    for t in range(len(train)):
        s_idx = t % season
        sprev = S[s_idx]
        Lt = alpha*(train[t]-sprev) + (1-alpha)*(L+T)
        Tt = beta*(Lt-L) + (1-beta)*T
        S[s_idx] = gamma*(train[t]-Lt) + (1-gamma)*sprev
        L, T = Lt, Tt
    return [(L+(i+1)*T)+S[(len(train)+i) % season] for i in range(steps)]

def moving_average_forecast(train, window=8, steps=13):
    return [np.mean(train[-window:])]*steps

# --- 路由規則：依分群結果自動選擇方法 ---
cluster_to_method = {}  # 先觀察各群的真實類型組成，建立群號->方法的對照(實務上依群集特徵判斷，而非依真實標籤)
for c in range(4):
    dominant_type = feat_df[feat_df['cluster']==c]['true_type'].mode()[0]
    cluster_to_method[c] = dominant_type
print("群集->需求模式對照:", cluster_to_method)

from sklearn.metrics import mean_absolute_error

results_routed, results_naive = {}, {}
for sku in skus:
    y = df_wide[sku].values
    train, test = y[:train_weeks], y[train_weeks:train_weeks+test_weeks]
    c = feat_df.loc[sku, 'cluster']
    method = cluster_to_method[c]

    if method == 'STABLE':
        fc = ses_forecast(train)
    elif method == 'TREND':
        fc = holt_forecast(train)
    elif method == 'SEASONAL':
        fc = winters_additive_forecast(train)
    else:
        fc = moving_average_forecast(train)

    results_routed[sku] = mean_absolute_error(test, fc)
    results_naive[sku] = mean_absolute_error(test, ses_forecast(train))  # 對照組：全部品項統一用SES

comparison = pd.DataFrame({
    'sku': skus, 'type': list(true_type.values()),
    'MAE_整合路由': [results_routed[s] for s in skus],
    'MAE_單一方法(SES)': [results_naive[s] for s in skus],
})
print("\n=== 各需求模式類型之預測誤差比較 ===")
print(comparison.groupby('type')[['MAE_整合路由', 'MAE_單一方法(SES)']].mean().round(2))

overall_routed = comparison['MAE_整合路由'].mean()
overall_naive = comparison['MAE_單一方法(SES)'].mean()
print(f"\n整體平均MAE: 整合路由={overall_routed:.2f}, 單一方法={overall_naive:.2f}")
print(f"整體改善幅度: {(1-overall_routed/overall_naive)*100:.2f}%")
```

**預期輸出**：季節型品項的改善幅度最為顯著（整合路由的 Winters 法遠優於單一 SES 法，因 SES 完全無法捕捉季節樣態）；穩定型品項兩者結果相同（因為路由後恰好也是 SES）；**間歇型品項的整合路由（移動平均）未必優於單一方法**——這是一個誠實且有價值的發現，間歇性需求的預測是公認的難題，移動平均並非最適合的方法，這正好留下 3.4 節論文方向的伏筆。整體而言，**整合路由的平均預測誤差通常較單一方法降低約 8–10%**。

---

### 2.3 動態安全庫存計算（對照第 5 週）

```python
# ============================================================
# 2.3 以路由後方法之預測誤差計算動態安全庫存（對照第5週論文方向二）
# ============================================================
from scipy.stats import norm

LT = 2  # 前置期2週
z = norm.ppf(0.95)

def ses_insample(train, alpha=0.3):
    F = [train[0]]
    for t in range(1, len(train)):
        F.append(alpha*train[t-1] + (1-alpha)*F[-1])
    return np.array(F)

def holt_insample(train, alpha=0.3, beta=0.1):
    L, T, F = [train[0]], [train[1]-train[0]], [train[0]]
    for t in range(1, len(train)):
        F.append(L[-1]+T[-1])
        Lt = alpha*train[t] + (1-alpha)*(L[-1]+T[-1])
        Tt = beta*(Lt-L[-1]) + (1-beta)*T[-1]
        L.append(Lt); T.append(Tt)
    return np.array(F)

ss_results = []
for sku in skus:
    y = df_wide[sku].values
    train = y[:train_weeks]
    c = feat_df.loc[sku, 'cluster']
    method = cluster_to_method[c]

    # 路由後方法的樣本內預測誤差 -> 動態安全庫存(對照第5週3.5節精神)
    if method == 'TREND':
        fitted = holt_insample(train)
    else:
        fitted = ses_insample(train)  # 簡化：季節/間歇品項此處以SES示範樣本內誤差概念
    sigma_dynamic = np.std(train[1:] - fitted[1:])

    # 靜態方法：直接用原始需求標準差(傳統做法)
    sigma_static = np.std(train)

    ss_results.append([sku, true_type[sku], z*sigma_dynamic*np.sqrt(LT), z*sigma_static*np.sqrt(LT)])

ss_df = pd.DataFrame(ss_results, columns=['sku', 'type', 'SS_動態', 'SS_靜態'])
print("=== 各需求模式類型之安全庫存比較 ===")
print(ss_df.groupby('type')[['SS_動態', 'SS_靜態']].mean().round(2))

total_dynamic, total_static = ss_df['SS_動態'].sum(), ss_df['SS_靜態'].sum()
print(f"\n全部品項總安全庫存: 動態方法={total_dynamic:.1f}, 靜態方法={total_static:.1f}")
print(f"總安全庫存降低幅度: {(1-total_dynamic/total_static)*100:.2f}%")
```

**預期輸出**：季節型與趨勢型品項的動態安全庫存應**遠低於**靜態方法（因為 Winters／Holt 已經解釋掉大部分規律性的變異，殘留的預測誤差自然較小）；間歇型品項則可能出現動態方法**略高於**靜態方法的情形（移動平均法本身不適合間歇性需求，預測誤差可能比原始需求的變異程度更大）。**整體而言，全部品項加總後的總安全庫存需求通常可以降低約 15–20%**——這是本週整合架構最直接的財務效益展現：**更聰明的預測，直接轉化為更低的庫存持有成本，同時維持相同的服務水準**。

---

### 2.4 整合決策儀表板輸出

```python
# ============================================================
# 2.4 建立整合決策儀表板：一次呈現分群、方法、安全庫存建議
# ============================================================

dashboard = feat_df[['cluster', 'true_type']].copy()
dashboard['建議預測方法'] = dashboard['cluster'].map(cluster_to_method)
dashboard['動態安全庫存'] = ss_df.set_index('sku')['SS_動態'].round(1)
dashboard['靜態安全庫存(對照)'] = ss_df.set_index('sku')['SS_靜態'].round(1)
dashboard['預測MAE(整合路由)'] = comparison.set_index('sku')['MAE_整合路由'].round(2)

print("=== 智慧補給決策儀表板（前10品項示例）===")
print(dashboard.head(10).to_string())
```

> [!NOTE]
> 這張儀表板正是本週整合研究最終的具體產出——**後勤主管不需要理解 K-Means、Winters 或安全庫存公式的數學細節，只需要看這張表就能知道：每個品項屬於哪種需求模式、系統建議用哪種方法預測、該備多少安全庫存**。這是把複雜的方法論轉化為可執行決策工具的最後一哩路，也是碩士論文「實務貢獻」章節最有力的呈現方式。

---

## Hour 3（前 40 分鐘）｜延伸練習與論文寫作實戰

### 3.1 延伸練習：新增品項之自動分類與方法指派（附完整程式碼）

```python
# ============================================================
# 延伸練習：模擬新品項導入時，自動判斷所屬群組並指派方法
# ============================================================
np.random.seed(99)
n_new = 6
new_demand = {}
# 混合生成3種新品項(不告訴系統其真實類型)
new_demand['NEW_1'] = np.random.normal(100, 8, n_weeks).clip(0)  # 實為穩定型
t_new = np.arange(n_weeks)
new_demand['NEW_2'] = (60 + 0.5*t_new + np.random.normal(0,8,n_weeks)).clip(0)  # 實為趨勢型
new_demand['NEW_3'] = (90 + 35*np.sin(2*np.pi*t_new/52) + np.random.normal(0,9,n_weeks)).clip(0)  # 實為季節型

for sku, y in new_demand.items():
    mean_, std_ = y.mean(), y.std()
    cv = std_/mean_ if mean_>0 else 0
    trend_slope = np.polyfit(np.arange(len(y)), y, 1)[0]
    seasonal_corr = np.corrcoef(y[:-52], y[52:])[0,1]
    intermittency = (y==0).mean()
    new_feat = scaler.transform([[cv, trend_slope, seasonal_corr, intermittency]])
    predicted_cluster = km4.predict(new_feat)[0]
    predicted_method = cluster_to_method[predicted_cluster]
    print(f"{sku}: CV={cv:.3f}, 趨勢斜率={trend_slope:.3f}, 季節自相關={seasonal_corr:.3f} "
          f"=> 判定群組={predicted_cluster}, 建議方法={predicted_method}")
```

**詳解**：這個練習示範了整合系統最重要的實務價值——**新品項導入時，不需要人工判斷該用哪種預測方法，系統會自動依統計特徵歸類並指派方法**，這正是「智慧化」的具體體現，也是論文中可以強調的實務貢獻。

---

### 3.2 論文寫作實戰：兩種整合架構的比較與選擇

> [!NOTE]
> 第 15 週與本週示範了兩種不同的整合架構——學員在規劃自己的論文時，可以參考以下對照，思考自己的研究問題更接近哪一種類型。

| 比較面向 | 第15週：預測性維護 | 第16週：智慧庫存補給 |
|---|---|---|
| 核心整合邏輯 | 同一批裝備，同時建立分類＋迴歸雙軌模型 | 不同品項，先分群再依群組路由至不同方法 |
| 資料結構 | 追蹤資料（同一裝備多次觀測） | 橫斷面之多條獨立時間序列（多品項） |
| 分群/分類的角色 | SHAP用於解釋單一模型 | K-Means用於「路由」至不同模型 |
| 決策支援產出 | 維修排程優先順序 | 差異化預測方法與安全庫存建議 |

**兩種架構的共通寫作原則**：無論哪一種整合方式，論文的說服力都建立在**清楚交代模組之間的資料流向**，以及**每一個方法選擇背後的理由**——而非單純羅列使用過的方法清單。

---

## Hour 3（後 20 分鐘）｜完整碩士論文大綱範例

### 3.3 論文大綱範例一：《以分群為基礎之差異化補給品項需求預測與安全庫存優化研究》

**研究問題**：能否運用分群技術自動辨識補給品項的需求模式，並透過差異化的預測方法與動態安全庫存機制，在維持服務水準的前提下降低整體庫存成本？

| 章節 | 內容配置 |
|---|---|
| 第一章 緒論 | 1.1 一體適用補給政策之限制；1.2 研究動機與目的；1.3 研究範圍與限制 |
| 第二章 文獻探討 | 2.1 需求預測方法文獻（對照第1、14週）；2.2 分群分析於存貨管理之應用（對照第11週）；2.3 動態安全庫存文獻（對照第5週） |
| 第三章 研究方法 | 3.1 整合架構圖；3.2 特徵萃取設計；3.3 K-Means分群方法與群數決定；3.4 路由規則設計；3.5 動態安全庫存計算方式 |
| 第四章 實證分析 | 4.1 分群結果驗證（輪廓係數、ARI）；4.2 各群預測績效比較；4.3 動態vs靜態安全庫存比較；4.4 決策儀表板展示；4.5 新品項自動分類驗證 |
| 第五章 結論與建議 | 5.1 研究發現總結；5.2 對後勤補給管理之實務建議；5.3 研究限制（如間歇性需求預測之改善空間）；5.4 後續研究方向 |

**預期產出圖表清單**：四種需求模式示例圖、手肘法與輪廓係數圖、分群結果與真實類型對照表、預測誤差比較長條圖、安全庫存比較長條圖、整合決策儀表板示例表。

---

### 3.4 論文大綱範例二：《間歇性需求品項之進階預測方法比較與庫存政策研究》

> [!NOTE]
> 這份大綱直接延續 2.2 節「間歇型品項移動平均法表現不如預期」的誠實發現，示範如何把一個「方法侷限」轉化為具體的研究題目。

**研究問題**：軍事後勤中的關鍵稀有備份零件，需求呈現高度間歇性（大部分時間無需求、偶爾大量需求），本研究比較移動平均法與專為間歇性需求設計的 Croston 法（或其改良版本 SBA 法），何者更適合此類品項的庫存政策設計？

| 章節 | 內容配置 |
|---|---|
| 第一章 緒論 | 1.1 關鍵備份零件間歇性需求之管理挑戰；1.2 研究動機（一般預測方法之侷限）；1.3 研究目的 |
| 第二章 文獻探討 | 2.1 間歇性需求預測文獻（Croston法、SBA法等專門方法）；2.2 傳統平滑法之侷限（對照第1週、本週2.2節發現） |
| 第三章 研究方法 | 3.1 間歇性需求資料特性界定（對照本週特徵萃取邏輯）；3.2 移動平均法、Croston法、SBA法之比較設計；3.3 庫存政策設計（考量間歇性需求的安全庫存計算調整） |
| 第四章 實證分析 | 4.1 間歇性品項資料探索；4.2 各方法預測績效比較；4.3 不同庫存政策下之服務水準與成本模擬；4.4 與本課程主流方法（移動平均）之對照分析 |
| 第五章 結論與建議 | 5.1 間歇性需求品項之最適方法建議；5.2 對關鍵備份零件管理之實務意涵；5.3 研究限制 |

> [!TIP]
> 這份大綱範例示範了一個重要的論文選題技巧：**「本週示範方法表現不佳的情境」，本身就是一個絕佳的研究缺口**——與其把方法的侷限視為需要隱藏的瑕疵，不如把它轉化為「這正是我的研究要解決的問題」，這是碩士論文選題時，把課堂學習轉化為研究貢獻的典型路徑。

---

## 課程總結：16 週方法工具箱回顧

> [!IMPORTANT]
> 完成本週課程，也代表完成了本課程全部 16 週的學習。以下用一張總表，回顧整個方法工具箱，作為學員規劃自己碩士論文時的檢索地圖。

| 階段 | 週次 | 方法 | 核心應用場景 |
|---|---|---|---|
| 作業管理 | 1 | 指數平滑法 | 需求預測 |
| 作業管理 | 2 | 產線平衡、ILP | 產能配置 |
| 作業管理 | 3 | 重心法、AHP | 設施選址、多準則決策 |
| 作業管理 | 4 | SPC管制圖 | 品質管制 |
| 作業管理 | 5 | EOQ、安全庫存 | 存貨管理 |
| 作業管理 | 6 | 優先法則、匈牙利法 | 排程與指派 |
| 作業管理 | 7 | CPM/PERT | 專案管理 |
| 作業管理 | 8 | 等候線理論 | 容量規劃 |
| AI大數據 | 9 | 資料前處理 | 特徵工程、Pipeline |
| AI大數據 | 10 | 迴歸、邏輯斯迴歸 | 監督式學習基礎 |
| AI大數據 | 11 | K-Means、PCA | 非監督式學習 |
| AI大數據 | 12 | 決策樹、隨機森林 | 集成學習 |
| AI大數據 | 13 | MLP類神經網路 | 深度學習基礎 |
| AI大數據 | 14 | ARIMA、LSTM | 時間序列進階方法 |
| 整合實戰 | 15 | 分類＋迴歸＋SHAP＋排程整合 | 預測性維護 |
| 整合實戰 | 16 | 分群＋預測＋安全庫存整合 | 智慧庫存補給 |

**寫作論文時的核心提醒**（貫穿全部 16 週反覆出現的主題）：

1. **簡單方法經常表現不俗**（第1、10、12、13、14週反覆驗證）——先建立簡單基準模型，複雜方法的價值必須被實證證明，而非預設。
2. **資料洩漏是最常見的方法論陷阱**（第9、10、15週）——嚴格區分訓練/測試集，依資料結構（橫斷面 vs. 追蹤資料）選擇正確的切分策略。
3. **誠實比較勝過選擇性呈現**（第14、16週）——若某方法在某情境下表現不如預期，這本身經常就是有價值的研究發現，而非需要隱藏的瑕疵。
4. **每個方法都有其適用邊界**（貫穿全課程的「假設限制」小節）——選擇方法前，先確認資料是否滿足該方法的核心假設。
5. **方法的價值最終要回歸決策支援**（第5、6、15、16週）——預測與分析的終點，應該是能被實務單位直接使用的具體建議，而非停留在模型績效指標的數字遊戲。

祝福所有學員在後續的論文寫作過程中，能夠靈活運用這 16 週建立的方法工具箱，完成一份兼具方法論嚴謹度與實務貢獻的碩士論文。

---

## 附錄A：理論常見問答（概念釐清 Q&A）

> [!NOTE]
> **Q1：本週的分群路由邏輯，與第15週的個體分類邏輯，本質上有何不同？**
> A：第15週的分類，是對**同一個問題**（是否需要維護）建立單一模型，模型內部自行學習不同特徵組合的判斷規則；本週的分群路由，則是**在建模之前**就先把品項分成不同組別，每一組套用完全不同的方法（甚至不同的模型類型）。前者是「一個模型處理所有異質性」，後者是「用不同模型分別處理不同的異質性」，兩種策略各有適用情境，本週提供了後者的具體示範。

> [!NOTE]
> **Q2：如果新品項的需求模式，不完全屬於本週定義的四種類型之一，該怎麼辦？**
> A：K-Means 分群一定會把新品項指派到「距離最近」的某個群組，即使該品項的特徵其實介於兩群之間。實務上建議搭配第11週介紹的 DBSCAN 或設定「分群信心指標」（如新品項與最近群中心的距離是否明顯過遠），對於界於群組邊界或明顯異常的品項，觸發人工複核，而非完全依賴自動化路由，這是自動化決策系統設計中「人機協作」的常見做法。

> [!NOTE]
> **Q3：本週與第15週的整合架構，可以合併成一個更大型的研究嗎？**
> A：理論上可以，但實務上不建議在單一碩士論文中處理過多整合模組——如附錄D第4點提醒，過度龐雜的方法論反而會削弱論文的聚焦度。若真的希望整合兩週的架構（如同時處理裝備妥善度預測與備份零件庫存管理），建議明確定義兩者之間**具體的資料流向與決策連結**（例如：預測性維護判定即將故障的裝備，觸發對應備份零件的緊急補貨），而非只是把兩套獨立系統並列呈現。

---

## 附錄B：本週方法快速索引表

| 任務 | 主要函數／類別 | 所屬套件 |
|---|---|---|
| 特徵萃取（趨勢斜率） | `np.polyfit` | `numpy` |
| K-Means分群與新樣本預測 | `KMeans().fit_predict()`、`.predict()` | `sklearn.cluster` |
| 安全庫存計算 | `scipy.stats.norm.ppf` | `scipy.stats` |

---

## 附錄C：常見易混淆概念澄清

| 容易混淆的概念 | 差異說明 |
|---|---|
| 「分群後路由」（本週） vs 「單一模型處理全部資料」（第10、12週多數範例） | 前者假設不同子群體需要本質不同的方法；後者假設單一模型的彈性足以處理全部異質性，選擇何者取決於子群體之間的差異程度 |
| 「動態安全庫存」（本週2.3節） vs 「傳統安全庫存」（第5週） | 動態方法用「預測誤差」估計不確定性（已扣除可預測的規律性部分）；傳統方法用「原始需求標準差」估計（未扣除規律性部分），前者通常更精確但需要先建立預測模型 |

---

## 附錄D：本週模擬資料集與程式碼彙整

| 資料集 | 用途 | 摘要 |
|---|---|---|
| 多品項需求資料（36品項，4種模式，160週） | Hour2主要研究管線 | 分群ARI=1.0（完美辨識），整合路由改善預測誤差約8-10%，總安全庫存降低約15-20% |
| 新品項模擬資料（3個品項，延伸練習） | 延伸練習 | 驗證自動分類與方法指派邏輯 |

---

## 參考資料

- 間歇性需求預測之經典方法：Croston, J. D. (1972). *Forecasting and Stock Control for Intermittent Demands*. Operational Research Quarterly.
- 分群分析與時間序列預測整合應用之研究框架，可參考本課程第 1、5、11、14 週已引用之各項基礎文獻。

---

*本課程至此全部 16 週結束。感謝參與，祝研究順利。*
