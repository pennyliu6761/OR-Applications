# 第 14 週｜時間序列分析與序列模型（ARIMA & LSTM）

> [!NOTE]
> **課程**：AI 大數據分析　**週次**：第 14 週／共 16 週　**時數**：3 小時
> **本週關鍵字**：平穩性、ADF 檢定、差分、ACF／PACF、ARIMA、AIC、LSTM、記憶單元、閘控機制
> **使用工具**：Google Colab（Python：statsmodels、TensorFlow／Keras、scikit-learn）

---

## 0. 本週課程時間分配

| 節次 | 時間 | 內容 |
|---|---|---|
| Hour 1 | 60 分鐘 | 理論深度：平穩性、ACF/PACF、ARIMA、模型選擇、LSTM 架構 |
| Hour 2 | 60 分鐘 | Python 示範演練：ARIMA 建模診斷、LSTM 序列預測、與第 1 週傳統平滑法三方比較 |
| Hour 3（前 40 分鐘） | 40 分鐘 | 延伸練習（兩組模擬資料，附完整程式碼）＋ 綜合案例研究 |
| Hour 3（後 20 分鐘） | 20 分鐘 | **碩士論文延伸應用**：兩個可行論文方向之主題定義、方法說明、模擬資料與 Python 實作展示 |

> [!NOTE]
> 本週正式回應第 1 週遺留的伏筆——當時課程尚在「作業管理專題」階段，只能以傳統指數平滑法處理需求預測；本週學員已具備完整的統計推論與深度學習基礎，可以正式引入 ARIMA 與 LSTM，並與第 1 週的方法進行嚴謹的三方比較。

---

## 1. 學習目標

完成本週課程後，學員應能夠：

1. 說明時間序列平穩性的意義，並運用 ADF 檢定與差分處理非平穩序列。
2. 運用 ACF、PACF 圖判斷 ARIMA 模型的合理階數。
3. 建立 ARIMA 模型，並運用 AIC 準則進行模型選擇。
4. 說明 LSTM 相對於傳統 RNN 的改良之處，理解記憶單元與三個閘控機制的運作邏輯。
5. 在 Python 中建立 ARIMA 與 LSTM 模型進行時間序列預測，並與第 1 週的傳統平滑法比較。
6. 說出至少兩個可將本週方法延伸為碩士論文題目的具體方向，並理解其對應的進階方法與實作方式。

---

## Hour 1｜理論深度（60 分鐘）

### 1.0 時間序列分析導論：從第 1 週到本週

第 1 週介紹的指數平滑法（SES、Holt、Winters），本質上是一種「加權平均」的預測邏輯——用過去觀測值的加權組合預測未來，權重隨時間呈指數遞減。這類方法計算簡單、參數意義明確，但**沒有嚴謹的統計理論基礎**（例如無法進行正式的統計顯著性檢定、無法提供理論完備的預測區間）。本週介紹的 **ARIMA** 模型，建立在嚴謹的統計理論之上，能明確描述時間序列的自我相關結構；**LSTM** 則是深度學習處理序列資料的代表性架構，能捕捉傳統統計模型難以描述的複雜非線性樣態。

---

### 1.1 平穩性與差分

**平穩性（Stationarity）**：一個時間序列若滿足平均數、變異數、自我共變異數皆不隨時間改變，稱為（弱）平穩序列。**ARIMA 模型的自迴歸與移動平均部分，都要求序列必須是平穩的**，這是使用 ARIMA 前必須確認的前提。

**單根檢定（ADF Test, Augmented Dickey-Fuller Test）**：正式檢定序列是否平穩的統計方法，虛無假設為「序列存在單根（非平穩）」：

$$
H_0: \text{序列非平穩（存在單根）} \quad vs. \quad H_1: \text{序列平穩}
$$

若 $p$ 值小於顯著水準（如 0.05），拒絕虛無假設，認定序列平穩。

**差分（Differencing）**：若序列非平穩（如存在趨勢），可透過差分消除趨勢，使其轉為平穩：

$$
y_t' = y_t - y_{t-1}
$$

若一次差分仍不平穩，可再次差分（二階差分），但實務上很少需要超過二階。

> [!NOTE]
> 第 1 週的油料消耗資料具有明顯的上升趨勢，若直接對原始數值套用 ARIMA 中的自迴歸估計，會產生統計理論上不成立的結果——這正是為什麼 ARIMA 名稱中的「I」（Integrated，整合）格外重要：**它負責先將非平穩序列差分為平穩序列，模型配適完成後，再將差分還原（積分）回原始尺度**，讓使用者不需要手動進行差分與還原，`statsmodels` 會自動處理這個過程。

---

### 1.2 自我相關與偏自我相關：ACF 與 PACF

**自我相關函數（Autocorrelation Function, ACF）**：衡量序列與其落後 $k$ 期的自身之間的相關程度：

$$
\rho_k = \frac{Cov(y_t, y_{t-k})}{Var(y_t)}
$$

**偏自我相關函數（Partial Autocorrelation Function, PACF）**：衡量序列與其落後 $k$ 期的相關程度，**扣除**中間各期已經解釋的部分後的「淨」相關程度。

**判斷 ARIMA 階數的經驗法則**：

| 圖形樣態 | 建議模型 |
|---|---|
| ACF 快速衰減至零、PACF 在落後 $p$ 期後截斷 | AR($p$) 模型，即 ARIMA($p$,d,0) |
| PACF 快速衰減至零、ACF 在落後 $q$ 期後截斷 | MA($q$) 模型，即 ARIMA(0,d,$q$) |
| 兩者皆緩慢衰減，無明顯截斷 | 可能需要 ARMA 混合模型，或需要進一步差分 |

> [!TIP]
> 這個經驗法則在實務資料中經常不夠明確（ACF、PACF 圖形不總是教科書般乾淨），本週 Hour 2 示範會展示更實務的做法：**與其單憑肉眼判讀 ACF/PACF 圖形，不如系統性嘗試多組候選階數，以 1.4 節的 AIC 準則客觀選擇最佳模型**，這是現代時間序列分析的標準做法。

---

### 1.3 ARIMA 模型

**ARIMA($p$, $d$, $q$)** 整合了三個部分：

$$
\underbrace{y_t' = c+\phi_1 y_{t-1}'+\cdots+\phi_p y_{t-p}'}_{AR(p)：自迴歸部分} + \underbrace{\theta_1\varepsilon_{t-1}+\cdots+\theta_q\varepsilon_{t-q}}_{MA(q)：移動平均部分} + \varepsilon_t
$$

其中 $y_t'$ 為經過 $d$ 階差分後的序列， $p$ 為自迴歸階數（使用過去幾期的數值）、 $q$ 為移動平均階數（使用過去幾期的預測誤差）。

- **AR（自迴歸）部分**：假設當期數值與過去數值存在線性關係，類似「動量」的概念。
- **MA（移動平均）部分**：假設當期數值受過去「預測誤差（意外衝擊）」的影響，捕捉衝擊的延續效果。
- **I（整合）部分**：如 1.1 節所述，透過差分處理非平穩性。

---

### 1.4 模型選擇：AIC 與 BIC

若不確定最佳的 $(p,d,q)$ 組合，可系統性嘗試多組候選模型，以**訊息準則（Information Criterion）**選擇最佳模型：

$$
AIC = -2\ln(L) + 2k
$$

其中 $L$ 為概似函數值（模型配適度）， $k$ 為模型參數個數。AIC 在「配適度」與「模型複雜度」之間取得平衡——**與第 10 週調整後 $R^2$ 、第 12 週正則化的核心精神完全一致**：懲罰過度複雜的模型，避免單純因為參數變多而「看起來」配適較好。AIC 越小代表模型越好。BIC（Bayesian Information Criterion）邏輯相近，但對參數個數的懲罰力道更重，傾向選擇更精簡的模型。

---

### 1.5 季節性 ARIMA（SARIMA）簡介

若資料存在季節性（如月資料的年週期），可延伸為 **SARIMA($p$,$d$,$q$)($P$,$D$,$Q$)$_m$**，額外加入季節性的自迴歸、差分、移動平均項， $m$ 為季節週期長度（月資料通常為 12）。SARIMA 的參數估計較為複雜，本課程不深入推導，但 `statsmodels` 的 `SARIMAX` 函數已完整支援，使用邏輯與一般 ARIMA 相同，僅需額外指定季節性參數。

---

### 1.6 LSTM 循環神經網路導論

**循環神經網路（Recurrent Neural Network, RNN）** 是第 13 週 MLP 的延伸——RNN 在處理序列的每一個時間點時，會將「前一個時間點的隱藏狀態」也一併輸入，讓網路具備某種「記憶」前面序列資訊的能力。但標準 RNN 存在嚴重的**梯度消失問題**（如第 13 週 1.2 節介紹）：序列越長，越早期的資訊對梯度的影響會以指數速度衰減，導致標準 RNN 難以學習長距離的相依關係。

**LSTM（Long Short-Term Memory，長短期記憶網路）** 由 Hochreiter 與 Schmidhuber 於 1997 年提出，透過引入**記憶單元（Cell State）**與**三個閘控機制（Gates）**，讓網路能夠選擇性地「記住」或「遺忘」資訊，大幅緩解了標準 RNN 的梯度消失問題。

---

### 1.7 LSTM 架構細節

| 閘控機制 | 作用 |
|---|---|
| **遺忘閘（Forget Gate）** | 決定記憶單元中哪些過去的資訊應該被「遺忘」 |
| **輸入閘（Input Gate）** | 決定當前時間點的新資訊，有多少應該被寫入記憶單元 |
| **輸出閘（Output Gate）** | 決定記憶單元中的資訊，有多少應該被輸出作為當前時間點的隱藏狀態 |

三個閘門都是以 Sigmoid 函數輸出介於 $[0,1]$ 的「開放程度」， $0$ 代表完全關閉（資訊完全不通過）、 $1$ 代表完全開放。透過這種機制，LSTM 的記憶單元能夠讓重要資訊跨越許多時間步驟持續傳遞，而不會像標準 RNN 一樣快速衰減。

> [!NOTE]
> 對照第 13 週的 Dropout（訓練時隨機關閉神經元）——LSTM 的閘控機制同樣是一種「選擇性開關資訊流動」的設計理念，但目的不同：Dropout 是為了正則化、避免過度配適；LSTM 的閘門是為了解決長序列的梯度消失問題、讓網路能有效學習長期依賴關係。兩者是深度學習中「用門控/選擇機制解決不同問題」的兩個代表性範例。

---

### 1.8 時間序列資料前處理：滑動窗口

LSTM（以及任何監督式學習模型）需要「輸入—輸出」配對的訓練樣本，但時間序列本身是單一連續的數列，須透過**滑動窗口（Sliding Window）**手法轉換：以過去 $w$ 期的數值作為輸入序列，下一期的數值作為預測目標，窗口逐期往後滑動，產生大量訓練樣本。

$$
(y_1,\ldots,y_w) \rightarrow y_{w+1}, \quad (y_2,\ldots,y_{w+1}) \rightarrow y_{w+2}, \quad \ldots
$$

**多步預測**：由於測試期的未來真實值未知，多步預測須採**遞迴方式**——用模型對第一期的預測值，滾動加入輸入窗口，作為預測下一期的依據，如此遞迴至完成所有預測期數（呼應第 1 週、第 9 週已介紹過的遞迴多步預測概念）。

> [!IMPORTANT]
> LSTM 對輸入資料的尺度同樣敏感（呼應第 13 週），時間序列資料在建立滑動窗口之前，務必先標準化（常用 Min-Max 縮放至 $[0,1]$ 或 $[-1,1]$ 區間，因為 LSTM 內部使用 Tanh 與 Sigmoid 激活函數，輸入值落在這個範圍內時，梯度傳遞的效果最好）。

---

### 1.9 方法選擇指南

| 情境 | 建議方法 |
|---|---|
| 序列具明確趨勢／季節性、樣本數中等 | 指數平滑法（第 1 週）或 ARIMA/SARIMA |
| 需要嚴謹的統計推論（信賴區間、顯著性檢定） | ARIMA |
| 序列存在複雜非線性樣態、樣本數充足 | LSTM |
| 樣本數有限（如少於 50 期） | 優先嘗試較簡單的方法（指數平滑法、低階 ARIMA），LSTM 通常需要更多資料才能穩定訓練 |
| 需要快速的基準模型 | 指數平滑法，計算成本最低，且如第 1、10、12、13 週反覆驗證，簡單方法經常表現不俗 |

### 1.10 常用中英文詞彙對照表

| 中文 | 英文 | 中文 | 英文 |
|---|---|---|---|
| 平穩性 | Stationarity | 單根檢定 | Unit Root Test |
| 差分 | Differencing | 自我相關函數 | Autocorrelation Function, ACF |
| 偏自我相關函數 | Partial Autocorrelation Function, PACF | 訊息準則 | Information Criterion |
| 循環神經網路 | Recurrent Neural Network, RNN | 長短期記憶網路 | Long Short-Term Memory, LSTM |
| 記憶單元 | Cell State | 遺忘閘 | Forget Gate |
| 滑動窗口 | Sliding Window | 遞迴多步預測 | Recursive Multi-step Forecasting |

### 1.11 本週公式總表

| 項目 | 公式 |
|---|---|
| 一階差分 | $y_t'=y_t-y_{t-1}$ |
| ARIMA(p,d,q) | AR部分+MA部分（見1.3節完整式） |
| AIC | $-2\ln(L)+2k$ |

### 1.12 各方法的假設條件與限制一覽

| 方法 | 隱含假設 | 主要限制 |
|---|---|---|
| ARIMA | 序列（差分後）平穩；線性自我相關結構 | 無法捕捉非線性樣態；長期預測的不確定性隨預測步數快速擴大 |
| LSTM | 無分配假設，資料驅動 | 通常需要較多資料；訓練與調校複雜度高；可解釋性低 |
| 指數平滑法（第1週） | 假設結構相對簡單（水準、趨勢、季節） | 無法捕捉複雜非線性關係或長距離依賴 |

---

## Hour 2｜Python 示範演練（60 分鐘）

### 2.0 環境設置與模擬資料生成

```python
# ============================================================
# 第14週 Hour 2 示範：ARIMA與LSTM時間序列建模
# 情境：延續第1週油料消耗主題，改為10年月資料的長序列
# ============================================================
import warnings
warnings.filterwarnings('ignore')
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib

!apt-get -qq install fonts-noto-cjk > /dev/null 2>&1
matplotlib.rcParams['font.sans-serif'] = ['Noto Sans CJK JT', 'Noto Sans CJK TC']
matplotlib.rcParams['axes.unicode_minus'] = False

np.random.seed(42)
n = 120  # 10年月資料

t = np.arange(n)
trend = 500 + 2.5*t
noise = np.random.normal(0, 15, n)
# 加入AR(1)自相關結構，讓ARIMA有明確的相依關係可以捕捉
ar_component = np.zeros(n)
for i in range(1, n):
    ar_component[i] = 0.4*ar_component[i-1] + noise[i]
demand = np.round(trend + ar_component, 1)

train, test = demand[:108], demand[108:]  # 訓練108期、測試12期
print(f"訓練集: {len(train)}期，測試集: {len(test)}期")

plt.figure(figsize=(10, 4))
plt.plot(range(1, n+1), demand, color='steelblue')
plt.axvline(108, color='red', linestyle='--', label='訓練/測試分界')
plt.xlabel('月份'); plt.ylabel('油料消耗量')
plt.title('模擬油料消耗序列（10年月資料）')
plt.legend(); plt.grid(alpha=0.3)
plt.show()
```

---

### 2.1 平穩性檢定與差分

```python
# ============================================================
# 2.1 ADF平穩性檢定與差分（對照1.1節）
# ============================================================
from statsmodels.tsa.stattools import adfuller

adf_result = adfuller(train)
print(f"原始序列ADF檢定: 統計量={adf_result[0]:.4f}, p值={adf_result[1]:.4f}")
print(f"{'=> 非平穩（存在單根）' if adf_result[1]>0.05 else '=> 平穩'}")

train_diff = np.diff(train)
adf_diff = adfuller(train_diff)
print(f"\n一階差分後ADF檢定: 統計量={adf_diff[0]:.4f}, p值={adf_diff[1]:.4f}")
print(f"{'=> 仍非平穩' if adf_diff[1]>0.05 else '=> 差分後已平穩'}")
```

**預期輸出**：原始序列 $p$ 值接近 1（明顯非平穩，因存在上升趨勢），一階差分後 $p$ 值趨近於 0（差分後序列已平穩），驗證 1.1 節理論。

---

### 2.2 ACF/PACF 圖判斷階數

```python
# ============================================================
# 2.2 ACF/PACF圖（對照1.2節）
# ============================================================
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
plot_acf(train_diff, lags=20, ax=axes[0])
axes[0].set_title('差分後序列之ACF')
plot_pacf(train_diff, lags=20, ax=axes[1])
axes[1].set_title('差分後序列之PACF')
plt.tight_layout()
plt.show()
```

---

### 2.3 ARIMA 建模與模型選擇

```python
# ============================================================
# 2.3 系統性搜尋最佳ARIMA階數（對照1.3、1.4節）
# ============================================================
from statsmodels.tsa.arima.model import ARIMA

best_aic, best_order = np.inf, None
for p in range(3):
    for d in [1]:
        for q in range(3):
            try:
                fit = ARIMA(train, order=(p, d, q)).fit()
                if fit.aic < best_aic:
                    best_aic, best_order = fit.aic, (p, d, q)
            except Exception:
                continue

print(f"最佳ARIMA階數（依AIC）: {best_order}, AIC={best_aic:.2f}")

final_model = ARIMA(train, order=best_order).fit()
print(final_model.summary())

arima_forecast = final_model.forecast(steps=12)
print(f"\nARIMA預測值: {arima_forecast.round(1)}")
```

---

### 2.4 LSTM 序列預測建模

```python
# ============================================================
# 2.4 LSTM序列預測（對照1.6-1.8節）
# ============================================================
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import MinMaxScaler

tf.random.set_seed(42)

# 標準化（對照1.8節提醒）
scaler = MinMaxScaler()
train_scaled = scaler.fit_transform(train.reshape(-1, 1)).flatten()

# 建立滑動窗口：用過去12期預測下一期
window_size = 12
def create_sequences(data, window):
    X, y = [], []
    for i in range(len(data) - window):
        X.append(data[i:i+window])
        y.append(data[i+window])
    return np.array(X), np.array(y)

X_train, y_train = create_sequences(train_scaled, window_size)
X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
print(f"訓練樣本數: {X_train.shape[0]}, 每筆序列長度: {window_size}")

# 建立LSTM模型
lstm_model = keras.Sequential([
    keras.layers.Input(shape=(window_size, 1)),
    keras.layers.LSTM(32, activation='tanh'),
    keras.layers.Dense(16, activation='relu'),
    keras.layers.Dense(1),
])
lstm_model.compile(optimizer='adam', loss='mse')
lstm_model.fit(X_train, y_train, epochs=100, batch_size=8, verbose=0)

# 遞迴多步預測（對照1.8節）
last_window = train_scaled[-window_size:].tolist()
predictions_scaled = []
for _ in range(12):
    x_input = np.array(last_window[-window_size:]).reshape((1, window_size, 1))
    pred = lstm_model.predict(x_input, verbose=0)[0, 0]
    predictions_scaled.append(pred)
    last_window.append(pred)

lstm_forecast = scaler.inverse_transform(np.array(predictions_scaled).reshape(-1, 1)).flatten()
print(f"\nLSTM預測值: {lstm_forecast.round(1)}")
```

---

### 2.5 三方比較：Holt 法（第1週）vs. ARIMA vs. LSTM

```python
# ============================================================
# 2.5 三方比較：傳統平滑法 vs. ARIMA vs. LSTM
# ============================================================
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error

# --- Holt法（對照第1週，重新網格搜尋最佳alpha,beta）---
def holt_forecast(train, alpha, beta, steps):
    L, T = [train[0]], [train[1]-train[0]]
    for tt in range(1, len(train)):
        Lt = alpha*train[tt] + (1-alpha)*(L[tt-1]+T[tt-1])
        Tt = beta*(Lt-L[tt-1]) + (1-beta)*T[tt-1]
        L.append(Lt); T.append(Tt)
    return [L[-1] + (i+1)*T[-1] for i in range(steps)]

best_sse, best_params = np.inf, (0.3, 0.1)
for a in np.arange(0.1, 1.0, 0.1):
    for b in np.arange(0.1, 1.0, 0.1):
        L, T = [train[0]], [train[1]-train[0]]
        sse = 0
        for tt in range(1, len(train)):
            sse += (train[tt] - (L[tt-1]+T[tt-1]))**2
            Lt = a*train[tt] + (1-a)*(L[tt-1]+T[tt-1])
            Tt = b*(Lt-L[tt-1]) + (1-b)*T[tt-1]
            L.append(Lt); T.append(Tt)
        if sse < best_sse:
            best_sse, best_params = sse, (a, b)

holt_forecast_vals = holt_forecast(train, *best_params, 12)

# --- 彙整三方比較結果 ---
def evaluate(name, forecast):
    rmse = np.sqrt(mean_squared_error(test, forecast))
    mape = mean_absolute_percentage_error(test, forecast) * 100
    print(f"{name}: RMSE={rmse:.2f}, MAPE={mape:.2f}%")

print("=== 三方預測績效比較 ===")
evaluate(f"Holt法(α,β={best_params})", holt_forecast_vals)
evaluate(f"ARIMA{best_order}", arima_forecast)
evaluate("LSTM", lstm_forecast)
```

**預期輸出**：這是本課程反覆出現的重要教學時刻——在這組相對單純（趨勢＋輕度 AR(1) 雜訊）的模擬資料上，**Holt 法的表現通常與 ARIMA、LSTM 相當、甚至更優**（三者 MAPE 皆在 1.5–2.5% 左右）。這再次驗證第 1、10、12、13 週反覆強調的核心原則：**方法的複雜度應該與資料的真實複雜度相匹配，越先進的方法不保證在越簡單的資料上表現越好**。ARIMA 與 LSTM 的優勢，會在資料存在更複雜的非線性樣態、或需要嚴謹統計推論時才會顯現，本週 Hour 3 的論文延伸方向會進一步探討這個議題。

---

## Hour 3（前 40 分鐘）｜延伸練習與綜合案例研究

### 3.1 延伸練習一：彈藥消耗序列之 ARIMA 建模（附完整程式碼）

```python
# ============================================================
# 延伸練習一：彈藥消耗序列ARIMA建模
# ============================================================
np.random.seed(51)
n1 = 100
t1 = np.arange(n1)
level1 = 200 + 0.8*t1
ar1 = np.zeros(n1)
for i in range(1, n1):
    ar1[i] = 0.5*ar1[i-1] + np.random.normal(0, 10)
series1 = np.round(level1 + ar1, 1)
train1, test1 = series1[:88], series1[88:]

adf1 = adfuller(train1)
print(f"ADF原始序列: p={adf1[1]:.4f}")
diff1 = np.diff(train1)
adf1d = adfuller(diff1)
print(f"ADF差分後: p={adf1d[1]:.4f}")

best_aic1, best_order1 = np.inf, None
for p in range(3):
    for q in range(3):
        try:
            fit = ARIMA(train1, order=(p, 1, q)).fit()
            if fit.aic < best_aic1:
                best_aic1, best_order1 = fit.aic, (p, 1, q)
        except Exception:
            continue

model1 = ARIMA(train1, order=best_order1).fit()
fc1 = model1.forecast(steps=12)
rmse1 = np.sqrt(mean_squared_error(test1, fc1))
mape1 = mean_absolute_percentage_error(test1, fc1) * 100
print(f"最佳階數={best_order1}, RMSE={rmse1:.2f}, MAPE={mape1:.2f}%")
```

**詳解**：原始序列 ADF 檢定應顯示非平穩（$p>0.05$），差分後應顯示平穩（$p<0.05$），最終選出的 ARIMA 模型應能達到 MAPE 2% 左右的良好預測表現。

---

### 3.2 延伸練習二：含季節性通信裝備使用量之 LSTM 建模（附完整程式碼）

```python
# ============================================================
# 延伸練習二：含季節性資料之LSTM建模
# ============================================================
tf.random.set_seed(42)
np.random.seed(61)
n2 = 110
t2 = np.arange(n2)
level2 = 300 + 1.5*t2 + 20*np.sin(2*np.pi*t2/12)  # 含年週期季節性
noise2 = np.random.normal(0, 12, n2)
series2 = np.round(level2 + noise2, 1)
train2, test2 = series2[:98], series2[98:]

scaler2 = MinMaxScaler()
train2_s = scaler2.fit_transform(train2.reshape(-1, 1)).flatten()
X2, y2 = create_sequences(train2_s, window_size)
X2 = X2.reshape((X2.shape[0], window_size, 1))

lstm2 = keras.Sequential([
    keras.layers.Input(shape=(window_size, 1)),
    keras.layers.LSTM(32, activation='tanh'),
    keras.layers.Dense(16, activation='relu'),
    keras.layers.Dense(1),
])
lstm2.compile(optimizer='adam', loss='mse')
lstm2.fit(X2, y2, epochs=100, batch_size=8, verbose=0)

last_win2 = train2_s[-window_size:].tolist()
preds2_s = []
for _ in range(12):
    x_in = np.array(last_win2[-window_size:]).reshape((1, window_size, 1))
    p = lstm2.predict(x_in, verbose=0)[0, 0]
    preds2_s.append(p)
    last_win2.append(p)
pred2 = scaler2.inverse_transform(np.array(preds2_s).reshape(-1, 1)).flatten()

rmse2 = np.sqrt(mean_squared_error(test2, pred2))
mape2 = mean_absolute_percentage_error(test2, pred2) * 100
print(f"LSTM: RMSE={rmse2:.2f}, MAPE={mape2:.2f}%")
```

**詳解**：LSTM 應能一定程度上學習到季節性樣態（因滑動窗口長度設為 12，恰好涵蓋一個完整年週期），MAPE 應落在 3–5% 左右——若進一步改用第 1.5 節介紹的 SARIMA，或加大 LSTM 的窗口長度涵蓋兩個完整週期，通常能進一步改善季節性樣態的捕捉效果。

---

### 3.3 綜合案例研究：建立國防後勤補給需求預測方法選擇框架

> [!NOTE]
> 以下是一個虛構但貼近實務的案例，目的是把本課程第 1、9、14 週的時間序列方法串接起來，示範一個完整的方法選擇決策脈絡。

**背景**：某聯兵旅後勤處希望為不同性質的補給品項，建立最適合的需求預測方法選擇準則。

**步驟一：資料探索與平穩性診斷**（對應 2.0–2.1 節）
針對每個品項的歷史消耗序列，先繪圖觀察是否存在明顯趨勢或季節性，並以 ADF 檢定正式確認平穩性。

**步驟二：依資料特性分流選擇方法**（對應 1.9 節方法選擇指南）
資料量少（如少於 24 期）、結構單純的品項，優先採用第 1 週的指數平滑法；資料量中等、需要嚴謹統計推論（如需要預測區間評估安全庫存，呼應第 5 週）的品項，採用 ARIMA；資料量充足、且懷疑存在複雜非線性樣態的品項，嘗試 LSTM。

**步驟三：以樣本外測試集系統性比較**（對應 2.5 節）
不預設任何方法必然最佳，而是對每個品項實際切分訓練/測試集，比較各方法的 RMSE、MAPE，選擇實證表現最佳者，而非憑印象或方法的「先進程度」決定。

**步驟四：建立長期監控機制**
定期（如每季）重新評估各品項的最適預測方法是否仍然成立——需求型態可能隨任務性質改變而改變（呼應第 5 週動態安全庫存的精神），最適方法也可能隨之改變。

---

## Hour 3（後 20 分鐘）｜碩士論文延伸應用

> [!NOTE]
> 以下兩個方向，示範如何把本週的時間序列方法延伸為具備研究貢獻的碩士論文題目——核心邏輯是：**ARIMA 只能捕捉線性自我相關結構，LSTM 能捕捉非線性樣態但決策過程不透明；混合兩者的可行性與限制值得系統性驗證；ARIMA 的多步預測不確定性理論上會隨預測步數擴大，這對制定「該對外承諾多遠的預測」有直接的實務意涵**。每個方向皆包含主題定義、方法說明、模擬資料設計與完整可執行的 Python 實作。

### 3.4 論文方向一：混合 ARIMA-LSTM 模型之適用情境驗證研究

**主題定義**：文獻中常見的混合時間序列預測策略，是先用 ARIMA 捕捉序列的線性趨勢與自我相關結構，再用 LSTM 學習 ARIMA 殘差中殘留的非線性樣態，最終預測為兩者加總。這個策略立意良好，但**是否必然優於單獨使用 ARIMA 或 LSTM，缺乏一致的實證結論**。本研究誠實地比較純 ARIMA、純 LSTM、混合模型三者，在一組具有明確非線性殘差結構的模擬資料上的表現，並探討混合策略失靈的可能原因。

**方法說明**：

- **混合模型建構流程**：先配適 ARIMA 於原始序列，取得樣本內殘差；將殘差序列標準化後，以相同的滑動窗口手法訓練另一個 LSTM，學習殘差本身的動態樣態；最終預測 = ARIMA 預測 + 殘差 LSTM 預測。
- **重要提醒**：殘差的多步預測，本質上是對「近似白噪音」的序列進行外推，若殘差中真正的可學習樣態強度不足，LSTM 可能只學到訓練期的雜訊特徵，多步外推時反而放大誤差——這正是本研究要具體驗證的疑慮。

**模擬資料**：模擬一條含有趨勢、季節性、與明確非線性殘差樣態（殘差本身呈現與時間相關的複合三角函數波動）的 150 期序列。

```python
# ============================================================
# 論文方向一：混合ARIMA-LSTM模型之適用情境驗證
# ============================================================
import warnings
warnings.filterwarnings('ignore')
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy as np
import tensorflow as tf
from tensorflow import keras
from statsmodels.tsa.arima.model import ARIMA
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

tf.random.set_seed(42)
np.random.seed(7)
n = 150

t = np.arange(n)
trend = 500 + 2.0*t
seasonal = 30*np.sin(2*np.pi*t/12)
# 刻意設計明確的非線性殘差樣態（而非單純白噪音）
nonlinear_pattern = 15*np.sin(t/4)**2 * np.cos(t/7)
noise = np.random.normal(0, 8, n)
demand = np.round(trend + seasonal + nonlinear_pattern + noise, 1)
train, test = demand[:135], demand[135:]

def create_seq(data, w):
    X, y = [], []
    for i in range(len(data)-w):
        X.append(data[i:i+w]); y.append(data[i+w])
    return np.array(X), np.array(y)

window = 12

# --- 方法A: 純ARIMA ---
best_aic, best_order = np.inf, None
for p in range(3):
    for q in range(3):
        try:
            fit = ARIMA(train, order=(p, 1, q)).fit()
            if fit.aic < best_aic:
                best_aic, best_order = fit.aic, (p, 1, q)
        except Exception:
            continue
arima_model = ARIMA(train, order=best_order).fit()
arima_forecast = arima_model.forecast(steps=15)
rmse_arima = np.sqrt(mean_squared_error(test, arima_forecast))
print(f"純ARIMA{best_order}: RMSE={rmse_arima:.2f}")

# --- 方法B: 純LSTM ---
scaler = MinMaxScaler()
train_scaled = scaler.fit_transform(train.reshape(-1, 1)).flatten()
Xtr, ytr = create_seq(train_scaled, window)
Xtr = Xtr.reshape((Xtr.shape[0], window, 1))
lstm_pure = keras.Sequential([
    keras.layers.Input(shape=(window, 1)), keras.layers.LSTM(32, activation='tanh'),
    keras.layers.Dense(16, activation='relu'), keras.layers.Dense(1),
])
lstm_pure.compile(optimizer='adam', loss='mse')
lstm_pure.fit(Xtr, ytr, epochs=100, batch_size=8, verbose=0)
last_win = train_scaled[-window:].tolist()
lstm_preds_scaled = []
for _ in range(15):
    x_in = np.array(last_win[-window:]).reshape((1, window, 1))
    p = lstm_pure.predict(x_in, verbose=0)[0, 0]
    lstm_preds_scaled.append(p); last_win.append(p)
lstm_forecast = scaler.inverse_transform(np.array(lstm_preds_scaled).reshape(-1, 1)).flatten()
rmse_lstm = np.sqrt(mean_squared_error(test, lstm_forecast))
print(f"純LSTM: RMSE={rmse_lstm:.2f}")

# --- 方法C: 混合ARIMA+LSTM(殘差) ---
arima_residuals = train - arima_model.fittedvalues
res_scaler = MinMaxScaler()
residuals_scaled = res_scaler.fit_transform(arima_residuals.reshape(-1, 1)).flatten()
Xtr_res, ytr_res = create_seq(residuals_scaled, window)
Xtr_res = Xtr_res.reshape((Xtr_res.shape[0], window, 1))
lstm_residual = keras.Sequential([
    keras.layers.Input(shape=(window, 1)), keras.layers.LSTM(16, activation='tanh'),
    keras.layers.Dense(8, activation='relu'), keras.layers.Dense(1),
])
lstm_residual.compile(optimizer='adam', loss='mse')
lstm_residual.fit(Xtr_res, ytr_res, epochs=50, batch_size=8, verbose=0)
last_win_res = residuals_scaled[-window:].tolist()
residual_preds_scaled = []
for _ in range(15):
    x_in = np.array(last_win_res[-window:]).reshape((1, window, 1))
    p = lstm_residual.predict(x_in, verbose=0)[0, 0]
    residual_preds_scaled.append(p); last_win_res.append(p)
residual_forecast = res_scaler.inverse_transform(np.array(residual_preds_scaled).reshape(-1, 1)).flatten()
hybrid_forecast = arima_forecast + residual_forecast
rmse_hybrid = np.sqrt(mean_squared_error(test, hybrid_forecast))
print(f"混合ARIMA+LSTM(殘差): RMSE={rmse_hybrid:.2f}")

print(f"\n=== 結論 ===")
print(f"純ARIMA RMSE={rmse_arima:.2f}, 純LSTM RMSE={rmse_lstm:.2f}, 混合模型 RMSE={rmse_hybrid:.2f}")
```

**預期輸出**：這是一個誠實、未經美化的實證結果——在本模擬設計下，**純 LSTM 的表現通常優於純 ARIMA，而混合模型的表現未必優於兩者**（甚至可能與純 ARIMA 相當或略差）。原因在於：混合模型的殘差 LSTM，是對「訓練期樣本內殘差」的樣態進行學習，但多步遞迴預測時，殘差本身的外推誤差會與 ARIMA 主體預測的外推誤差**疊加**，若殘差的可學習樣態不足以抵銷這個疊加效應，混合模型反而可能得不償失。

> [!IMPORTANT]
> 這個結果對論文寫作有重要啟示：**混合模型不是「兩種方法各取優點」的免費午餐，其成敗高度取決於「殘差是否真的存在穩定、可外推的樣態」**。若殘差近似白噪音（不可預測），對殘差建模只是徒增模型複雜度與額外的誤差來源。一份嚴謹的論文，應該誠實比較混合模型與各單獨方法的表現，而非預設混合必然更好；若混合模型表現不如預期，這本身就是一個有價值的研究發現，而非需要隱藏的「失敗結果」。

**論文延伸建議**：可進一步系統性改變殘差中非線性樣態的強度（從近似白噪音到強烈非線性），觀察混合模型相對優勢隨樣態強度變化的規律，找出「混合模型值得使用」的資料特性門檻；也可以嘗試以 LSTM 直接同時輸入原始序列與 ARIMA 預測值作為額外特徵（而非事後對殘差二次建模），比較這種「並聯式」而非「序列式」的整合策略是否更穩健。

---

### 3.5 論文方向二：ARIMA 與 LSTM 多步預測地平線之誤差擴散比較研究

**主題定義**：ARIMA 的統計理論指出，隨著預測步數（Forecast Horizon）增加，預測的不確定性（預測區間寬度）會隨之擴大，這是差分後單根過程的理論性質。但 LSTM、指數平滑法是否也呈現相同的誤差擴散模式，並無同樣嚴謹的理論保證。本研究系統性比較三種方法在短期（1–3 步）與長期（10–12 步）預測地平線下的誤差表現，驗證何種方法的預測品質隨地平線增加而衰退最快，為「該對外承諾多遠的預測」提供實證依據。

**方法說明**：

- **地平線切割比較**：不看單一次「12 步預測的平均誤差」，而是把 12 步拆解為「短期（1–3 步）」與「長期（10–12 步）」兩段，分別計算平均絕對誤差，比較各方法在兩段地平線上的相對排名是否一致。
- **理論預期**：ARIMA 的預測區間理論上會隨地平線增加而擴大（尤其對含差分的 $I(1)$ 過程，預測變異數會隨步數線性甚至更快速增加）；指數平滑法的長期預測仰賴趨勢項的線性外推，長期而言也可能偏離實際；LSTM 因遞迴預測會逐步累積每一步的預測誤差，長期預測品質同樣可能劣化。

**模擬資料**：延續本週 Hour 2 之主範例資料（10 年月資料，趨勢＋AR(1)雜訊）。

```python
# ============================================================
# 論文方向二：多步預測地平線之誤差擴散比較
# ============================================================
import warnings
warnings.filterwarnings('ignore')
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy as np
import tensorflow as tf
from tensorflow import keras
from statsmodels.tsa.arima.model import ARIMA
from sklearn.preprocessing import MinMaxScaler

tf.random.set_seed(42)
np.random.seed(42)
n = 120
t = np.arange(n)
trend = 500 + 2.5*t
noise = np.random.normal(0, 15, n)
ar_component = np.zeros(n)
for i in range(1, n):
    ar_component[i] = 0.4*ar_component[i-1] + noise[i]
demand = np.round(trend + ar_component, 1)
train, test = demand[:108], demand[108:]

# --- Holt法 ---
def holt_forecast(train, alpha, beta, steps):
    L, T = [train[0]], [train[1]-train[0]]
    for tt in range(1, len(train)):
        Lt = alpha*train[tt] + (1-alpha)*(L[tt-1]+T[tt-1])
        Tt = beta*(Lt-L[tt-1]) + (1-beta)*T[tt-1]
        L.append(Lt); T.append(Tt)
    return [L[-1] + (i+1)*T[-1] for i in range(steps)]
holt_pred = holt_forecast(train, 0.2, 0.1, 12)

# --- ARIMA ---
arima_fit = ARIMA(train, order=(1, 1, 2)).fit()
arima_pred = arima_fit.forecast(steps=12)

# --- LSTM ---
scaler = MinMaxScaler()
train_scaled = scaler.fit_transform(train.reshape(-1, 1)).flatten()
window = 12
def create_seq(data, w):
    X, y = [], []
    for i in range(len(data)-w):
        X.append(data[i:i+w]); y.append(data[i+w])
    return np.array(X), np.array(y)
Xtr, ytr = create_seq(train_scaled, window)
Xtr = Xtr.reshape((Xtr.shape[0], window, 1))
lstm = keras.Sequential([
    keras.layers.Input(shape=(window, 1)), keras.layers.LSTM(32, activation='tanh'),
    keras.layers.Dense(16, activation='relu'), keras.layers.Dense(1),
])
lstm.compile(optimizer='adam', loss='mse')
lstm.fit(Xtr, ytr, epochs=100, batch_size=8, verbose=0)
last_win = train_scaled[-window:].tolist()
preds_scaled = []
for _ in range(12):
    x_in = np.array(last_win[-window:]).reshape((1, window, 1))
    p = lstm.predict(x_in, verbose=0)[0, 0]
    preds_scaled.append(p); last_win.append(p)
lstm_pred = scaler.inverse_transform(np.array(preds_scaled).reshape(-1, 1)).flatten()

# --- 逐步誤差比較：拆解短期(1-3步) vs 長期(10-12步) ---
print(f"{'步數':>6}{'Holt誤差':>10}{'ARIMA誤差':>10}{'LSTM誤差':>10}")
for h in range(12):
    e_holt = abs(test[h] - holt_pred[h])
    e_arima = abs(test[h] - arima_pred[h])
    e_lstm = abs(test[h] - lstm_pred[h])
    print(f"{h+1:>6}{e_holt:>10.2f}{e_arima:>10.2f}{e_lstm:>10.2f}")

short_holt = np.mean([abs(test[i]-holt_pred[i]) for i in range(3)])
short_arima = np.mean([abs(test[i]-arima_pred[i]) for i in range(3)])
short_lstm = np.mean([abs(test[i]-lstm_pred[i]) for i in range(3)])
long_holt = np.mean([abs(test[i]-holt_pred[i]) for i in range(9, 12)])
long_arima = np.mean([abs(test[i]-arima_pred[i]) for i in range(9, 12)])
long_lstm = np.mean([abs(test[i]-lstm_pred[i]) for i in range(9, 12)])

print(f"\n短期(1-3步)平均誤差: Holt={short_holt:.2f}, ARIMA={short_arima:.2f}, LSTM={short_lstm:.2f}")
print(f"長期(10-12步)平均誤差: Holt={long_holt:.2f}, ARIMA={long_arima:.2f}, LSTM={long_lstm:.2f}")
print(f"\n誤差擴大倍數: Holt={long_holt/short_holt:.2f}倍, ARIMA={long_arima/short_arima:.2f}倍, "
      f"LSTM={long_lstm/short_lstm:.2f}倍")
```

**預期輸出**：逐步誤差表本身會有相當的隨機波動（單一序列的單次模擬結果，不應過度解讀個別步數的誤差高低），但彙總短期與長期兩段平均誤差後，通常能觀察到 **ARIMA 的誤差擴大倍數明顯高於 Holt 法與 LSTM**——這與 ARIMA 差分後單根過程「預測變異數隨步數擴大」的理論性質一致。

> [!IMPORTANT]
> 這個發現對實務預測承諾有直接意涵：**若必須提供遠期（如年度）預測承諾，應該優先檢視 ARIMA 類方法在長地平線下的誤差擴散是否可以接受，必要時搭配預測區間（而非單點預測）呈現，讓決策者了解遠期預測的不確定性遠高於近期預測**。這也呼應第 7 週 PERT 理論「路徑轉移」與第 5 週安全庫存的核心精神——**預測與規劃不應該只提供一個「點」，而應該誠實揭露其背後隱含的不確定性隨時間如何變化**。

**論文延伸建議**：可進一步以正式的預測區間（ARIMA 可解析計算，LSTM 可用 Bootstrap 或 Monte Carlo Dropout 近似估計）取代本示範的點預測誤差比較，更嚴謹地量化三種方法在不同地平線下的不確定性；也可以將此分析框架應用於多個不同性質的實際序列（強趨勢 vs. 強季節性 vs. 高雜訊），探討「地平線誤差擴散模式」是否因序列特性而系統性不同。

---

## 附錄A：理論常見問答（概念釐清 Q&A）

> [!NOTE]
> **Q1：ARIMA 中的「d」（差分階數）可以任意選擇嗎？選太高會怎樣？**
> A：不建議任意選擇， $d$ 應由 1.1 節的 ADF 檢定客觀決定（差分到序列平穩為止，通常 $d=1$ 或 $d=2$ 已足夠）。過度差分（$d$ 選得比實際需要更高）會引入不必要的雜訊、讓序列的變異數異常放大，並使模型解讀變得更複雜，實務上很少需要 $d>2$ 。

> [!NOTE]
> **Q2：為什麼 LSTM 的訓練過程中，同一段程式碼重複執行兩次，結果可能不完全相同？**
> A：即使設定了隨機種子（如 `tf.random.set_seed(42)`），類神經網路的訓練仍可能因為底層數值運算的平行化、硬體差異等因素產生些微不同的結果（這與第 5 週蒙地卡羅模擬「單次結果不穩定，須多次重複」的提醒相呼應）。若需要嚴謹比較不同方法或架構，建議如本週論文方向的示範，用多組不同的隨機種子重複實驗，比較平均表現而非單次結果。

> [!NOTE]
> **Q3：ARIMA 與 LSTM，哪一個比較容易在生產環境中部署維運？**
> A：ARIMA 通常較容易維運——模型參數少、重新訓練速度快、`statsmodels` 的輸出包含完整的統計診斷資訊，方便監控模型是否仍然適配當前資料。LSTM 的訓練時間較長、超參數較多，重新訓練的成本較高，但若資料量持續累積、且業務對精確度的要求極高，仍值得投入。這是典型的「模型效能」與「維運成本」的權衡取捨，與第 12 週 XGBoost vs. 隨機森林的討論精神一致。

> [!NOTE]
> **Q4：可以用 LSTM 同時預測多個未來時間點（而非逐步遞迴預測）嗎？**
> A：可以，這稱為「序列到序列（Sequence-to-Sequence）」或「多輸出（Multi-output）」預測架構——讓輸出層直接產生多個時間點的預測值，而非只預測下一期。這種做法能避免遞迴預測的誤差累積問題，但模型設計更複雜，超出本課程範圍，有興趣的學員可自行查閱進階資料。

> [!NOTE]
> **Q5：本週的滑動窗口大小（`window_size=12`）該如何決定？**
> A：若已知資料具有季節性，窗口長度通常應涵蓋至少一個完整季節週期（如月資料設為 12），讓模型有機會學到季節樣態；若無明顯季節性，窗口長度可視為一個超參數，透過交叉驗證比較不同窗口長度下的驗證集表現決定，過短可能無法捕捉足夠的歷史資訊，過長則會減少可用的訓練樣本數量（因為每個窗口都要佔用相應長度的資料）。

---

## 附錄B：本週方法快速索引表

| 任務 | 主要函數／類別 | 所屬套件 |
|---|---|---|
| ADF平穩性檢定 | `adfuller` | `statsmodels.tsa.stattools` |
| ACF/PACF繪圖 | `plot_acf`、`plot_pacf` | `statsmodels.graphics.tsaplots` |
| ARIMA建模 | `ARIMA` | `statsmodels.tsa.arima.model` |
| 季節性ARIMA | `SARIMAX` | `statsmodels.tsa.statespace.sarimax` |
| LSTM層 | `LSTM` | `tensorflow.keras.layers` |
| Min-Max標準化 | `MinMaxScaler` | `sklearn.preprocessing` |

---

## 附錄C：常見易混淆概念澄清

| 容易混淆的概念 | 差異說明 |
|---|---|
| 「AR模型」 vs 「MA模型」 | AR用過去的「數值」預測現在；MA用過去的「預測誤差」預測現在，兩者捕捉的相依結構型態不同 |
| 「ACF」 vs 「PACF」 | ACF是原始相關（可能包含透過中間期間接傳遞的相關）；PACF是扣除中間期影響後的「淨」相關 |
| 「AIC」 vs 「調整後R²」（第10週） | 兩者都是在配適度與模型複雜度之間權衡的指標，AIC用於比較不同機率模型（如不同階數的ARIMA），調整後R²專用於線性迴歸模型比較，數學形式不同但精神一致 |
| 「RNN的隱藏狀態」 vs 「LSTM的記憶單元」 | RNN只有單一隱藏狀態，資訊隨每個時間步驟直接覆蓋，容易遺忘久遠資訊；LSTM額外維護一個記憶單元，透過閘門機制選擇性更新，能保留更久遠的資訊 |
| 「差分」（本週） vs 「殘差」（第10週迴歸診斷） | 差分是「同一序列相鄰兩期的差」，用於消除趨勢使序列平穩；殘差是「實際值與模型預測值的差」，用於評估模型配適優劣，兩者計算方式與用途皆不同 |

---

## 附錄D：本週與後續課程週次的關聯

| 後續週次 | 關聯方式 |
|---|---|
| 第 1 週：需求預測 | 本週正式回應第1週埋下的伏筆，以嚴謹統計方法（ARIMA）與深度學習方法（LSTM）重新檢視同一類預測問題 |
| 第 5 週：存貨管制 | ARIMA提供理論完備的預測區間，可直接作為第5週安全庫存計算中前置期需求標準差的更精確估計來源 |
| 第 7 週：CPM/PERT | 本週3.5節「多步預測誤差擴散」的概念，與第7週「路徑轉移風險」皆提醒：點預測或單一路徑估計，可能低估長期不確定性 |
| 第 15、16 週：論文整併實戰 | 本週建立的時間序列預測框架，將作為後續整合型專論中處理動態預測需求的核心模組 |

## 附錄E：本週模擬資料集與程式碼彙整

| 資料集 | 用途 | 摘要 |
|---|---|---|
| 油料消耗序列（120期，延續第1週） | Hour2主範例 | Holt/ARIMA/LSTM三方MAPE皆約1.5-2.5%，表現相近 |
| 彈藥消耗序列（100期） | 延伸練習一 | ARIMA最佳階數搜尋，MAPE約2% |
| 通信裝備使用量（110期，含季節性） | 延伸練習二 | LSTM窗口涵蓋完整年週期，MAPE約3-5% |
| 複合趨勢季節非線性序列（150期） | 論文方向一 | 混合ARIMA-LSTM未必優於純LSTM，誠實呈現 |
| 油料消耗序列（120期，地平線分析） | 論文方向二 | ARIMA長期誤差擴大倍數明顯高於Holt、LSTM |

---

## 參考資料

- ARIMA 模型與 ACF/PACF 判斷準則之標準理論框架，詳細技術文件可參考 [statsmodels 官方文件：時間序列分析](https://www.statsmodels.org/stable/tsa.html)。
- LSTM 原始論文：Hochreiter, S., & Schmidhuber, J. (1997). *Long Short-Term Memory*. Neural Computation, 9(8), 1735-1780.

---

*下週課程：第 15 週｜論文整併實戰（一）－預測性維護整合研究。*
