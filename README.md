# ML_prj

Repo nay giu nguyen toan bo file cu. Pipeline moi de train/backtest duoc tao rieng trong:

- [training_RF_demo_core_.py](/C:/Users/doank/OneDrive/Documents/dev/ML_prj/Trainning_model/training_RF_demo_core_.py)

Tai lieu thiet ke cu cua ban van o day:

- [trading_bot_system_design.md](/C:/Users/doank/OneDrive/Documents/dev/ML_prj/Other/trading_bot_system_design.md)

## Pipeline moi dang lam gi

Script moi se:

1. Doc du lieu OHLCV 1D tu MongoDB collection `raw_ohlcv_daily`
2. Generate feature theo tung ticker rieng, khong tron rolling window giua cac ma
3. Tao label theo logic:
   - Signal o ngay `t`
   - Vao lenh o `open` ngay `t+1`
   - Theo doi trong `H` phien tiep theo
   - Co `target return`, `stop loss`, `fee`
   - Neu het `H` phien ma chua cham target/stop thi thoat o `close` phien cuoi
4. Gop tat ca ticker thanh mot bang event duy nhat de train
5. Chia train/test theo thoi gian, khong chia random
6. Train `RandomForestClassifier`
7. Backtest tren tap test va xuat ket qua ra `outputs/rf_demo_core`

## Feature set hien tai

Script dang dung dung bo feature ban vua chot:

- `ma10_ma50_ratio`
- `close_ma10_ratio`
- `dist_ma10_ma50_pct`
- `ma10_slope_3d`
- `ma50_slope_3d`
- `golden_cross`
- `return_3d`
- `return_5d`
- `rsi`
- `macd_hist`
- `volume_ratio_20`
- `volume_change_pct`
- `atr_pct`
- `rolling_std_10`
- `body_ratio`
- `upper_wick_ratio`
- `lower_wick_ratio`
- `close_position`

Luu y:

- Neu train theo mode `crossover_only`, script chi giu cac dong `golden_cross == 1`
- Khi do `golden_cross` thanh hang so, script se tu dong bo feature nay ra khoi model
- Neu muon model scan moi cay nen thi dung mode `all_rows`

## Cach chia du lieu khi data tach theo ticker

Day la cach chia phu hop nhat cho bai toan cua ban:

1. Tinh feature theo tung ticker rieng
2. Sau khi tinh xong moi gop lai thanh mot bang lon
3. Khong split random
4. Split theo `signal_date` toan cuc: 80% ngay dau train, 20% ngay sau test
5. Cho phep cung ticker xuat hien o train va test, nhung o cac moc thoi gian khac nhau

Ly do:

- Ban dang muon trade chinh tap ticker nay trong tuong lai
- Vi vay split theo thoi gian la mo phong live trading dung nhat
- Khong nen rolling feature tren du lieu da gop nhieu ticker

Neu sau nay ban muon kiem tra kha nang tong quat hoa sang ticker chua tung thay, hay lam them mot split phu theo ticker hold-out. Nhung do khong nen la split chinh o giai doan nay.

## Candidate mode nen dung

- `crossover_only`
  - Chi tao mau train tai diem `golden_cross`
  - Phu hop nhat voi triet ly "MA crossover + ML filter"
- `all_rows`
  - Tao mau train tren moi dong co du du lieu
  - Model hoc rong hon nhung it "thuần crossover" hon

Goi y:

- Bat dau voi `crossover_only`
- Khi baseline on roi moi thu `all_rows`

## Logic label va backtest

Script moi dang dung logic:

- `signal_date`: ngay nhin thay setup
- `entry_date`: ngay ke tiep, vao o `open`
- `target_return`: muc loi nhuan net mong muon
- `fee_pct`: tong phi/slippage uoc luong cho 1 round-trip
- `stop_loss`: muc cat lo gross
- `horizon`: so phien toi da giu lenh
- `buy_label = 1` neu trade sau khi tru phi van loi nhuan `> 0`

Backtest hien tai:

- Chi lay cac lenh test co `pred_proba >= prob_threshold`
- Moi ticker chi giu 1 vi the tai 1 thoi diem
- Co the mo nhieu ticker cung luc
- Tong PnL tinh theo `capital_per_trade` co dinh cho moi lenh

## Gioi han quan trong cua data 1D

Voi du lieu 1D, neu cung 1 cay nen vua cham target vua cham stop thi khong biet cai nao xay ra truoc.

Trong script moi:

- Mac dinh dung `ambiguity_mode=conservative`
- Nghia la neu cung 1 ngay vua cham target vua cham stop thi coi nhu stop xay ra truoc

Day la cach an toan hon khi backtest daily data. Neu sau nay ban muon ung dung that, nen them du lieu intraday (`1h`, `15m`, `5m`) cho phan execution/backtest.

## Mongo schema script mong doi

Collection `raw_ohlcv_daily` can toi thieu:

- `ticker`
- `trading_date`
- `timeframe`
- `open`
- `high`
- `low`
- `close`
- `volume`

Mac dinh script:

- Database: `stock_ml`
- Collection: `raw_ohlcv_daily`
- Timeframe: `1D`

## Cach chay

### Cach 1: dat env truoc

```powershell
$env:MONGO_URI="your_mongo_uri"
python "Trainning_model\training_RF_demo_core_.py"
```

### Cach 2: truyen tham so truc tiep

```powershell
python "Trainning_model\training_RF_demo_core_.py" `
  --mongo-uri "your_mongo_uri" `
  --mongo-db "stock_ml" `
  --mongo-collection "raw_ohlcv_daily" `
  --candidate-mode "crossover_only" `
  --horizon 10 `
  --target-return 0.03 `
  --stop-loss -0.02 `
  --fee-pct 0.003 `
  --prob-threshold 0.55 `
  --capital-per-trade 10000000
```

### Chay mot nhom ticker nho de debug nhanh

```powershell
python "Trainning_model\training_RF_demo_core_.py" `
  --tickers "VCB,FPT,HPG,MBB,TCB" `
  --candidate-mode "crossover_only"
```

## Output sinh ra

Sau khi chay xong, script se tao:

- `outputs/rf_demo_core/train_dataset.csv`
- `outputs/rf_demo_core/test_predictions.csv`
- `outputs/rf_demo_core/selected_trades.csv`
- `outputs/rf_demo_core/feature_importance.csv`
- `outputs/rf_demo_core/summary.json`

Neu them `--export-excel` va may da cai `openpyxl`, script se tao them:

- `outputs/rf_demo_core/rf_demo_core_report.xlsx`

## Goi y thu nghiem tiep theo

Nen bat dau voi:

1. `candidate_mode=crossover_only`
2. `horizon=10`
3. `target_return=0.03`
4. `stop_loss=-0.02`
5. `prob_threshold=0.55`

Sau do so sanh:

- `prob_threshold = 0.50 / 0.55 / 0.60`
- `horizon = 7 / 10 / 15`
- `target_return = 0.02 / 0.03 / 0.04`
- `crossover_only` voi `all_rows`

## De tien toi live trading

Sau baseline nay, pipeline nen bo sung:

- Walk-forward validation thay vi 1 lan split 80/20
- Transaction cost thuc te hon
- Slippage
- Position sizing
- Risk cap theo tong danh muc
- Sector exposure control
- Intraday execution check
- Logging lenh va order state

File moi nay du de ban co mot baseline chay duoc, backtest duoc, va thay model co dang loc duoc crossover tot/xau hay khong.
