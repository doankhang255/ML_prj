# Hướng Dẫn Thuyết Trình Project: Market vs Sentiment SVR cho VNINDEX

## 1. Mục tiêu project

Mục tiêu của project là kiểm tra xem dữ liệu **thị trường** và dữ liệu **sentiment từ tin tức** có hỗ trợ dự đoán return của VNINDEX hay không.

Project được chia thành hai hướng dự đoán:

```text
1. Daily prediction
   Dự đoán VNINDEX return theo ngày.

2. Weekly prediction
   Dự đoán VNINDEX return theo tuần.
```

Trong mỗi hướng, model được chạy với ba nhóm feature:

```text
market-only     chỉ dùng dữ liệu thị trường
sentiment-only  chỉ dùng dữ liệu sentiment/news
combined        kết hợp market + sentiment
```

Câu hỏi nghiên cứu chính:

```text
Sentiment từ tin tức có giúp cải thiện khả năng dự đoán VNINDEX return so với chỉ dùng dữ liệu thị trường hay không?
```

## 2. Tổng quan pipeline

Flow tổng thể của project:

```text
Thu thập dữ liệu VNINDEX
        |
        v
Xử lý dữ liệu tin tức
        |
        v
Tạo sentiment dictionary và gán nhãn tin tức
        |
        v
Tổng hợp sentiment index theo ngày/tuần
        |
        v
Merge sentiment với VNINDEX
        |
        v
Tạo feature market và sentiment
        |
        v
Train SVR theo daily/weekly
        |
        v
Tuning hyperparameters
        |
        v
So sánh kết quả với baseline
```

## 3. Dữ liệu đầu vào

### 3.1. Dữ liệu thị trường

Dữ liệu thị trường của VNINDEX được lấy bằng thư viện `vnstock`.

File crawl:

```text
Crawl_Data/crawl_vnstock.py
```

Dữ liệu gốc gồm các thông tin:

```text
date / time
open
high
low
close
volume
```

Trong project, các cột này được chuẩn hóa thành:

```text
open_price
high_price
low_price
close_price
vol_total
daily_return / weekly_return
log_vol_total
```

Ý nghĩa:

```text
open_price       giá mở cửa
high_price       giá cao nhất
low_price        giá thấp nhất
close_price      giá đóng cửa
vol_total        khối lượng giao dịch
log_vol_total    log của khối lượng giao dịch
```

### 3.2. Dữ liệu tin tức

Dữ liệu tin tức được xử lý để tạo ra sentiment index. Các file dữ liệu quan trọng:

```text
Dataset/equity_news_clean_content.parquet
Dataset/equity_news_tokenized_vncorenlp.parquet
Dataset/candidate_ngram_terms.parquet
Dataset/candidate_ngram_terms_dictionary.parquet
Dataset/equity_news_content_sentiment_ratios.parquet
Dataset/market_sentiment_index_daily.parquet
```

Ý tưởng xử lý:

```text
Tin tức thô
-> làm sạch nội dung
-> tokenize tiếng Việt
-> trích xuất n-gram terms
-> gán nhãn positive/negative/neutral
-> tính sentiment score cho từng bài viết
-> tổng hợp thành sentiment index theo ngày/tuần
```

## 4. Xử lý sentiment từ tin tức

### 4.1. Tạo sentiment dictionary

File:

```text
Preprocess_for_model/build_sentiment_dictionary.py
```

File này dùng model PhoBERT sentiment để gán nhãn cho các candidate terms.

Output:

```text
Dataset/candidate_ngram_terms_dictionary.parquet
```

Mỗi term được gán một trong ba nhãn:

```text
positive
negative
neutral
```

Cách trình bày:

```text
Sau khi trích xuất các cụm từ thường xuất hiện trong tin tức,
em dùng mô hình PhoBERT sentiment để phân loại các cụm từ này
thành tích cực, tiêu cực hoặc trung lập. Kết quả là một bộ từ điển sentiment
dùng để đánh giá nội dung tin tức.
```

### 4.2. Gán sentiment cho nội dung tin tức

File:

```text
Preprocess_for_model/build_sentiment_label_content.py
```

Logic chính:

```text
Đếm số token/term positive
Đếm số token/term negative
Đếm số token/term neutral
Tính sentiment_score
Gán sentiment_label cho bài viết
```

Công thức sentiment score:

```text
sentiment_score = (positive_count - negative_count) / (positive_count + negative_count)
```

Nếu bài viết có quá ít từ mang sắc thái cảm xúc, bài viết được gán là neutral.

## 5. Merge sentiment với VNINDEX

### 5.1. Daily merge

File:

```text
Preprocess_for_model/merge_vnindex_daily_with_sentiment.py
```

Output:

```text
Dataset/vnindex_daily_sentiment_merged.csv
```

Điểm quan trọng:

```text
Tin tức ngày T được map sang phiên giao dịch tiếp theo.
```

Lý do:

```text
Tránh look-ahead bias.
Model không được dùng thông tin tương lai để dự đoán hiện tại.
```

Daily target hiện tại:

```text
future_ret_1d
future_ret_5d
future_ret_10d
future_ret_20d
```

Ý nghĩa:

```text
future_ret_1d   return của 1 ngày giao dịch tiếp theo
future_ret_5d   return của 5 ngày giao dịch tiếp theo
future_ret_10d  return của 10 ngày giao dịch tiếp theo
future_ret_20d  return của 20 ngày giao dịch tiếp theo
```

### 5.2. Weekly merge

File:

```text
Preprocess_for_model/merge_vnindex_weekly_with_sentiment.py
```

Output:

```text
Dataset/vnindex_weekly_sentiment_merged.parquet
```

Weekly target hiện tại:

```text
future_ret_1w
future_ret_4w
```

Ý nghĩa:

```text
future_ret_1w  return của tuần tiếp theo
future_ret_4w  return của 4 tuần tiếp theo
```

## 6. Feature engineering

Feature đã được tách riêng khỏi pipeline train model để dễ đọc và dễ sửa.

Daily feature file:

```text
Model/SVR/Daily/daily_feature_utils.py
```

Weekly feature file:

```text
Model/SVR/Weekly/weekly_feature_utils.py
```

Trong project, feature được chia thành hai nhóm chính:

```text
1. Market features
2. Sentiment features
```

## 7. Nhóm market features

Market features được tạo từ dữ liệu giá và thanh khoản.

### 7.1. Daily market features

```text
daily_return
return_lag_1d
return_lag_5d
return_lag_20d
return_ma_5d
return_ma_20d
return_vol_5d
return_vol_20d
log_vol_total
volume_shock_20d
high_low_range_pct
range_shock_20d
return_shock_z_20d
large_down_day
```

Ý nghĩa:

```text
return_lag       return trong quá khứ
return_ma        xu hướng trung bình gần đây
return_vol       độ biến động
log_vol_total    thanh khoản đã log-transform
volume_shock     bất thường về thanh khoản
range_shock      bất thường về biên độ giao dịch
large_down_day   đánh dấu ngày giảm mạnh
```

### 7.2. Weekly market features

```text
weekly_return
return_lag_1w
return_lag_4w
return_ma_4w
return_ma_12w
return_vol_4w
return_vol_12w
log_vol_total
volume_shock_12w
range_shock_12w
return_shock_z_12w
large_down_week
negative_sentiment_market_stress
negative_sentiment_volume_shock
```

Ghi chú:

```text
Một số weekly feature là interaction giữa sentiment và market stress.
Ví dụ: negative_sentiment_market_stress.
```

## 8. Nhóm sentiment features

Sentiment features được tạo từ sentiment index và thông tin số lượng bài viết.

### 8.1. Daily sentiment features

```text
sentiment_index_z
log_article_count
positive_ratio
negative_ratio
sentiment_lag_1d
sentiment_lag_2d
sentiment_lag_5d
sentiment_ma_5d
sentiment_ma_20d
sentiment_z_shock_1d
sentiment_shock_vs_ma5
extreme_negative_sentiment
negative_ratio_change_1d
news_attention_shock
```

### 8.2. Weekly sentiment features

```text
sentiment_index_z
log_article_count
positive_ratio
negative_ratio
sentiment_lag_1w
sentiment_lag_4w
sentiment_ma_8w
sentiment_z_shock_1w
sentiment_shock_vs_ma4
extreme_negative_sentiment
negative_attention
negative_ratio_change_1w
news_attention_shock
```

Ý nghĩa:

```text
sentiment_index_z        sentiment đã chuẩn hóa
positive_ratio           tỷ lệ bài viết tích cực
negative_ratio           tỷ lệ bài viết tiêu cực
sentiment_lag            sentiment trong quá khứ
sentiment_ma             trung bình động sentiment
sentiment_z_shock        thay đổi sentiment bất thường
news_attention_shock     số lượng tin tăng bất thường
extreme_negative         đánh dấu giai đoạn sentiment rất xấu
```

## 9. Giảm cộng tuyến

Để tránh đưa các biến quá trùng lặp vào model, project có file kiểm tra correlation:

```text
Model/SVR/feature_correlation_report.py
```

Lệnh chạy:

```powershell
python Model\SVR\feature_correlation_report.py --frequency daily --feature-set combined --method pearson --threshold 0.85

python Model\SVR\feature_correlation_report.py --frequency weekly --feature-set combined --method pearson --threshold 0.85
```

Một số feature đã bị loại khỏi model:

Daily:

```text
sentiment_balance
close_open_return
negative_attention
log_val_total
value_shock_20d
```

Weekly:

```text
sentiment_balance
sentiment_lag_2w
sentiment_ma_4w
close_open_return
future_ret_4w_compound
```

Cách nói khi thuyết trình:

```text
Sau khi tạo feature, em kiểm tra tương quan giữa các biến.
Những biến có thông tin trùng lặp cao được loại bỏ để giảm rủi ro cộng tuyến
và hạn chế overfitting.
```

## 10. Model SVR

Daily pipeline:

```text
Model/SVR/Daily/svr_daily_sentiment_pipeline.py
```

Weekly pipeline:

```text
Model/SVR/Weekly/svr_weekly_sentiment_pipeline.py
```

SVR được train theo flow:

```text
Load data
-> chọn feature set
-> split train/validation/test theo thời gian
-> scale feature và target
-> tạo lookback window
-> train SVR RBF
-> predict
-> inverse scale
-> evaluate
-> compare với baseline
```

Vì đây là dữ liệu time series, dữ liệu không được shuffle.

## 11. Lookback

Lookback là số ngày hoặc số tuần quá khứ được đưa vào model.

Ví dụ:

```text
lookback = 4 trong weekly
=> model nhìn 4 tuần gần nhất để dự đoán return tương lai.
```

Lưu ý:

```text
Nếu lookback = 1 nhưng feature có return_ma_12w,
model vẫn nhận được thông tin quá khứ 12 tuần thông qua rolling feature.
```

## 12. Tuning model

File chạy nhiều experiment:

```text
Model/SVR/Daily/run_daily_sentiment_experiments.py
Model/SVR/Weekly/run_weekly_sentiment_experiments.py
```

Tuning sẽ thử nhiều tổ hợp tham số:

```text
C
gamma
epsilon
look_back
scaler
objective
target
feature_set
```

Ý nghĩa tham số:

```text
C        mức phạt sai số, C lớn model có thể phức tạp hơn
gamma    mức ảnh hưởng của từng điểm dữ liệu trong RBF kernel
epsilon  vùng sai số cho phép của SVR
scaler   standard hoặc robust
objective metric dùng để chọn model tốt nhất trên validation
```

Objective:

```text
mae     ưu tiên giảm sai số trung bình
rmse    ưu tiên giảm lỗi lớn
corr    ưu tiên tương quan cao giữa actual và predicted
diracc  ưu tiên dự đoán đúng chiều tăng/giảm
```

## 13. Lệnh chạy mẫu

### 13.1. Chạy daily market-only

```powershell
python Model\SVR\Daily\svr_daily_sentiment_pipeline.py --target future_ret_5d --feature-set market --tune --objective mae
```

### 13.2. Chạy daily sentiment-only

```powershell
python Model\SVR\Daily\svr_daily_sentiment_pipeline.py --target future_ret_5d --feature-set sentiment --tune --objective mae
```

### 13.3. Chạy daily combined

```powershell
python Model\SVR\Daily\svr_daily_sentiment_pipeline.py --target future_ret_5d --feature-set combined --tune --objective mae
```

### 13.4. Chạy weekly market-only

```powershell
python Model\SVR\Weekly\svr_weekly_sentiment_pipeline.py --target future_ret_1w --feature-set market --tune --objective mae
```

### 13.5. Chạy weekly sentiment-only

```powershell
python Model\SVR\Weekly\svr_weekly_sentiment_pipeline.py --target future_ret_1w --feature-set sentiment --tune --objective mae
```

### 13.6. Chạy weekly combined

```powershell
python Model\SVR\Weekly\svr_weekly_sentiment_pipeline.py --target future_ret_1w --feature-set combined --tune --objective mae
```

### 13.7. Chạy tất cả experiment daily

```powershell
python Model\SVR\Daily\run_daily_sentiment_experiments.py
```

### 13.8. Chạy tất cả experiment weekly

```powershell
python Model\SVR\Weekly\run_weekly_sentiment_experiments.py
```

## 14. Metric đánh giá

Bảng metric gồm:

```text
MAE
RMSE
DirAcc
Corr
TrueMean
PredMean
TrueStd
PredStd
```

Ý nghĩa:

```text
MAE       sai số tuyệt đối trung bình, càng thấp càng tốt
RMSE      phạt lỗi lớn mạnh hơn MAE, càng thấp càng tốt
DirAcc    tỷ lệ dự đoán đúng chiều tăng/giảm
Corr      tương quan giữa return thực tế và return dự đoán
TrueMean  return trung bình thực tế
PredMean  return trung bình model dự đoán
TrueStd   độ biến động thực tế
PredStd   độ biến động dự đoán
```

Cách đọc kết quả:

```text
Model tốt không chỉ có MAE thấp.
Model cần tốt hơn baseline, có Corr dương, DirAcc ổn định,
và PredStd không quá nhỏ so với TrueStd.
```

Nếu `PredStd` quá nhỏ:

```text
Model đang dự đoán gần trung bình.
Đây có thể là dấu hiệu underfit hoặc objective MAE làm model quá an toàn.
```

## 15. Baseline comparison

Baseline là các cách dự đoán đơn giản để làm mốc so sánh.

Baseline trong project:

```text
Zero return
Train mean
Previous return
Moving average return
```

Ý nghĩa:

```text
Zero return       luôn dự đoán return = 0
Train mean        luôn dự đoán bằng return trung bình của train
Previous return   dùng return gần nhất làm dự đoán
Moving average    dùng trung bình động return
```

Câu nói quan trọng:

```text
Nếu model không tốt hơn baseline đơn giản,
thì model chưa chứng minh được giá trị dự đoán.
```

## 16. Cách so sánh kết quả

Khi trình bày kết quả, nên so sánh theo 4 trục:

```text
1. Daily vs Weekly
2. Market-only vs Sentiment-only vs Combined
3. Target ngắn hạn vs target dài hơn
4. Objective mae vs corr vs diracc
```

Bảng so sánh nên có dạng:

```text
Frequency | Target | Feature set | Objective | Val MAE | Test MAE | Val Corr | Test Corr | Test improvement vs baseline
Daily     | 5D     | market      | mae       | ...     | ...      | ...      | ...       | ...
Daily     | 5D     | sentiment   | mae       | ...     | ...      | ...      | ...       | ...
Daily     | 5D     | combined    | mae       | ...     | ...      | ...      | ...       | ...
Weekly    | 1W     | market      | mae       | ...     | ...      | ...      | ...       | ...
Weekly    | 1W     | sentiment   | mae       | ...     | ...      | ...      | ...       | ...
Weekly    | 1W     | combined    | mae       | ...     | ...      | ...      | ...       | ...
```

Cách kết luận:

```text
Nếu combined tốt hơn market-only:
    sentiment có đóng góp.

Nếu combined gần bằng market-only:
    sentiment chưa tạo thêm nhiều thông tin.

Nếu sentiment-only yếu:
    sentiment không đủ mạnh để dự đoán độc lập.

Nếu model thua train mean:
    model chưa có giá trị dự đoán thật sự.
```

## 17. Điểm cần nhấn mạnh trong thuyết trình

Bạn nên nhấn mạnh 4 điểm:

```text
1. Project không chỉ train model, mà xây dựng pipeline đầy đủ từ data đến evaluation.
2. Sentiment được xử lý từ nội dung tin tức, không phải nhập thủ công.
3. Model được so sánh với baseline để tránh kết luận ảo.
4. Kết quả được kiểm tra qua validation/test theo đúng thứ tự thời gian.
```

## 18. Hạn chế

Những hạn chế nên trình bày:

```text
1. Return tài chính rất nhiễu.
2. Tin tức không phải yếu tố duy nhất ảnh hưởng VNINDEX.
3. Thị trường có regime shift: Covid, bull market, bear market.
4. SVR không phải model sequence mạnh như LSTM/Transformer.
5. Sentiment dictionary có thể còn nhiễu.
6. Một số target ngắn hạn có thể quá khó để dự đoán.
```

## 19. Hướng phát triển

Hướng mở rộng:

```text
1. Thử classification: dự đoán tăng/giảm thay vì return cụ thể.
2. So sánh SVR với Random Forest, XGBoost, LSTM.
3. Thêm dữ liệu macro: lãi suất, tỷ giá, VN30, nước ngoài mua/bán.
4. Tách regime thị trường: uptrend, downtrend, sideways.
5. Backtest chiến lược giao dịch nếu model ổn định.
```

## 20. Outline slide đề xuất

```text
Slide 1  - Tên đề tài
Slide 2  - Mục tiêu và câu hỏi nghiên cứu
Slide 3  - Dữ liệu sử dụng
Slide 4  - Xử lý tin tức và tạo sentiment index
Slide 5  - Merge sentiment với VNINDEX daily/weekly
Slide 6  - Feature engineering: market vs sentiment
Slide 7  - Giảm cộng tuyến bằng correlation
Slide 8  - Mô hình SVR và train/validation/test split
Slide 9  - Tuning hyperparameters
Slide 10 - Metric và baseline comparison
Slide 11 - Kết quả so sánh daily/weekly, market/sentiment/combined
Slide 12 - Kết luận, hạn chế, hướng phát triển
```

## 21. Đoạn mở đầu gợi ý

```text
Trong project này, em tập trung vào bài toán dự đoán return của VNINDEX
bằng cách kết hợp dữ liệu thị trường và dữ liệu sentiment từ tin tức.
Em chia bài toán thành hai tần suất dữ liệu là daily và weekly,
sau đó so sánh ba nhóm feature: market-only, sentiment-only và combined.
Mục tiêu là kiểm tra xem sentiment có thực sự bổ sung thông tin cho model
so với việc chỉ dùng dữ liệu giá và volume hay không.
```

## 22. Đoạn kết luận gợi ý

```text
Kết quả của project cho thấy dự đoán return thị trường là một bài toán khó,
đặc biệt với horizon ngắn. Vì vậy, việc so sánh với baseline là rất quan trọng.
Project đã xây dựng được một pipeline đầy đủ từ thu thập dữ liệu,
xử lý tin tức, tạo sentiment index, merge với VNINDEX, train SVR,
tuning tham số và đánh giá kết quả trên validation/test.
Hướng tiếp theo là mở rộng sang các model khác và kiểm tra sentiment
trong từng regime thị trường.
```
