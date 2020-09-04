# Quarterly Stock Prices Analysis in Comparison to Fundamentals Data  

## Vietnamese Version 
Kho này chứa những script phân thích lãi tối đa cổ phiếu theo quý, tính quy hồi tuyến tính với các dữ liệu/chỉ số tài chính như doanh thu, LNST, ROE, ROA (tính bởi tăng trưởng cùng kỳ năm trước). 

### Trước khi bấm chạy 
* Đọc flowchart 
* ```pip3 install pandas numpy seaborn matplotlib sklearn``` 
* ```pip install pandas numpy seaborn matplotlib sklearn```
* Link API cần được install trước khi chạy file TCBS

### Ngành 
Ngân hàng dùng ```optimising_features_bank.py``` <br> 
Ngành khác dùng ```optimising_features_non_bank.py```

### Chạy 
Vì object và các function đã được viết trong script, người dùng bấm chạy luôn trên terminal/editor. <br> 
Khi chạy trong terminal, người dùng sẽ được input những chỉ tiêu nhất định theo mong muốn. 
1. Save figures and data to -> path nơi bảng csv và chart sẽ được lưu vào 
2. Ticker
3. Simple or Aggregate Model (simple/aggregate)? -> chọn giữa model đơn biến hoặc đa biến 
4. Which list would you list to run? (ratios/income/balance) -> chọn tổ hợp fundamentals  
5. Mean Option -> chọn tiêu chí "trung bình" Ri hoặc Rq 

### Chi Tiết Code (Chạy cùng nhau)
1. ```correlation(dataset=df, threshold=int)```: loại bỏ biến có co-linearity 
2. ```clean_data()```: dọn data kéo từ database tcbs 
3. ```generate_quarter_data()```: vì data giá cổ phiểu theo ngày, data phải được process thành data theo quý. Ở đây, ai biến "trung bình" giá được tính. 
    * Ri - max/min trong quý 
    * Rq - % return trung bình quý này so với trung bình quý trước 
4. ```import_financials()```: kéo data fundmentals từ trên DB 
5. ```single_feature()```: tính quy hồi tuyến tính đơn biến cho tất cả dữ liệu trong tổ hợp
6. ```multi_features()```: tính quy hồi tuyến tính tất cả các combo có thể trong tổ hợp fundamentals 

### Intepreting & Storing Results
Bảng CSV - sort theo giá trị R2 cao nhất - sẽ trả ra các cột kết quả sau: 
1. Ticker 
2. Features -> các dữ liệu fundamentals được dùng để tính hồi quy
3. Features_count -> bao nhiêu dữ liệu fundamentals đã được dùng (dùng cho việc tìm số để tối ưu hoá lựa chọn số liệu) 
4. R2 -> độ tương quan giữa nhóm features và Ri/Rq 
5. Coefs -> a trong Ri/Rq = a1*x + a2*x + ... + an*x + b 
6. Intercept -> b trong Ri/Rq = a1*x + a2*x + ... + an*x + b 

