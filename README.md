# Stock Prices Analysis By Quarter in Comparison to Financial Record/Ratios 

## English Version 
This repository contains scripts that serve to analyse maximum potential gain of stocks by quarter, then regressed against financial records/ratios such as net operating income, net profit, ROE, ROA (calculated as %YoY Growth). <br> 
*Note*: Non-TCBS script has limited functionality due to user having to pull data from the internet (version beta -- constantly improving). 

### Beforing Running / Trước khi bấm chạy 
* ```pip3 install pandas numpy seaborn matplotlib sklearn``` 
* ```pip install pandas numpy seaborn matplotlib sklearn```
* For TCBC-specific script, API is required

### Documentation of Functions and Parameters / Function và Parameter 
1. ```analyse_single_feature(str: ticker)``` found in ```function_tcbs_specific.py```: regression model between single variable fundamental datas & ratios + maximum price change potentials
2. ```analyse_aggregate_feature(str: ticker)``` found in ```function_tcbs_specific_agg.py```: bivarite regression models between single variable fundamental datas & ratios + maximum price change potentials

### Intepreting & Storing Results
When applicable, the script will ask where you want to store your files (this include a table of the highest R-squared + regression models for all tickers/data). Please specific this with the path in which you wish to store your file. 

## Vietnamese Version 
Kho này chứa những python script phân thích lãi tối đa cổ phiếu theo quý, tính quy hồi tuyến tính với các dữ liệu/chỉ số tài chính như doanh thu, LNST, ROE, ROA (tính bởi tăng trưởng cùng kỳ năm trước). 

* API tools required for tcbs specific script / API cần được install nếu chạy script của tcbs  

### Trước khi bấm chạy 
* ```pip3 install pandas numpy seaborn matplotlib sklearn``` 
* ```pip install pandas numpy seaborn matplotlib sklearn```
* Link API cần được install trước khi chạy file TCBS 

### Function và Parameter 
1. ```analyse_single_feature(str: ticker)``` trong ```function_tcbs_specific.py```: quy hồi tuyến tính giữa các dữ liệu/chỉ số tài chính fundamental + lãi tối đa cổ phiếu theo quý
2. ```analyse_aggregate_feature(str: ticker)``` found in ```function_tcbs_specific_agg.py```: quy hồi tuyến tính đa biến giữa tất cả các dữ liệu/chỉ số tài chính fundamental + lãi tối đa cổ phiếu theo quý

### Intepreting & Storing Results
Khi được áp dụng, file sẽ hỏi bạn nơi mà bạn muốn lưu file output (bảng có các chỉ số r-squared cao nhất + các chart QHTT). Copy path nơi bạn muốn save file và pass vào terminal.
