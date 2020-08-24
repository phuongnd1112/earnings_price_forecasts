# Stock Prices Analysis By Quarter in Comparison to Financial Record/Ratios 

This repository contains scripts that serve to analyse maximum potential gain of stocks by quarter, then regressed against financial records/ratios such as net operating income, net profit, ROE, ROA (calculated as %YoY Growth).

Kho này chứa những script (viết bằng python) phân thích lãi tối đã cổ phiếu theo quý, tính quy hồi tuyến tính với các dữ liệu/chỉ số tài chính như doanh thu, LNST, ROE, ROA (tính bởi tăng trưởng cùng kỳ năm trước). 

##Beforing Running / Trước khi bấm chạy 
* pip3 install pandas numpy seaborn matplotlib sklearn 
* pip install pandas numpy seaborn matplotlib sklearn
* API tools required for tcbs specific script / API cần được install nếu chạy script của tcbs  
 
##Documentation of Functions and Parameters / Function và Parameter 
1. ```analyse_single_feature(str: ticker)```: regress feature by feature onto quarterly price changes / quy hồi tuyến tính từng dữ liệu/chỉ số trên lãi 
2. ```analyse_aggregate_feature(str: ticker)```: regress all features onto quarterly price changes / quy hồi tuyến tính đa biến 

##Notes for Amendments and Version Updates
1. ```function_tcbs_specific``` : data pulled from tcbs database, API needed 
2. ```function_general```: analyse csv with historical data pullable from investing.com


