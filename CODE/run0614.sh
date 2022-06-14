###
 # @Descripttion: 
 # @version: 
 # @Author: Shenqiang Ke
 # @Date: 2022-06-14 19:09:35
 # @LastEditors: 
 # @LastEditTime: 2022-06-14 19:10:01
### 

for model_name in 'arima' 'prophet' 'holt';do
{
    python baseline_test.py --model_name $model_name &
}
done
