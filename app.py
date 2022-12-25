from flask import Flask, json, request, render_template, redirect, url_for, flash, Response
from linearScript import *
from adaScript import *
from gradientScript import *
import pandas as pd

app =  Flask(__name__)


@app.route('/' ,methods=['GET','POST'])
def index():
   
    market= message= model = indicator=''
    
    if request.method == 'POST':
        market = request.form.get('stockname')
        indicator = request.form['indicator']
        model = request.form.get('model')
      
            # linear regressor
        
        if (indicator=="RSI" and model =='linear'):
            RSI_Value = str (EMAline(market))
            message = RSI_Value    
     
        elif(indicator=="EMA" and model =='linear'):
            EMA_Value = str (EMAline(market))
            message=EMA_Value

        elif(indicator=="MACD" and model =='linear'):
            MACD_Value = str (MACDline(market))
            message= MACD_Value
        
        elif(indicator=="OBV" and model =='linear'):
            OBV_Value = str (OBVline(market))
            message= OBV_Value

        elif(indicator=="MFI" and model =='linear'):
            MFI_Value = str (MFIline(market))
            message= MFI_Value

            # ada boost 

        elif (indicator=="RSI" and model =='ada'):
            RSI_Value = str (RSIada(market))
            message = RSI_Value    
     
        elif(indicator=="EMA" and model =='ada'):
            EMA_Value = str (EMAada(market))
            message=EMA_Value

        elif(indicator=="MACD" and model =='ada'):
            MACD_Value = str (MACDada(market))
            message= MACD_Value
        
        elif(indicator=="OBV" and model =='ada'):
            OBV_Value = str (OBVada(market))
            message= OBV_Value

        elif(indicator=="MFI" and model =='ada'):
            MFI_Value = str (MFIada(market))
            message= MFI_Value

            # gradient boost

        elif (indicator=="RSI" and model =='grad'):
            RSI_Value = str (RSIgrad(market))
            message = RSI_Value    
     
        elif(indicator=="EMA" and model =='grad'):
            EMA_Value = str (EMAgrad(market))
            message=EMA_Value

        elif(indicator=="MACD" and model =='grad'):
            MACD_Value = str (MACDgrad(market))
            message= MACD_Value
        
        elif(indicator=="OBV" and model =='grad'):
            OBV_Value = str (OBVgrad(market))
            message= OBV_Value

        elif(indicator=="MFI" and model =='grad'):
            MFI_Value = str (MFIgrad(market))
            message= MFI_Value

        elif(indicator =="None"):
            Primary_Value  = str (primary(market))
            message = Primary_Value
    

    return render_template("index.html", message = message )
 

if __name__ == "__main__" :
    app.run(debug= True)

