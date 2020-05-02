from mindsdb import Predictor


afc_predictor = Predictor(name='afc')

retrain = True

if retrain:
    afc_predictor.learn(to_predict='banking_crisis', from_data='https://docs.google.com/spreadsheets/d/e/2PACX-1vSA35cAZFvQgJhw2ShJwumY-VRjPkcHI9hgmn4OSisU6AZLfO3hjsGo53Ijf8fvaidcviOJGdhsrwCX/pub?output=csv')




prediction = afc_predictor.predict(when = {
    'case': 1,
    'cc3': 'DZA',
    'country':	'Algeria',
    'year':	2019,
    'systemic_crisis': 0,
    'exch_usd': 87.9706983,
    'domestic_debt_in_default': 0,
    'sovereign_external_debt_default': 0,
    'gdp_weighted_default': 0,
    'inflation_annual_cpi': 2.917,
    'independence': 1,
    'currency_crises': 1,
    'inflation_crises': 0
})

print(prediction[0]['banking_crisis'])