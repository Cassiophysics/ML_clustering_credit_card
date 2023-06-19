# Importação das Bibliotecas
import streamlit as st
import pandas as pd
import joblib

# Carregar o modelo
modelo_lgbm = joblib.load('modelo_lgbm.sav')

# Criar a interface do Streamlit
st.title('🎯 Classificador de Clientes')
st.header('Insira os Dados')

# Dados numéricos
BALANCE = st.slider('SALDO', 0, 19043)
BALANCE_FREQUENCY = st.slider('FREQUÊNCIA DE ATUALIZAÇÃO DO SALDO', 0.0, 1.0)
COMPRAS = st.slider('COMPRAS', 0, 49039)
ONEOFF_COMPRAS = st.slider('VALOR DE COMPRAS DE UMA SÓ VEZ', 0, 40761)
INSTALLMENTS_COMPRAS = st.slider('VALOR DE COMPRAS PARCELADAS', 0, 22500)
CASH_ADVANCE = st.slider('DINHEIRO ADIANTADO', 0, 47137)
COMPRAS_FREQUENCY = st.slider('FREQUÊNCIA DE COMPRAS', 0.0, 1.0)
ONEOFF_COMPRAS_FREQUENCY = st.slider('FREQUÊNCIA DE COMPRAS DE UMA SÓ VEZ', 0.0, 1.0)
COMPRAS_INSTALLMENTS_FREQUENCY = st.slider('FREQUÊNCIA DE COMPRAS PARCELADAS', 0.0, 1.0)
CASH_ADVANCE_FREQUENCY = st.slider('FREQUÊNCIA DE DINHEIRO ADIANTADO', 0.0, 1.5)
CASH_ADVANCE_TRX = st.slider('TRANSAÇÕES COM DINHEIRO ADIANTADO', 0, 123)
COMPRAS_TRX = st.slider('TRANSAÇÕES DE COMPRAS', 0, 358)
CREDIT_LIMIT = st.slider('LIMITE DE CRÉDITO', 50, 30000)
PAYMENTS = st.slider('PAGAMENTOS', 1901, 50721)
MINIMUM_PAYMENTS = st.slider('QUANTIDADE MÍNIMA DE PAGAMENTOS', 0.01, 76406.2)
PRC_FULL_PAYMENT = st.slider('PORCENTAGEM DO PAGAMENTO INTEGRAL', 0.0, 1.0)
TENURE	 = st.slider('DETENÇÃO DO CARTÃO DE CRÉDITO', 6, 12)

# Realizar a transformação dos dados de entrada
X = pd.DataFrame({
    'BALANCE': [BALANCE],
    'BALANCE_FREQUENCY': [BALANCE_FREQUENCY],
    'COMPRAS': [COMPRAS],
    'ONEOFF_COMPRAS': [ONEOFF_COMPRAS],
    'INSTALLMENTS_COMPRAS': [INSTALLMENTS_COMPRAS],
    'CASH_ADVANCE': [CASH_ADVANCE],
    'COMPRAS_FREQUENCY': [COMPRAS_FREQUENCY],
    'ONEOFF_COMPRAS_FREQUENCY': [ONEOFF_COMPRAS_FREQUENCY],
    'COMPRAS_INSTALLMENTS_FREQUENCY': [COMPRAS_INSTALLMENTS_FREQUENCY],
    'CASH_ADVANCE_FREQUENCY': [CASH_ADVANCE_FREQUENCY],
    'CASH_ADVANCE_TRX': [CASH_ADVANCE_TRX],
    'COMPRAS_TRX': [COMPRAS_TRX],
    'CREDIT_LIMIT': [CREDIT_LIMIT],
    'PAYMENTS': [PAYMENTS],
    'MINIMUM_PAYMENTS': [MINIMUM_PAYMENTS],
    'PRC_FULL_PAYMENT': [PRC_FULL_PAYMENT],
    'TENURE': [TENURE]

})

# Fazer a previsão usando o modelo carregado
if st.button('Fazer Previsão'):
    resultado = modelo_lgbm.predict(X)
    st.header('Resultado da Previsão')
    if resultado == 0:
        st.write('Cliente Premium')
    elif resultado == 1:
        st.write('Cliente de Baixo Limite')
    elif resultado == 2:
        st.write('Cliente com Poucas Compras')
    else:
        st.write('Categoria desconhecida')