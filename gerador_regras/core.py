import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder
import pickle
import os
import ssl
import time

ssl._create_default_https_context = ssl._create_unverified_context

playlist_csv = os.environ['DATA']
def salvar_regras(regras, nome_arquivo):
    with open(nome_arquivo, 'wb') as handle:
        pickle.dump(regras, handle, protocol=pickle.HIGHEST_PROTOCOL)

def preprocessar_dados(df):
    df = df.drop(columns=['track_uri', 'album_name', 'album_uri', 'artist_name', 'artist_uri', 'duration_ms'])
    return df.groupby('pid')['track_name'].apply(list).tolist()

def treinar_e_salvar_modelo(dados, min_sup=0.07, min_conf=0.5, nome_arquivo='data/rules.pickle'):
    while True:
        te = TransactionEncoder()
        te_ary = te.fit(dados).transform(dados)
        df_encoded = pd.DataFrame(te_ary, columns=te.columns_)
        
        conjuntos_frequentes = apriori(df_encoded, min_support=min_sup, use_colnames=True)
        regras = association_rules(conjuntos_frequentes, metric="confidence", min_threshold=min_conf)
        salvar_regras(regras, nome_arquivo)
        time.sleep(304)

if __name__ == "__main__":
    while True:
        dados_spotify = pd.read_csv(playlist_csv)
        dados_preprocessados = preprocessar_dados(dados_spotify)
        treinar_e_salvar_modelo(dados_preprocessados)