
'''
Ce script prend un tableau de données du corpus X 
et récupère les données qui nous interesse 

Passage d'une considération par token déclencheurs en une vision par SitEmo
'''

import spacy 
import pandas as pd

nlp = spacy.load("fr_core_news_sm")


def lecture_tabl():
    '''Lecture du fichier tsv du corpus d'origine  
    '''
    file = "./files/infos_par_token_de_declencheur_avec_lemme.tsv"
    tableau = pd.read_csv(file, sep='\t')
    return tableau

def on_veut_que_montrer(table):
    '''
    Récuperer uniquement les émotions montrées
    '''
    #list_modes = table["Mode"].drop_duplicates().to_list()
    montrer = table.query("`Mode` == 'Montree'") 
    #print(list_modes)
    montrer.to_csv("./files/Emo_montree_enToken.tsv", index=False, sep='\t')
    return montrer

def on_veut_les_SitEmo(montrer_tableau):
    '''
    Création d'un nouveau tableau par SitEmo
    '''
    #enlever les colonnes qui concernent les tokens déclencheurs
    table = montrer_tableau.drop(columns=['Lemme', 'Token_declencheur'])

    # supprimer les lignes qui ont le même Id de SitEmo 
    table2 = table.drop_duplicates(subset=['Id_SitEmo'])  
    
    # Rajouter la lemmatisation des déclencheurs et du SitEmo 
    table2['Lemmes_declencheurs'] = table2['Declencheur_entier'].apply(lemmatisation)
    table2['Lemmes_SitEmo'] = table2['Texte_SitEmo'].apply(lemmatisation)
    
    #Création du nouveau tableau 
    table2.to_csv("./files/Emo_Montree_SitEmo_lemma.tsv", index=False, sep='\t') 
    
    return table2
    
def lemmatisation(declencheurs): 
    '''
    fonction de lemmatisation 
    '''
    doc = nlp(declencheurs)
    lemma = [token.lemma_ for token in doc]
    lemmes = ' '.join(lemma)
    print(lemma)
    return lemmes


def main(): 
    table = lecture_tabl()
    montrer_tableau = on_veut_que_montrer(table)
    tableau2= on_veut_les_SitEmo(montrer_tableau)
    print(tableau2)
    #print(montrer_tableau)
    #print(len(table))



if __name__ == "__main__":
    main()