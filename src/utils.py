import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet as wn
from nltk import pos_tag
from collections import defaultdict

# Descargar recursos necesarios de nltk
# import nltk
# nltk.download('punkt')
# nltk.download('stopwords')
# nltk.download('wordnet')
# nltk.download('averaged_perceptron_tagger')

def preprocess_text(df: pd.DataFrame, text_column, language='english'):
    """
    Preprocesa texto en un DataFrame siguiendo los pasos: eliminar líneas en blanco, 
    convertir a minúsculas, tokenizar, remover stopwords, caracteres no alfabéticos y realizar lematización.
    
    Args:
        dataframe (pd.DataFrame): El DataFrame que contiene los datos.
        text_column (str): Nombre de la columna de texto a procesar.
        
    Returns:
        pd.DataFrame: DataFrame con una nueva columna 'text_final' con el texto procesado.
    """

    # Step - a (1): Eliminar filas con valores NaN en la columna de texto
    df[text_column].dropna(inplace=True)
    
    # Step - b (2): Convertir a minúsculas
    df[text_column] = df[text_column].str.lower()
    
    # Step - c (3): Tokenizar el texto
    df[text_column] = df[text_column].apply(word_tokenize)
    
    # Step - d (4): Remover stopwords, caracteres no alfabéticos y lematizar
    # Mapeo de etiquetas POS para WordNetLemmatizer
    tag_map = defaultdict(lambda: wn.NOUN)
    tag_map['V'] = wn.VERB
    # tag_map['J'] = wn.ADJ
    # tag_map['R'] = wn.ADV

    # Instancia del lematizador
    lemmatizer = WordNetLemmatizer()

    def process_tokens(tokens):
        """Procesa tokens individuales: remueve stopwords, caracteres no alfabéticos y lematiza."""
        final_words = []
        for word, tag in pos_tag(tokens):
            if word not in stopwords.words(language) and word.isalpha():
                # Lematización basada en la etiqueta POS
                lemma = lemmatizer.lemmatize(word, tag_map[tag[0]])
                final_words.append(lemma)
        return final_words

    # Aplicar el procesamiento a cada fila
    df['text_final'] = df[text_column].apply(process_tokens)
    
    # Convertir la lista de palabras en una cadena si es necesario
    df['text_final'] = df['text_final'].apply(lambda x: ' '.join(x))
    
    return df
