from datasets import load_dataset
import re


# Eliminar saltos de línea y espacios repetidos
def delete_spaces(comment):
    spaces_pattern = r'[\n\r]+|\s+'
    return re.sub(spaces_pattern, ' ', comment)


# Convertir todo a minúsculas
def lower_text(comment):
    return comment.lower()


# Eliminar URL
def delete_urls(comment):
    url_pattern = r'http[s]?://\S+'
    return re.sub(url_pattern, '', comment)


# Eliminar consonantes repetidas y puntos suspensivos
def delete_repeated_consonants(comment):
    repeated_consonant_pattern = r'([^aeiou\s\r\n0-9])\1{1,}'
    def replace(match):
        char = match.group(1)
        if char in 'rcnl':
            return char * 2
        else:
            return char

    return re.sub(repeated_consonant_pattern, replace, comment, flags=re.IGNORECASE)


# Dejar como máximo 2 vocales iguales contiguas
def delete_repeated_vowels(comment):
    repeated_vowels_pattern = r'([aeiouAEIOU])\1{2,}'
    return re.sub(repeated_vowels_pattern, r'\1\1', comment, flags=re.IGNORECASE)


# Eliminar acentos no empleados en Español
def delete_accents (comment):
    comment = re.sub(r"[àâãäå]", "a", comment)
    comment = re.sub(r"ç", "c", comment)
    comment = re.sub(r"[èêë]", "e", comment)
    comment = re.sub(r"[ìîï]", "i", comment)
    comment = re.sub(r"[òôõö]", "o", comment)
    comment = re.sub(r"[ùû]", "u", comment)
    comment = re.sub(r"[ýÿ]", "y", comment)
    return comment


# Eliminar caracteres inusuales
def delete_characters(comment):
    special_characters = r'[ºª|·~¬\^`[\]¨´#\\\'\(\)*\<>_]'
    return re.sub(special_characters, '', comment)


# Eliminar emoticonos
def delete_emoticons(comment):
    emoticon_pattern = r'[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF\U0001F700-\U0001F77F\U0001F900-\U0001F9FF]'
    return re.sub(emoticon_pattern, '', comment)


# Unificar las distintas formas de expresar la risa
def unify_laughs (comment):
    laugh_pattern = r"\b(a*ha+h[ha]*|o?l+o+l+[ol]*|x+d+[x*d*]*|a*ja+[j+a+]+|j+e+j+[ej]*)\b"
    return re.sub(laugh_pattern, 'jaja', comment, flags=re.IGNORECASE)


# Función para preprocesar el texto
def preprocess_comment(comment):
    comment = delete_spaces(comment)
    comment = lower_text(comment)
    comment = delete_urls(comment)
    comment = delete_repeated_consonants(comment)
    comment = delete_repeated_vowels(comment)
    comment = delete_accents(comment)
    comment = delete_characters(comment)
    comment = delete_emoticons(comment)
    comment = unify_laughs(comment)
    return comment
