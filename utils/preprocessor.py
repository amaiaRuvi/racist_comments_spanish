import re
import emoji


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


# Eliminar otros caracteres inusuales
def delete_specific_characters(comment):
    specific_chars_pattern = r'[«»✈ºø♀♂€🇮±°ª‘’“”🇱]'
    return re.sub(specific_chars_pattern, '', comment)


# Eliminar emoticonos
def delete_emoticons(comment):
    return emoji.replace_emoji(comment, replace='')


# Eliminar caracteres árabes
def delete_arabic_letters(text):
    arabic_pattern = r'[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF]'
    return re.sub(arabic_pattern, '', text)


# Unificar las distintas formas de expresar la risa
def unify_laughs (comment):
    laugh_pattern = r"\b(a*ha+h[ha]*|o?l+o+l+[ol]*|x+d+[x*d*]*|a*ja+[j+a+]+|j+e+j+[ej]*|j+s+j+[sj]*|j+i+j+[ij]*)\b"
    return re.sub(laugh_pattern, 'jaja', comment, flags=re.IGNORECASE)


def delete_extra (comment):
    return re.sub(r'\bmikeldi\b', '', comment, flags=re.IGNORECASE)


# Función para preprocesar el texto
def preprocess_comment(comment):
    comment = delete_spaces(comment)
    comment = lower_text(comment)
    comment = delete_urls(comment)
    comment = delete_repeated_consonants(comment)
    comment = delete_repeated_vowels(comment)
    comment = delete_accents(comment)
    comment = delete_characters(comment)
    comment = delete_specific_characters(comment)
    comment = delete_emoticons(comment)
    comment = delete_arabic_letters(comment)
    comment = unify_laughs(comment)
    comment = delete_extra(comment)
    return comment
