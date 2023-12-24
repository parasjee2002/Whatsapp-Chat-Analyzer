import re
import pandas as pd
import spacy

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

def preprocess_nlp(data):
    pattern = r'\d{1,2}/\d{1,2}/\d{2,4},\s\d{1,2}:\d{2}\s-\s'

    messages = re.split(pattern, data)[1:]
    dates = re.findall(pattern, data)

    df = pd.DataFrame({'user_message': messages, 'message_date': dates})
    df['message_date'] = pd.to_datetime(df['message_date'], format='%d/%m/%Y, %H:%M - ')

    # Apply spaCy for tokenization and part-of-speech tagging
    tokenized_messages = []
    pos_tags = []

    for doc in nlp.pipe(df['user_message'], disable=["ner", "parser"]):
        tokens = [token.text for token in doc]
        tokenized_messages.append(tokens)

        tags = [token.pos_ for token in doc]
        pos_tags.append(tags)

    df['tokenized_message'] = tokenized_messages
    df['pos_tags'] = pos_tags

    # Extract named entities using spaCy
    named_entities = []

    for doc in nlp.pipe(df['user_message'], disable=["parser", "tagger"]):
        entities = [ent.text for ent in doc.ents]
        named_entities.append(entities)

    df['named_entities'] = named_entities

    return df

# Example usage
data = """12/31/2022, 09:45 - Alice: Hello! How are you?
12/31/2022, 10:00 - Bob: I'm good, thanks. What about you?
"""

result_df = preprocess_nlp(data)
print(result_df)
