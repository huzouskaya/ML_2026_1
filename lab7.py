import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

nltk.download('stopwords', quiet=True)

raw_dialogue = """<span class=participant_1>я музыкант<br />я люблю вкусно поесть<br />я живу за городом<br />я занимаюсь плаванием<br />я люблю путешествовать<br /></span>	<span class=participant_2>Я переводчик.<br />Люблю вкусно поесть.<br />Я хочу научиться плавать.<br />Я боюсь темноты.<br />Я люблю танцы.<br /></span>	<span class=participant_1>Пользователь 1: Я очень люблю музыку</span><br /><span class=participant_2>Пользователь 2: привет, как дела?</span><br /><span class=participant_1>Пользователь 1: А что любите вы?</span><br /><span class=participant_1>Пользователь 1: Дела отлично!</span><br /><span class=participant_1>Пользователь 1: Как у вас?</span><br /><span class=participant_2>Пользователь 2: я люблю вкусно поесть 😂 а вы?</span><br /><span class=participant_1>Пользователь 1: Это мое любимое занятие))) ха</span><br /><span class=participant_1>Пользователь 1: Но перед едой я конечно же сначала поплаваю)</span><br /><span class=participant_2>Пользователь 2: а я из спорта люблю  спортивные танцы)</span><br /><span class=participant_1>Пользователь 1: Да согласен- танцы мне знакомы, я иногда танцую, когда<br />путешествую )))</span><br /><span class=participant_1>Пользователь 1: А где вы живете?</span><br /><span class=participant_2>Пользователь 2: я кстати очень хочу научиться плавать .... поможете<br />? 😉</span><br /><span class=participant_1>Пользователь 1: конечно же))</span><br /><span class=participant_2>Пользователь 2: а живу я в городе. а вы?</span><br /><span class=participant_1>Пользователь 1: Приезжайте ко мне-я живу за городом))</span><br /><span class=participant_1>Пользователь 1: когда вас ожидать?</span><br /><span class=participant_2>Пользователь 2: может уже сегодня ?) только я боюсь темноты, поэтому<br />лучше днем</span><br /><span class=participant_1>Пользователь 1: ок</span><br /><span class=participant_1>Пользователь 1: Ладно до завтра,мне нужно бежать))</span><br /><span class=participant_2>Пользователь 2: пока!</span><br />"""

# ============= ИЗВЛЕЧЕНИЕ ДИАЛОГА =============
pattern = r'<span class=participant_(\d+)>(.*?)</span>'
matches = re.findall(pattern, raw_dialogue, re.DOTALL)

dialogue = []
for participant_id, text in matches:
    # Очищаем текст от HTML-тегов
    text = re.sub(r'<[^>]+>', ' ', text)
    # Убираем "Пользователь X:" 
    text = re.sub(r'Пользователь\s*\d+:\s*', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    if text:
        dialogue.append((participant_id, text))

# ============= ОБЪЕДИНЯЕМ РЕПЛИКИ ОДНОГО УЧАСТНИКА =============
merged = {}
for speaker, text in dialogue:
    if speaker not in merged:
        merged[speaker] = []
    merged[speaker].append(text)

dialogue_merged = []
for speaker, texts in merged.items():
    dialogue_merged.append((speaker, " ".join(texts)))

dialogue = dialogue_merged

# ============= ОБЪЕДИНЯЕМ ТЕКСТ =============
full_text = " ".join([text for _, text in dialogue])

# ============= ОЧИСТКА ТЕКСТА =============
text_lower = full_text.lower()
text_no_punct = re.sub(r'[^а-яА-Яa-zA-Z\s]', ' ', text_lower)
text_no_punct = re.sub(r'\s+', ' ', text_no_punct).strip()
tokens = text_no_punct.split()

# ============= УДАЛЕНИЕ СТОП-СЛОВ =============
russian_stopwords = set(stopwords.words('russian'))
tokens_no_stop = [word for word in tokens if word not in russian_stopwords and len(word) > 2]
if len(tokens_no_stop) < 3:
    tokens_no_stop = tokens

# ============= СТЕММИНГ =============
stemmer = SnowballStemmer("russian")
stemmed_tokens = [stemmer.stem(word) for word in tokens_no_stop]
stemmed_text = " ".join(stemmed_tokens)

if not stemmed_text.strip():
    stemmed_text = " ".join(tokens[:10])

# ============= ВЕКТОРИЗАЦИЯ =============
vectorizer_bow = CountVectorizer()
bow_matrix = vectorizer_bow.fit_transform([stemmed_text])
bow_features = vectorizer_bow.get_feature_names_out()

vectorizer_tfidf = TfidfVectorizer()
tfidf_matrix = vectorizer_tfidf.fit_transform([stemmed_text])
tfidf_features = vectorizer_tfidf.get_feature_names_out()
tfidf_values = tfidf_matrix.toarray()[0]

vectorizer_ngram = CountVectorizer(ngram_range=(2, 3))
ngram_matrix = vectorizer_ngram.fit_transform([stemmed_text])
ngram_features = vectorizer_ngram.get_feature_names_out()

for speaker, text in dialogue:
    print(f"  Участник {speaker}: {text[:150]}...")

print(f"\nВсего участников: {len(dialogue)}")

print("\nTF-IDF: ")
word_tfidf_pairs = list(zip(tfidf_features, tfidf_values))
word_tfidf_pairs.sort(key=lambda x: x[1], reverse=True)
for i, (word, score) in enumerate(word_tfidf_pairs[:10], 1):
    print(f"  {i}. {word} - {score:.4f}")

print("\nBag-of-Words: ")
bow_pairs = list(zip(bow_features, bow_matrix.toarray()[0]))
bow_pairs.sort(key=lambda x: x[1], reverse=True)
for i, (word, count) in enumerate(bow_pairs[:10], 1):
    print(f"  {i}. {word} - {count}")