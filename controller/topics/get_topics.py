import json

def get_topics_from_keywords(sentence):
    # JSON dosyasını oku
    with open("controller/topics/topics.json", "r", encoding="utf-8") as file:
        topics = json.load(file)

    # Cümleyi küçük harfe çevir
    sentence = sentence.lower()
    matching_topics = []

    # Konuları ve anahtar kelimeleri kontrol et
    for topic, keywords in topics.items():
        for keyword in keywords:
            if keyword in sentence and topic not in matching_topics:
                    matching_topics.append(topic)
    
    # Eğer eşleşen bir konu yoksa
    if not matching_topics:
        return ["Belirlenmemiş"]
    
    # Birden fazla konu varsa virgülle ayır
    return ", ".join(matching_topics)