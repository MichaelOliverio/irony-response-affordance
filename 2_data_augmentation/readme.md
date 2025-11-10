# Data Augmentation

Da single-turn a multi-turn

## Prompt
### Prompt 1

>Given this pair of messages 
>
>u1: {user1_utterance} 
>
>u2: {user2_utterance} 
>
>generate the 3 previous dialogue turns (where a turn is composed by a message from u1 and u2), consistent and coherent with the conversation, that lead up to the given u1 and u2

### Prompt 2
>Given this pair of messages extracted from social media:
>
>u1: {user1_utterance}
>
>u2: {user2_utterance}
>
>Generate the 3 previous dialogue turns (each turn consisting of a message from u1 and a message from u2) that logically lead up to the given messages. Make sure to:
>- Maintain the original writing style of both u1 and u2.
>- Preserve the sentiment and tone of the conversation.
>- Keep the dialogue coherent and consistent with the context of the given messages

### Note

- come euristica generare dei dialoghi a partire da single-turn soltanto se il numero di token è sopra una certa soglia (mediana lunghezza token per ogni lingua)
- mettere nel prompt il fatto di mantere lo stesso stile di U1 e U2 e di **non alterare** il sentiment

## Modelli per S2M
- [meta-llama/Llama-3.1-8B-Instruct](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct)
- ...

## Valutazione

- [background metriche](https://consensus.app/search/how-to-evaluate-the-generation-of-data-augmentatio/LXpYcghUR-C3SGO1EvNqZg/)
- [perplexity](https://huggingface.co/spaces/evaluate-metric/perplexity): per valutare la coerenza (da approfondire)
- [Human Equivalence Score (HEQ)](https://arxiv.org/pdf/2109.12264): da approfondire
- [LLM judge](https://huggingface.co/learn/cookbook/llm_judge)
    - criteri:
        - stile u1/u2 (utilizzo maiscuole, minuscole, punteggiatura, emoji..)
        - fluenct/naturalezzza
            - a coppie (con focus sull'ultima)
            - globale
        - common sense
        - varietà (minimizzare la ridondanza/ripetizione, cioè il fatto che non vengano ripetute cose nello scambio)
        - preservare tono/sentiment (da scartare? non oggettivo)
    - usare un prompt unico oppure 1 prompt per ogni criterio?