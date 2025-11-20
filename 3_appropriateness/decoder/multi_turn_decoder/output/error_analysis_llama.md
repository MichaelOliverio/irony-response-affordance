# Error Analysis — Llama-3.1-8B-Instruct

## 1. Reasoning errors

### 1.1 Incorrect reasoning: irony classification out of context

**Observed phenomenon:**
The model tends to classify the *u2* message before the last *u1* message as ironic/non-ironic, leading to incorrect reasoning.

**Example:**

* **id:** `17091512765802921252_8735726914411658796`
* **Generated dialog:**

  ```json
   {
      "u1":"I just found this old movie online. Do you remember it?",
      "u2":"Oh yeah, I remember that one! It was so cheesy."
   },
   {
      "u1":"Holy shit it’s been so long since I’ve watched this. I used to love it when I was like 10",
      "u2":"I love it too! It reminds me of granny babysitting on Saturday nights!"
   }
  ```
* **Model reasoning:**

  ```
   1. First, we need to identify the last message sent by u1. In this case, it is \"Holy shit it’s been so long since I’ve watched this. I used to love it when I was like 10\".
   2. Then, we need to check if u2's response is ironic. In this case, u2's response is \"Oh yeah, I remember that one! It was so cheesy.\".
   3. To determine if the response is ironic, we need to check if the sentiment of u2's response is opposite to the sentiment of u1's last message. In this case, u1's last message is a positive statement about the movie, and u2's response is a negative statement about the movie.
   4. Since the sentiment of u2's response is opposite to the sentiment of u1's last message, we can conclude that u2's response is ironic.
   5. However, we also need to consider the context of the conversation. In this case, u2's response is a joke, and u2 is being sarcastic about the movie being cheesy. This is a common way to respond to a nostalgic message, and it is not necessarily inappropriate.
   6. Therefore, we can conclude that u2's response is appropriate.
  ```
* **Actual label:** `0`
* **Predicted label:** `1`

---

### 1.2 Incorrect reasoning: the model goes off-topic

**Observed phenomenon:**
The model does not evaluate whether *u2*’s response is ironic or not, but whether it is “appropriate” in context, completely shifting the task.

**Example:**

* **id:**
  `11160340551004577019_3353532200050318167`
  `12760707994161305770_10779634747600532941`
* **Generated dialog:**

  ```json
  {
    "u1": "She gets really upset when I try to talk to her about it. I feel like I'm not doing enough.",
    "u2": "I understand. It's hard to know how to approach these situations. Have you tried talking to her in a more supportive way?"
  },
  {
    "u1": "Tried to get my mom to stop drinking when I was a teen, she got really upset. She still drinks alot.",
    "u2": "I’m sorry to hear, I’m in the same boat"
  }
  ```
* **Model reasoning:**

  ```
  The first message from u1 expresses frustration and a sense of inadequacy when trying to discuss a sensitive issue with someone. The second message from u1 reveals that they have a history of trying to intervene in someone's drinking problem, which is likely related to the first message. u2's response is supportive and non-confrontational, which is appropriate given the context. u2 is not responding ironically to u1's message, but rather offering a helpful suggestion. Therefore, the output should be 1 (appropriate).
  ```
* **Actual label:** `0`
* **Predicted label:** `1`

---

## 2. Tone shift from single-turn to multi-turn

**Observed phenomenon:**
When a dialog is generated starting from the post-reply pair, in some cases the tone changes.
A message that appears encouraging in the original pair becomes sarcastic in the reconstructed dialog, altering the evaluation of irony appropriateness.

**Example:**

* **id:** `12730524217281879155_12945194585296310579`
* **Generated dialog:**

  ```json
   {
      "u1":"Oh, I can relate. I've been trying to catch up on some games too. Maybe we should schedule a match soon?",
      "u2":"That sounds great! Let's do it. How about this weekend?"
   },
   {
      "u1":"@USER Congratulations Alex, plenty of game time is the key mate!  All the best… unleash ?",
      "u2":"@USER Thanks very much Rob. Hope you’re keeping well ??"
   }
  ```
* **Model reasoning:**

  ```
  1. We need to determine if u2 can respond ironically to u1’s last message. 
  2. u1's last message is '@USER Congratulations Alex, plenty of game time is the key mate!  All the best… unleash?' 
  3. The message is a congratulatory message but the tone is sarcastic as it is not a real congratulatory message and the phrase 'unleash' is not related to the context.
  4. Since the tone is sarcastic, it is appropriate for u2 to respond ironically.
  ```
* **Actual label:** `0`
* **Predicted label:** `1`

---

## 3. Errors in dialog generation

**Observed phenomenon:**
In some cases, dialog generation produces duplicated or defective messages.

**Example:**

* **id:** `17358410354806874318_16246648676958779322`
* **Generated dialog:**

  ```json
   {
      "u1":"u2: I've been looking for a good news source for local stories, what do you recommend?",
      "u2":"I've been looking for a good news source for local stories, what do you recommend?"
   },
   {
      "u1":"RTÉ is good for the basic news, Irish Times is good for analysis, Journal.ie is a a cesspit.",
      "u2":"The school in Dublin got leaked on thejournal, cesspit is an understatment, you cant post an article to r\/news it is such tripe"
   }
  ```
