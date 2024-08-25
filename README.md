# SIMPLE PDF RAG CHATBOT

Two implementations of a simple PDF RAG chatbot using Langchain and gpt-4o-mini, pdf_qa_with_history.py provides a chatbot with history and pdf_qa.py provides a chatbot with memory.

- openai api key is required to run the code
- langchain api key is required to run the code

both keys must be set as environment variables

### Example of running the chatbot with history, asking questions about the famouse 'attention is all you need' paper:

- Chatbot: Hello! I'm ready to answer your questions about the pdf document. Type 'exit' to end the conversation.
- You: what is the attention equation?
- Chatbot: The attention equation for Scaled Dot-Product Attention is given by:
Attention(Q, K, V) = softmax(QK^T / √dk) V
where Q represents the queries, K represents the keys, V represents the values, and dk is the dimension of the keys.
- You: is Q a scalar of a matrix?
- Chatbot: Q is a matrix, specifically a matrix of queries, where each row corresponds to a different query vector. The dimensions of Q depend on the number of queries and the dimensionality of each query vector.
- You: what is the purpose of dk?
- Chatbot: The purpose of dk, the dimension of the keys, is to scale the dot products of the queries and keys in the attention mechanism. By dividing the dot products by the square root of dk, it helps to prevent the softmax function from being pushed into regions with extremely small gradients, which can occur when the dot products grow large in magnitude. This scaling improves the stability and effectiveness of the attention mechanism during training.
- You: Give me a intuition about V
- Chatbot: In the context of the attention mechanism, V represents the values associated with the keys. Intuitively, while the queries (Q) are used to determine how much attention to pay to different keys (K), the values (V) are the actual information that gets passed along based on that attention. When the attention scores are calculated, they determine how much each value contributes to the final output, allowing the model to focus on the most relevant information for the given query.
- You: give me the email address of the authors of the paper
- Chatbot: The email addresses of the authors of the paper are:
Ashish Vaswani: avaswani@google.com
Noam Shazeer: noam@google.com
Niki Parmar: nikip@google.com
Jakob Uszkoreit: usz@google.com
Llion Jones: llion@google.com
Aidan N. Gomez: aidan@cs.toronto.edu
Łukasz Kaiser: lukaszkaiser@google.com
Illia Polosukhin: illia.polosukhin@gmail.com

- You: at which section of the paper they discuss about the optimer used?
- Chatbot: The optimizer used in the paper is discussed in Section 5.3 titled "Optimizer."
